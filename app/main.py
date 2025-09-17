import os
import uuid
import logging
import asyncio
import time
import hashlib
from typing import Dict, Optional, List, Any, Tuple
import re
import json
import concurrent.futures

from dotenv import load_dotenv
load_dotenv()

import requests
import vertexai
from vertexai.generative_models import GenerativeModel

from fastapi import FastAPI, HTTPException, Request, Response, Form, Depends, Cookie, APIRouter, Query, Body
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import text
from db import init_db, create_chat_session, log_chat, SessionLocal, User, LoginLog, ChatSession, ChatLog, Note, PromptConfig, get_prompt_config, upsert_prompt_config, delete_prompt_config
from db_vector import get_alloydb
from rag_utils import embed_query, build_rag_prompt, search_chunks_by_embedding, search_chunks_by_embedding_filtered

from langsmith import traceable
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema import BaseRetriever, Document

# LangChain base embedding interface
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Logging setup
logging.basicConfig(level=logging.DEBUG)

# GCP and model settings
PROJECT_ID = "hyperscale-ai-442809"
LOCATION = "us-central1"
MODEL_NAME = "gemini-1.5-flash-002"

# Initialize Vertex AI and load model
vertexai.init(project=PROJECT_ID, location=LOCATION)
text_model = GenerativeModel(model_name=MODEL_NAME)
paraphrase_model = GenerativeModel("gemini-1.5-flash-002")

# Custom Embeddings class wrapping existing embed_query
class GenAIEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return embed_query(text)

# Configure embedding model and vector store
embedding_model = GenAIEmbeddings()
CONNECTION_STRING = "postgresql+psycopg2://postgres:1q2w3e4r5t@10.20.0.2:5432/postgres"

vector_store = PGVector.from_existing_index(
    embedding=embedding_model,
    connection_string=CONNECTION_STRING,
    collection_name="chunks",      # 실제 테이블명
    id_column="id",                # 기본 키 컬럼
    vector_column="embedding",     # 벡터 저장 컬럼명
    content_column="content",      # 텍스트 콘텐츠 컬럼명
    metadata_column="metadata",    # JSONB 메타데이터 컬럼명
    distance_strategy="cosine"     # 인덱스 생성 시 사용한 유사도 메트릭
)

# Create LangChain retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Initialize FastAPI and templates
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# In-memory chat history
user_chat_histories: Dict[tuple, ChatMessageHistory] = {}

model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
# model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

compressor = CrossEncoderReranker(model=model, top_n=14)

def get_all_documents(db: Session):
    return db.execute("SELECT id, title, source_uri, created_at FROM documents ORDER BY created_at DESC").fetchall()

HISTORY_QUERY_KEYWORDS = [
    "요약", "방금", "이전", "아까", "되짚어", "다시", "대화 내역", "다시 말해", "요약해줘"
]

def is_history_query(user_question: str) -> bool:
    """과거 대화를 요구하는 질문인지 판별"""
    user_question = user_question.lower()
    return any(kw in user_question for kw in HISTORY_QUERY_KEYWORDS)

def get_user_chat_history(user_id: str, session_id: str) -> ChatMessageHistory:
    key = (user_id, session_id)
    if key not in user_chat_histories:
        user_chat_histories[key] = ChatMessageHistory()
    return user_chat_histories[key]

def fetch_paraphrases_for(question: str) -> List[str]:
    """
    주어진 question에 대해 generate_paraphrases로 3개의 paraphrase를 생성하여 반환
    실패 시 빈 리스트를 반환
    """
    try:
        paras = generate_paraphrases(question, count=3)
        return paras
    except Exception as e:
        logging.warning(f"[RAG] fetch_paraphrases_for failed for {question!r}: {e}")
        return []

class ParaphraseRequest(BaseModel):
    text: str
    count: int = 3

class ParaphraseResponse(BaseModel):
    paraphrases: List[str]

class NoteSaveReq(BaseModel):
    user_id: str
    content: str
    role: str
    session_id: str = None

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    messages: List[Message]
    filters: Optional[Dict[str, Any]] = None

class RetrievedContext(BaseModel):
    content: str
    page_number: Optional[int] = None
    block_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    title: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    session_id: str
    retrieved_contexts: List[RetrievedContext] = Field(default_factory=list)

class SQLRetriever(BaseRetriever):

    db: Session
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None

    def get_relevant_documents(self, query: str) -> List[Document]:
        # 1) 질문 임베딩
        emb = embed_query(query)

        logging.info("[FILTER] using filters=%s", self.filters)

        # 2) 수동 SQL 검색
#         chunks = search_chunks_by_embedding(emb, self.db, top_k=self.top_k)
        chunks = search_chunks_by_embedding_filtered(emb, self.db, top_k=self.top_k, filters=self.filters)
        # 3) langchain.Document 포맷으로 변환

    mismatches = 0
    if self.filters:
        for ch in chunks:
            md = ch.get("metadata") or {}
            if j := self.filters.get("jurisdiction"):
                mismatches += int((md.get("jurisdiction") != j))
            if l := self.filters.get("language"):
                mismatches += int((md.get("language") != l))
            if bt := self.filters.get("block_type"):
                b = md.get("block_type")
                if isinstance(bt, (list, tuple)):
                    mismatches += int(b not in bt)
                else:
                    mismatches += int(b != bt)
        if mismatches:
            logging.warning("[FILTER] %d mismatched rows detected under filters=%s", mismatches, self.filters)
        docs: List[Document] = []
        for chunk in chunks:
            md = chunk["metadata"] or {}
            # 메타데이터에 page_number 추가
            md["page_number"] = chunk.get("page_number")
            docs.append(
                Document(
                    page_content=chunk["content"],
                    metadata=md
                )
            )
        return docs

def check_langsmith_connection():
    try:
        res = requests.get(
            f"{os.getenv('LANGCHAIN_ENDPOINT')}/health",
            headers={"Authorization": f"Bearer {os.getenv('LANGCHAIN_API_KEY')}"}
        )
        logging.info("LangSmith status: %s", res.status_code)
    except Exception as e:
        logging.warning("LangSmith check failed: %s", e)

check_langsmith_connection()

# --- 경량 생성·개별 타임아웃·인메모리 TTL 캐시 유틸 ---

# 경량 생성 파라미터(지연/비용 상한)
PARA_MAX_TOKENS = 90
HYDE_MAX_TOKENS = 160
PARA_TIMEOUT = 1.5   # seconds
HYDE_TIMEOUT  = 2.0  # seconds

# TTL 캐시 (in-memory; 멀티 인스턴스면 추후 Redis 전환)
TTL_SECONDS = 6 * 60 * 60  # 6시간
_cache: Dict[str, tuple] = {}  # key -> (expire_ts: int, value: Any)

def _now() -> int:
    return int(time.time())

def _norm(s: str) -> str:
    # 공백/대소문자 정규화: 캐시 키 안정화
    return re.sub(r"\s+", " ", s.strip().lower())

def _ck(kind: str, q: str) -> str:
    # 내부용이므로 버전 태깅 없이 간단 키 사용
    return f"{kind}:{hashlib.sha1(_norm(q).encode()).hexdigest()}"

def cache_get(kind: str, q: str):
    key = _ck(kind, q)
    item = _cache.get(key)
    if not item:
        return None
    exp, val = item
    if exp < _now():
        _cache.pop(key, None)
        return None
    return val

def cache_set(kind: str, q: str, val):
    _cache[_ck(kind, q)] = (_now() + TTL_SECONDS, val)


# --- 경량 Paraphrase/HyDE 생성기 ---

def gen_paraphrases_light(q: str, n: int = 1) -> List[str]:
    """
    한 문장 paraphrase만 간결히 생성 (candidate=1, max_tokens 제한).
    """
    prompt = f"""다음 문장을 의미 유지해 {n}개, 한 문장씩 간결히 의역해줘.
- 번호/불릿 금지, 문장만 반환
문장: "{q}" """
    resp = paraphrase_model.generate_content(
        prompt,
        generation_config={
            "candidate_count": 1,
            "max_output_tokens": PARA_MAX_TOKENS,
            "temperature": 0.3
        }
    )
    text = (getattr(resp, "text", "") or "")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[:n]

def gen_hyde_light(q: str) -> str:
    """
    HyDE를 4~6문장으로 제한해 일반론/장문화 방지 + 지연 통제.
    """
    prompt = f"""질문에 대한 배경지식/핵심요지를 한국어로 4~6문장, 간결히 작성해줘.
- 제목/불릿 금지, 본문만
질문: {q}"""
    resp = paraphrase_model.generate_content(
        prompt,
        generation_config={
            "candidate_count": 1,
            "max_output_tokens": HYDE_MAX_TOKENS,
            "temperature": 0.3
        }
    )
    return (getattr(resp, "text", "") or "").strip()


# --- 동시 실행 + 타임아웃 + 캐시 + 폴백 래퍼 ---

async def get_light_queries(question: str, want_para: int = 1):
    """
    - 캐시 조회 → 없으면 경량 Paraphrase/HyDE를 '동시에' 생성(각각 타임아웃)
    - 성공분만 캐시 저장
    - 최종 질의: [원문] + optional(para1) + optional(hyde1)
    - 둘 다 실패/지연해도 '원문만'으로 즉시 진행(폴백)
    """
    cached_para = cache_get("para", question)
    cached_hyde = cache_get("hyde", question)

    async def run_para():
        if cached_para is not None:
            return cached_para
        return await asyncio.to_thread(gen_paraphrases_light, question, want_para)

    async def run_hyde():
        if cached_hyde is not None:
            return cached_hyde
        return await asyncio.to_thread(gen_hyde_light, question)

    tasks = [
        asyncio.wait_for(run_para(), timeout=PARA_TIMEOUT),
        asyncio.wait_for(run_hyde(), timeout=HYDE_TIMEOUT),
    ]

    paras: List[str] = []
    hyde: Optional[str] = None
    try:
        done, pending = await asyncio.wait(
            tasks, timeout=max(PARA_TIMEOUT, HYDE_TIMEOUT)
        )
        # 완료된 것만 반영
        for t in done:
            try:
                v = t.result()
                if isinstance(v, list):
                    paras = v or []
                else:
                    hyde = (v or "").strip() or None
            except Exception:
                logging.warning("[RAG] LightGen task failed/timeout", exc_info=False)
        # 남은 태스크 취소
        for p in pending:
            p.cancel()
        if pending:
            logging.debug(f"[RAG] cancelled {len(pending)} light-gen task(s)")
    except Exception:
        logging.warning("[RAG] LightGen wait failed", exc_info=False)

    # 성공분만 캐시
    if cached_para is None and paras:
        cache_set("para", question, paras)
    if cached_hyde is None and hyde:
        cache_set("hyde", question, hyde)

    # 폴백 포함 최종 질의 세트
    queries = [question]
    if paras:
        queries += paras[:1]  # Paraphrase 1개만 사용
    if hyde:
        queries += [hyde]     # HyDE 1개 사용
    return queries, {"para_cached": cached_para is not None, "hyde_cached": cached_hyde is not None}

@traceable
async def get_model_response(
    user_id: str,
    session_id: str,
    user_question: str,
    alloydb_session: Session,
    filters: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[Dict[str, Any]]]: # 반환 타입 변경

    # JSON 직렬화를 위한 유틸리티 함수 추가
    def _jsonable(o):
        try:
            return json.loads(json.dumps(o, default=str))
        except Exception:
            return str(o)

    hist = get_user_chat_history(user_id, session_id)

    if is_history_query(user_question) and len(hist.messages) <= 1:
        return "요약할 과거 대화가 없습니다. 먼저 대화를 시작해 주세요.", []

    # ---------------- 질의 생성부 (교체된 부분) ----------------
    all_questions, cache_info = await get_light_queries(user_question, want_para=1)

    logging.info(
        f"[RAG] queries={len(all_questions)} "
        f"(para_cached={cache_info['para_cached']}, hyde_cached={cache_info['hyde_cached']})"
    )
    for idx, q in enumerate(all_questions):
        kind = ["원문", "paraphrase", "HyDE"][idx] if idx < 3 else f"q{idx}"
        logging.info(f"[RAG] 검색 질의({kind}): {q[:120]}")
    # -----------------------------------------------------------

    retrieved_docs: List[Document] = []
    seen_texts = set()
    loop = asyncio.get_running_loop()

    ThreadSession = sessionmaker(bind=alloydb_session.get_bind(), autoflush=False, autocommit=False)

    def query_task(q):
        with ThreadSession() as local_db:
            sql_retriever = SQLRetriever(db=local_db, top_k=8, filters=filters)
            try:
                docs = sql_retriever.get_relevant_documents(q)
                logging.info(f"[RAG] 질문: {q[:100]} ... => 검색 결과 {len(docs)}건")
                return q, docs
            except Exception as e:
                logging.warning(f"[RAG] Retrieval failed for '{q}': {e}")
                return q, []

    # 동시 검색 실행
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [loop.run_in_executor(executor, query_task, q) for q in all_questions]
        results = await asyncio.gather(*tasks)

    for q, docs in results:
        if q != user_question and len(docs) < 2:
            logging.debug(f"[RAG] Skipping paraphrased/HyDE query due to insufficient results: {q}")
            continue
        for doc in docs:
            hashable = hash(doc.page_content)
            if hashable not in seen_texts:
                seen_texts.add(hashable)
                retrieved_docs.append(doc)

    logging.debug(f"[RAG] Total unique docs retrieved: {len(retrieved_docs)}")

    candidates = retrieved_docs[:]

    if len(candidates) < 14:
        booster = SQLRetriever(db=alloydb_session, top_k=60, filters=filters)
        more = booster.get_relevant_documents(user_question)
        seen = set(d.page_content for d in candidates)
        for d in more:
            if d.page_content not in seen:
                candidates.append(d); seen.add(d.page_content)

    def _clean_for_rerank(doc: Document) -> Document:
        textc = re.sub(r'\s+', ' ', doc.page_content).strip()
        textc = re.sub(r'\(출처:[^)]+\)\s*$', '', textc)
        textc = textc[:1800]
        return Document(page_content=textc, metadata=doc.metadata)

    candidates_clean = [_clean_for_rerank(d) for d in candidates]
    reranked = compressor.compress_documents(documents=candidates_clean, query=user_question)

    keep_k = 14
    compressed_docs = reranked[:keep_k]
    if len(compressed_docs) < keep_k:
        booster = SQLRetriever(db=alloydb_session, top_k=60, filters=filters)
        more = booster.get_relevant_documents(user_question)
        seen = set(d.page_content for d in compressed_docs)
        for d in more:
            if d.page_content not in seen:
                compressed_docs.append(d); seen.add(d.page_content)
                if len(compressed_docs) >= keep_k:
                    break

    for idx, doc in enumerate(compressed_docs, start=1):
        meta = doc.metadata or {}
        title = meta.get("title") or meta.get("source_file_name") or "알 수 없음"
        page = meta.get("page_number", "N/A")
        score = meta.get("relevance_score", meta.get("score"))
        logging.debug(f"[RERANK] Rank {idx}: title={title}, page={page}, score={score}")

    def _take_diverse_by_title(docs: List[Document], k: int = 8, per_title: int = 2):
        picked, seen_per = [], {}
        for d in docs:
            meta = d.metadata or {}
            title = meta.get("title") or meta.get("source_file_name") or "UNK"
            cnt = seen_per.get(title, 0)
            if cnt < per_title:
                picked.append(d); seen_per[title] = cnt + 1
            if len(picked) >= k: break
        return picked

    # context_chunks 구성
    context_chunks = []
    # 답변 생성에 사용될 최종 컨텍스트를 다양성 필터를 적용해 선택
    final_docs = _take_diverse_by_title(compressed_docs, k=8, per_title=2)
    for doc in final_docs:
        meta = doc.metadata or {}
        title = meta.get("title") or meta.get("source_file_name") or "알 수 없음"
        page = meta.get("page_number", "N/A")
        preview = doc.page_content.replace("\n", " ")[:80]
        logging.debug(f"[RAG] Using chunk: title={title}, page={page}, preview={preview}")

        cited_content = f"{doc.page_content.strip()} (출처: 제목={title}, 페이지={page})"
        context_chunks.append({
            "content": cited_content,
            "page_number": page,
            "block_type": meta.get("block_type"),
            "metadata": meta,
        })

    # 프롬프트 구성
    final_prompt = build_rag_prompt(context_chunks, user_question, history_messages=hist.messages)

    # 응답 생성
    loop = asyncio.get_running_loop()
    resp = await loop.run_in_executor(
        None,
        lambda: text_model.generate_content(
            final_prompt,
            generation_config={"candidate_count": 1, "max_output_tokens": 2048, "temperature": 0.35}
        )
    )
    answer = resp.text
    logging.debug(f"[RAG] Final generated answer: {answer[:200]}...")

    # 히스토리 저장
    hist.add_user_message(HumanMessage(content=user_question))
    hist.add_ai_message(AIMessage(content=answer))

    # 클라이언트로 내려줄 최종 컨텍스트(dict 리스트, 메타 포함)
    retrieved_contexts: List[Dict[str, Any]] = []
    for c in context_chunks:
        meta = c.get("metadata") or {}
        retrieved_contexts.append({
            "content": c.get("content", ""),
            "page_number": c.get("page_number"),
            "block_type": c.get("block_type"),
            "metadata": _jsonable(meta),
            "title": meta.get("title") or meta.get("source_file_name") or None,
        })

    return answer, retrieved_contexts


def generate_paraphrases(prompt: str, count: int = 3) -> list[str]:
    gemini_prompt = f"""
다음 문장을 의미는 유지하면서 다양한 표현으로 {count}개 바꿔줘.
- 오직 의역 문장만 번호와 함께 반환(예: 1. ~, 2. ~, 3. ~)
- 안내문, 설명, 추가 텍스트는 절대 넣지 말 것

문장: "{prompt}"
"""
    response = paraphrase_model.generate_content(gemini_prompt)
    lines = [line.strip() for line in response.text.strip().split("\n") if line.strip()]

    cleaned = []
    for line in lines:
        # 맨 앞 불릿/번호 제거: "1. ", "1) ", "- ", "* "
        text = re.sub(r'^\s*(?:\d+[.)]|[-*])\s*', '', line)
        if text:
            cleaned.append(text)
    return cleaned[:count]

# DB session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Startup event
@app.on_event("startup")
def on_startup():
    init_db()

# Auth & chat endpoints
@app.get("/login", response_class=HTMLResponse)
async def get_login(request: Request):
    return templates.TemplateResponse("login.html", {"request":request, "error":""})

@app.post("/login", response_class=HTMLResponse)
async def post_login(response: Response, request: Request, user_id: str = Form(...), password: str = Form(...), db: Session=Depends(get_db)):
    user = db.query(User).filter(User.user_id==user_id).first()
    if not user or user.password!=password:
        return templates.TemplateResponse("login.html", {"request":request, "error":"Invalid credentials"})
    db.add(LoginLog(user_id=user_id, name=user.name)); db.commit()
    resp = RedirectResponse(url="/chat", status_code=302)
    resp.set_cookie(key="user_id", value=user_id)
    return resp

@app.get("/register", response_class=HTMLResponse)
async def get_register(request: Request):
    return templates.TemplateResponse("register.html", {"request":request, "error":""})

@app.post("/register", response_class=HTMLResponse)
async def post_register(response: Response, request: Request, user_id: str=Form(...), password: str=Form(...), name: str=Form(...), email: str=Form(...), affiliation: str=Form(...), db: Session=Depends(get_db)):
    if db.query(User).filter(User.user_id==user_id).first():
        return templates.TemplateResponse("register.html", {"request":request, "error":"User exists"})
    db.add(User(user_id=user_id, password=password, name=name, email=email, affiliation=affiliation)); db.commit()
    return RedirectResponse(url="/login", status_code=302)

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(
    request: Request,
    user_id: Optional[str] = Cookie(None),
    cloudsql_db: Session = Depends(get_db),
    alloydb: Session = Depends(get_alloydb),
):
    if not user_id:
        return RedirectResponse(url="/login")
    # 채팅 세션은 CloudSQL에서 조회
    sessions = cloudsql_db.query(ChatSession).filter(ChatSession.user_id == user_id).order_by(ChatSession.created_at.desc()).all()
    # 소스 파일 목록은 AlloyDB에서 조회
    result = alloydb.execute(text(
        "SELECT id, title, source_uri, created_at FROM documents ORDER BY created_at DESC"
    ))
    source_files = [
        {
            "id": row.id,
            "title": row.title,
            "source_uri": row.source_uri,
            "created_at": row.created_at,
        }
        for row in result
    ]
    notes = cloudsql_db.execute(
        text("SELECT id, content, role, created_at FROM notes WHERE user_id=:uid ORDER BY created_at DESC"),
        {"uid": user_id}
    ).fetchall()
    notes_data = [
        {
            "id": n.id,
            "content": n.content,
            "role": n.role,
            "created_at": n.created_at,
        }
        for n in notes
    ]
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "chat_sessions": sessions,
            "user_id": user_id,
            "source_files": source_files,
            "notes": notes_data,
        },
    )


@app.post("/query", response_model=QueryResponse)
async def query_model(
    req: QueryRequest,
    db: Session = Depends(get_db),
    alloydb: Session = Depends(get_alloydb),
):
    uid = req.user_id
    ques = "\n".join(m.content for m in req.messages if m.role == "user")
    sid = req.session_id or str(uuid.uuid4())

    if not req.session_id:
        create_chat_session(db, uid, ques, sid)

    ans, ctxs = await get_model_response(uid, sid, ques, alloydb, filters=req.filters)
    log_chat(db, uid, sid, ques, ans)

    return QueryResponse(
        response=ans,
        session_id=sid,
        retrieved_contexts=ctxs
    )

@app.post("/load_history")
async def load_history(payload: Dict[str,str], db:Session=Depends(get_db)):
    uid, sid = payload.get("user_id"), payload.get("session_id")
    logs = db.query(ChatLog).filter(ChatLog.user_id==uid, ChatLog.session_id==sid).order_by(ChatLog.created_at.asc()).all()
    msgs = []
    for l in logs:
        msgs += [{"role":"user", "content":l.question}, {"role":"ai", "content":l.answer}]
    hist = get_user_chat_history(uid, sid)
    hist.messages.clear()
    for m in msgs:
        if m["role"] == "user":
            hist.add_user_message(HumanMessage(content=m["content"]))
        else:
            hist.add_ai_message(AIMessage(content=m["content"]))
    return {"messages":msgs, "session_id":sid}

@app.post("/reset_history")
async def reset_history(payload: Dict[str,str]):
    uid = payload.get("user_id")
    sid = payload.get("session_id")
    get_user_chat_history(uid, sid).messages.clear()
    return {"message":"History cleared"}

@app.get("/logout")
async def logout(response:Response):
    response.delete_cookie("user_id")
    return RedirectResponse(url="/login", status_code=302)

@app.get("/")
async def root():
    return RedirectResponse(url="/chat")

@app.get("/debug_search")
async def debug_search(alloydb: Session = Depends(get_alloydb)):
    question = "자산 배분에 대해 설명해줘"
    # 1) 질문 임베딩
    embedding = embed_query(question)
    # 2) 수동 SQL 검색
    chunks = search_chunks_by_embedding(embedding, alloydb, top_k=5)
    # 3) 결과 반환
    return {
        "query": question,
        "returned_count": len(chunks),
        "examples": [
            {
                "title": c["metadata"].get("source_file_name"),
                "page": c["page_number"],
                "snippet": c["content"][:100]  # 앞 100자만 예시로
            }
            for c in chunks
        ]
    }


@app.get("/debug_retriever")
async def debug_retriever(alloydb: Session = Depends(get_alloydb)):
    question = "자산 배분에 대해 설명해줘"
    # PGVector 대신 직접 SQLRetriever 사용
    sql_retriever = SQLRetriever(db=alloydb, top_k=5)
    docs = sql_retriever.get_relevant_documents(question)
    return {
        "query": question,
        "returned_count": len(docs),
        "examples": [
            {
                "title": d.metadata.get("title") or d.metadata.get("source_file_name"),
                "page": d.metadata.get("page_number"),
                "snippet": d.page_content[:100],
            }
            for d in docs
        ]
    }

@app.get("/debug_paraphrase")
async def debug_paraphrase():
    question = "자산 배분에 대해 알려줘"
    examples = generate_paraphrases(question, count=3)
    return {"original": question, "paraphrases": examples}

@app.post("/save_note")
async def save_note_api(req: NoteSaveReq, db: Session = Depends(get_db)):
    note = Note(
        user_id=req.user_id,
        content=req.content,
        role=req.role,
        session_id=req.session_id,
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    return {"success": True, "note_id": note.id}


@app.get("/get_notes")
async def get_notes(user_id: str, db: Session = Depends(get_db)):
    notes = db.execute(
        text("SELECT id, content, role, created_at FROM notes WHERE user_id=:uid ORDER BY created_at DESC"),
        {"uid": user_id}
    ).fetchall()
    notes_data = [
        {
            "id": n.id,
            "content": n.content,
            "role": n.role,
            "created_at": n.created_at.isoformat() if hasattr(n.created_at, 'isoformat') else str(n.created_at),
        }
        for n in notes
    ]
    return JSONResponse(content={"notes": notes_data})

@app.post("/delete_note")
async def delete_note(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    note_id = data.get("note_id")
    if not note_id:
        return JSONResponse({"success": False, "error": "No note_id"}, status_code=400)
    note = db.query(Note).filter(Note.id == note_id).first()
    if note:
        db.delete(note)
        db.commit()
        return {"success": True}
    else:
        return {"success": False, "error": "Note not found"}

@app.get("/api/prompts")
async def get_prompts(user_id: str = Query(...), db: Session = Depends(get_db)):
    # PromptConfig 테이블에서 해당 user_id의 모든 프롬프트 반환
    prompts = db.query(PromptConfig).filter(PromptConfig.user_id == user_id).all()
    result = [
        {"prompt_type": p.prompt_type, "content": p.content}
        for p in prompts
    ]
    return {"prompts": result}


@app.post("/paraphrase", response_model=ParaphraseResponse)
async def paraphrase(req: ParaphraseRequest):
    """
    retrieve 없이, 지정된 count 만큼 paraphrase_model 호출만 해서
    딱 그 수만큼의 의역문을 돌려줍니다.
    """
    # 스레드풀에 오프로딩
    paras = await asyncio.to_thread(generate_paraphrases, req.text, req.count)

    paras = paras[: req.count]
    return ParaphraseResponse(paraphrases=paras)
