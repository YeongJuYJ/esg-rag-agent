import os
import uuid
import logging
import asyncio
from typing import Dict, Optional, List, Any
import re
import json
import concurrent.futures

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

import requests
import vertexai
from vertexai.generative_models import GenerativeModel

from fastapi import FastAPI, HTTPException, Request, Response, Form, Depends, Cookie, APIRouter, Query, Body
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import text
from db import init_db, create_chat_session, log_chat, SessionLocal, User, LoginLog, ChatSession, ChatLog, Note, PromptConfig, get_prompt_config, upsert_prompt_config, delete_prompt_config
from db_vector import get_alloydb
from rag_utils import embed_query, build_rag_prompt, search_chunks_by_embedding

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

from huggingface_hub import login, HfFolder

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

compressor = CrossEncoderReranker(model=model, top_n=5)

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")     # 대용량 모델 빠른 전송
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf-cache")  # Cloud Run 임시 디스크 캐시

token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if token:
    try:
        login(token=token, add_to_git_credential=False)  # 토큰을 세션에 등록
        logging.info("[HF] token detected and loaded.")
    except Exception as e:
        logging.warning("[HF] login() failed: %s", e)
else:
    logging.warning("[HF] HUGGINGFACE_HUB_TOKEN not set.")

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

class QueryResponse(BaseModel):
    response: str
    session_id: str

class SQLRetriever(BaseRetriever):

    db: Session
    top_k: int = 5

    def get_relevant_documents(self, query: str) -> List[Document]:
        # 1) 질문 임베딩
        emb = embed_query(query)
        # 2) 수동 SQL 검색
        chunks = search_chunks_by_embedding(emb, self.db, top_k=self.top_k)
        # 3) langchain.Document 포맷으로 변환
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

@traceable
async def get_model_response(user_id: str, session_id: str, user_question: str, alloydb_session: Session) -> str:
    hist = get_user_chat_history(user_id, session_id)

    # '과거 대화를 요구하는' 질문인데 히스토리가 없거나 1개 이하면 안내문구
    if is_history_query(user_question) and len(hist.messages) <= 1:
        return "요약할 과거 대화가 없습니다. 먼저 대화를 시작해 주세요."

    paraphrased = fetch_paraphrases_for(user_question)
    all_questions = [user_question] + paraphrased if paraphrased else [user_question]

    logging.debug(f"[RAG] Original question: {user_question}")
    logging.debug(f"[RAG] Paraphrased questions: {all_questions[1:]}")

    # 전체 질문 리스트 로그
    for idx, q in enumerate(all_questions):
        kind = "원본" if idx == 0 else f"paraphrase_{idx}"
        logging.info(f"[RAG] 검색에 사용될 질문 ({kind}): {q}")

    retrieved_docs: List[Document] = []
    seen_texts = set()
    loop = asyncio.get_running_loop()

    # per-thread 세션 생성
    ThreadSession = sessionmaker(bind=alloydb_session.get_bind(), autoflush=False, autocommit=False)

    def query_task(q):
        with ThreadSession() as local_db:
            sql_retriever = SQLRetriever(db=local_db, top_k=5)
            try:
                docs = sql_retriever.get_relevant_documents(q)   # (invoke로 바꿔도 됨)
                logging.info(f"[RAG] 질문: {q[:100]} ... => 검색 결과 {len(docs)}건")
                return q, docs
            except Exception as e:
                logging.warning(f"[RAG] Retrieval failed for '{q}': {e}")
                return q, []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [loop.run_in_executor(executor, query_task, q) for q in all_questions]
        results = await asyncio.gather(*tasks)

    for q, docs in results:
        if q != user_question and len(docs) < 2:
            logging.debug(f"[RAG] Skipping paraphrased query due to insufficient results: {q}")
            continue
        for doc in docs:
            hashable = hash(doc.page_content)
            if hashable not in seen_texts:
                seen_texts.add(hashable)
                retrieved_docs.append(doc)

    logging.debug(f"[RAG] Total unique docs retrieved: {len(retrieved_docs)}")

    base_retriever = SQLRetriever(db=alloydb_session, top_k=20)

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor
    )

    compressed_docs = compression_retriever.invoke(user_question)

    for idx, doc in enumerate(compressed_docs, start=1):
        meta  = doc.metadata or {}
        title = meta.get("title") or meta.get("source_file_name") or "알 수 없음"
        page  = meta.get("page_number", "N/A")
        score = meta.get("score")
        logging.debug(f"[RERANK] Rank {idx}: title={title}, page={page}, score={score}")


    # 2. context_chunks 구성 (최대 5개 사용)
    context_chunks = []
    for doc in compressed_docs:
        meta   = doc.metadata or {}
        title  = meta.get("title") or meta.get("source_file_name") or "알 수 없음"
        page   = meta.get("page_number", "N/A")
        preview = doc.page_content.replace("\n", " ")[:80]
        logging.debug(f"[RAG] Using chunk: title={title}, page={page}, preview={preview}")

        cited_content = f"{doc.page_content.strip()} (출처: 제목={title}, 페이지={page})"
        context_chunks.append({
            "content": cited_content,
            "page_number": page,
            "block_type": meta.get("block_type"),
            "metadata": meta,
        })

    # 3. 프롬프트 구성
    get_user_chat_history(user_id, session_id)

    final_prompt = build_rag_prompt(context_chunks, user_question, history_messages=hist.messages)


    # 4. 응답 생성
    loop = asyncio.get_running_loop()
    resp = await loop.run_in_executor(
        None,
        lambda: text_model.generate_content(
            final_prompt,
            generation_config={"candidate_count": 1, "max_output_tokens": 2048, "temperature": 0.6}
        )
    )
    answer = resp.text
    logging.debug(f"[RAG] Final generated answer: {answer[:200]}...")

    # 5. 히스토리 저장
    hist.add_user_message(HumanMessage(content=user_question))
    hist.add_ai_message(AIMessage(content=answer))
    return answer

def generate_paraphrases(prompt: str, count: int = 3) -> list[str]:
    gemini_prompt = f"""
다음 문장을 의미는 유지하면서 다양한 표현으로 {count}개 바꿔줘.
- 오직 의역 문장만 번호와 함께 반환(예: 1. ~, 2. ~, 3. ~)
- 안내문, 설명, 추가 텍스트는 절대 넣지 말 것

문장: "{prompt}"
"""
    response = paraphrase_model.generate_content(gemini_prompt)
    lines = [line.strip() for line in response.text.strip().split("\n") if line.strip()]
    # 1. "1.", "2.", "3." 또는 "- "로 시작하는 줄만 남기기
    # 2. 또는, 첫 줄이 안내문이면 제외하고 2번째 줄부터 count개 추출
    filtered = []
    for line in lines:
        if re.match(r'^(\d+[.])|(-|\*) ', line):  # 1. ~ or - ~ or * ~
            filtered.append(line)
        elif len(filtered) == 0 and len(lines) > 1 and len(lines) <= (count+1):
            # 안내문 패턴이면 패스(첫 줄만 안내문이고, 나머지 3개면 사용)
            continue
    # count개만 리턴
    return filtered[:count]

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
async def query_model(req:QueryRequest, db:Session=Depends(get_db), alloydb:Session=Depends(get_alloydb)):
    uid, ques = req.user_id, "\n".join(m.content for m in req.messages if m.role=="user")
    sid = req.session_id or str(uuid.uuid4());
    if not req.session_id: create_chat_session(db, uid, ques, sid)
    ans = await get_model_response(uid, sid, ques, alloydb)
    log_chat(db, uid, sid, ques, ans)
    return QueryResponse(response=ans, session_id=sid)

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
async def logout(response:Response): response.delete_cookie("user_id"); return RedirectResponse(url="/login", status_code=302)
@app.get("/")
async def root(): return RedirectResponse(url="/chat")

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