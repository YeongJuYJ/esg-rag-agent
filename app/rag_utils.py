from typing import List, Dict, Any
from sqlalchemy.orm import Session
from google import genai
from google.genai.types import EmbedContentConfig
from sqlalchemy import text
from langchain_core.messages import BaseMessage
from langchain_core.messages import get_buffer_string

PROJECT_ID = "hyperscale-ai-442809"
LOCATION = "us-central1"

# Vertex AI 임베딩 클라이언트
def get_embedding_client():
    return genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# 쿼리 임베딩
def embed_query(text: str) -> List[float]:
    client = get_embedding_client()
    response = client.models.embed_content(
        model="text-multilingual-embedding-002",
        contents=[text],
        config=EmbedContentConfig(output_dimensionality=768)
    )
    return response.embeddings[0].values

def search_chunks_by_embedding(
    embedding: List[float],
    db: Session,
    top_k: int = 5,
    similarity_threshold: float = 0.7,  # 완화된 임계값
    enable_fallback: bool = True
) -> List[Dict[str, Any]]:
    import logging
    vals = ",".join(str(v) for v in embedding)
    distance_threshold = 1 - similarity_threshold

    # 1차: 유사도 필터링 쿼리
    sql = text(f"""
        SELECT content, page_number, block_type, metadata,
               embedding <-> '[{vals}]'::vector AS score
        FROM chunks
        WHERE embedding <-> '[{vals}]'::vector <= {distance_threshold}
        ORDER BY score
        LIMIT {top_k}
    """)
    result = db.execute(sql).fetchall()

    # fallback: 필터 결과가 너무 적은 경우 top-k만 사용
    if enable_fallback and len(result) < 3:
        logging.warning(f"[RAG] Only {len(result)} chunks passed threshold ({similarity_threshold}). Falling back to top-{top_k}.")
        sql = text(f"""
            SELECT content, page_number, block_type, metadata,
                   embedding <-> '[{vals}]'::vector AS score
            FROM chunks
            ORDER BY score
            LIMIT {top_k}
        """)
        result = db.execute(sql).fetchall()

    # 반환 및 디버깅
    chunks: List[Dict[str, Any]] = []
    logging.debug(f"[RAG] Retrieved {len(result)} chunks (similarity ≥ {similarity_threshold})")
    for i, row in enumerate(result, 1):
        title = row[3].get("title") or row[3].get("source_file_name") or "알 수 없음"
        preview = row[0].replace("\n", " ")[:80]
        logging.debug(f"[RAG] Chunk#{i}: title={title}, page={row[1]}, preview={preview}")
        chunks.append({
            "content": row[0],
            "page_number": row[1],
            "block_type": row[2],
            "metadata": row[3],
        })
    return chunks


def build_rag_prompt(
    context_chunks: List[Dict[str, Any]],
    user_question: str,
    history_messages: List[BaseMessage] = None
) -> str:
    from langchain_core.messages import get_buffer_string

    history_text = get_buffer_string(history_messages) if history_messages else ""

    # 영어 few-shot 예시 추가
    # fewshot_examples = """
    # 질문: 분쟁영향 및 고위험 지역(CAHRA) 목록의 역할에 대한 혼란이 발생하는 이유는 무엇인가요?
    # Question: Why is there confusion about the role of the list of Conflict-Affected and High-Risk Areas (CAHRAs)?
    # 답변: 규정에 따르면, 목록에 언급되지 않은 지역에서 광물을 조달하는 EU 수입업자도 실사 의무를 준수할 책임이 있으므로, 목록의 목적이 모호해집니다.
    # Answer: According to the regulation, EU importers are also responsible for due diligence even when sourcing minerals from areas not mentioned in the list, which creates ambiguity in the list's purpose.

    # 질문: 산업 주도 실사 체계 및 '화이트리스트'의 사용은 어떤 위험을 초래할 수 있나요?
    # Question: What risks can arise from the use of industry-led due diligence systems and 'whitelists'?
    # 답변: EU 수입업자가 실사 체계의 일부이거나 화이트리스트에 등재되어 있더라도 개별 실사 의무를 준수할 책임이 있으므로, 이들이 자동으로 규정의 모든 요구사항을 충족한다고 가정할 수 없으며, 이는 규정의 주요 목표를 훼손할 위험이 있습니다.
    # Answer: Even if EU importers are part of due diligence systems or appear on a whitelist, they remain individually responsible for compliance. Assuming automatic compliance risks undermining the regulation's key goals.

    # 질문: 폐기물 재활용 비율은 어떻게 측정되나요?
    # Question: How is the waste recycling rate measured?
    # 답변: 폐기물 재활용 비율은 총 재생 폐기물 재활용량(재사용 포함)을 재활용(재사용 포함) 대상 폐기물 발생량으로 나누어 계산합니다.
    # Answer: The waste recycling rate is calculated by dividing the total amount of recycled waste (including reuse) by the total amount of recyclable (including reusable) waste generated.
    # """

    format_instructions = """
    # [역할]
    너는 복잡한 정보를 명확하고 구조적으로 정리하는 전문 분석가야. 전문적이지만 이해하기 쉬운 어조를 사용해야 해.

    # [작업 지침]
    아래 요청에 대해 답변을 생성할 때, 다음 사고 흐름과 출력 형식을 반드시 따라줘.

    ### 사고 흐름 (Chain-of-Thought)
    1.  사용자 요청의 핵심 주제를 파악한다.
    2.  주제에 대한 가장 중요한 결론이나 요점을 먼저 생각한다.
    3.  그 요점을 뒷받침할 세부 항목들을 정리한다.
    4.  참고할 만한 예시나 주의점이 있는지 고려한다.
    5.  아래 "출력 형식"에 맞춰 단계적으로 내용을 구성한다.

    ### 출력 형식 (Markdown)
    - **헤더**: 각 주제는 `##` 또는 `###` 헤더로 구분해줘.
    - **리스트**: 세부 항목이나 중요 포인트는 `-` 또는 `*`를 사용해서 글머리 기호 목록으로 만들어줘.
    - **인용구**: 예시, 참고사항, 주의점은 `>`를 사용해서 강조해줘.
    - **표/코드**: 필요하다면 표(`|`)나 코드 블록(```)을 자유롭게 사용해도 좋아.
    - **간결성**: 불필요한 미사여구나 서론 없이 핵심 정보만 명확하게 전달해줘.
    - **문단 구분**: 내용의 흐름에 맞게 문단을 자연스럽게 나눠줘.
    - **참조 표기**: 문서에서 인용하거나 참고한 내용 뒤에는 반드시 `(출처: 제목={title}, 페이지={page})` 형식으로 표시해줘. 예를 들면  
      ```
      해당 규칙은 엄격합니다. (출처: Sustainability Report 2024, 페이지=63)
      ```

    ### 제약 조건
    - 답변을 "네, 요청하신 내용에 대해 답변해 드리겠습니다." 와 같은 문장으로 시작하지 마.
    - 각 글머리 기호 항목은 가급적 2~3 문장 이내로 간결하게 작성해.

    ---
    ### 샘플 예시
    ## CAHRA 목록의 혼란
    - CAHRA 목록에 포함되지 않은 국가(예: 스위스, 미국 등)도 고위험 지역에서 광물을 수입할 수 있음
    - 실사 의무 적용 범위 혼동
    > 참고: CAHRA 미포함 국가 사례

    | 구분 | 내용 |
    |---|---|
    | CAHRA 국가 | 분쟁/고위험지역 포함 |
    | 비CAHRA 국가 | 우회수입, 실사 의무 주의 |
    ---
    """

    fewshot_examples = """
    질문: EU 시장에서 에너지 관련 제품이 유통되려면 어떤 주요 인증 요건을 충족해야 하나요?
    Question: What are the key certification requirements for energy-related products to be marketed in the EU?
    답변:
    ## EU 에너지 관련 제품의 핵심 인증 요건

    에너지 관련 제품이 EU 시장에 출시되고 자유롭게 유통되려면 다음의 주요 요건을 충족해야 합니다.

    - **'CE' 마크 부착**: 제품이 EU 지침의 필수 요구사항을 모두 충족했음을 의미하는 'CE' 마크를 반드시 부착해야 합니다. (출처: EU_energy labelling regulation, 페이지=10) 이는 제조업체의 자기 선언에 기반합니다.
    - **적합성 추정 (Presumption of Conformity)**: EU의 조화된 표준(harmonised standard)에 따라 제품을 시험하고 제조한 경우, 해당 제품은 자동으로 지침의 요구사항을 충족하는 것으로 '추정'됩니다. 이는 인증 절차를 간소화하는 중요한 혜택입니다.

    > **참고**: 'CE' 마크는 제품의 안전, 건강, 환경 보호와 관련된 요구사항을 충족했다는 의미이며, 품질 보증 마크는 아닙니다.

    Answer:
    ## Key Certification Requirements for EU Energy-Related Products

    For energy-related products to be placed on the EU market and circulate freely, they must meet the following key requirements.

    - **'CE' Marking**: Products must bear the 'CE' mark, signifying they meet all essential requirements of EU directives. (Source: EU_energy labelling regulation, page=10) This is based on the manufacturer's self-declaration.
    - **Presumption of Conformity**: If a product is tested and manufactured according to EU's harmonised standards, it is 'presumed' to automatically conform to the directive's requirements. This is a significant benefit that simplifies the certification process.

    > **Note**: The 'CE' mark indicates compliance with safety, health, and environmental protection requirements; it is not a quality mark.

    ---
    질문: 미국 소비자제품안전위원회(CPSC)는 어떤 유형의 소비자 제품을 '위험 제품'으로 지정하여 금지할 수 있으며, 구체적인 사례는 무엇인가요?
    Question: What types of consumer products can the U.S. Consumer Product Safety Commission (CPSC) ban as 'hazardous products,' and what are some specific examples?
    답변:
    ## CPSC의 위험 제품 지정 및 금지 권한

    미국 소비자제품안전위원회(CPSC)는 특정 조건 하에 소비자 제품을 '위험 제품'으로 지정하고 시장에서 퇴출시킬 수 있습니다.
    (출처: 미국_소비자제품안전_번역본(국회도서관), 페이지=103)

    ### 금지 조건
    - 제품이 **'불합리한 상해 위험(unreasonable risk of injury)'**을 유발한다고 판단될 경우
    - 기존의 다른 안전 기준으로는 대중을 충분히 보호할 수 없다고 판단될 경우

    ### 실제 규제 및 금지 사례
    - 버튼 셀 또는 코인 배터리를 포함한 제품
    - 특정 프탈레이트가 함유된 어린이 장난감
    - 유아용 경사형 수면기 (Infant inclined sleepers)
    - 아기 침대 범퍼 (Crib bumpers)

    Answer:
    ## CPSC's Authority to Designate and Ban Hazardous Products

    The U.S. Consumer Product Safety Commission (CPSC) can designate consumer products as 'hazardous products' and remove them from the market under specific conditions.
    (Source: 미국_소비자제품안전_번역본(국회도서관), page=103)

    ### Conditions for a Ban
    - If the product is found to pose an **'unreasonable risk of injury'**
    - If existing safety standards are deemed insufficient to adequately protect the public

    ### Examples of Regulated and Banned Products
    - Products containing button cell or coin batteries
    - Children's toys containing certain phthalates
    - Infant inclined sleepers
    - Crib bumpers

    ---
    질문: 일본 대기오염방지법에서 대기 오염으로 인해 피해가 발생했을 경우, 사업자의 책임은 어떻게 규정되나요?
    Question: Under Japan's Air Pollution Control Act, how is a business operator's liability defined in cases of harm caused by air pollution?
    답변:
    ## 일본 대기오염방지법상 사업자 책임: 무과실 책임 원칙

    일본 대기오염방지법은 대기 오염으로 인한 건강 피해 발생 시, 피해자 보호를 위해 사업자에게 매우 엄격한 책임을 부과합니다.
    (출처: 일본_대기오염방지법_영문본(2017.06.02), 페이지=45)

    - **핵심 원칙**: 사업자는 자신의 활동으로 배출된 오염 물질로 인해 타인의 건강에 피해가 발생하면, **고의나 과실이 없더라도** 손해를 배상할 책임을 집니다. 이를 '무과실 책임' 또는 '엄격 책임'이라고 합니다.
    - **입증 책임**: 피해자는 사업자의 과실을 입증할 필요 없이, 오염 물질 배출과 자신의 건강 피해 사이의 인과관계만 입증하면 됩니다.

    > **주의: 적용 예외**
    > 이 무과실 책임 조항은 사업 활동 과정에서 발생한 근로자의 직무상 부상, 질병 또는 사망에는 적용되지 않습니다. 이 경우는 산업재해보상보험법 등 다른 법률이 적용됩니다.

    Answer:
    ## Business Operator Liability under Japan's Air Pollution Control Act: Strict Liability Principle

    Japan's Air Pollution Control Act imposes very strict liability on business operators to protect victims in cases of health damage caused by air pollution.
    (Source: 일본_대기오염방지법_영문본(2017.06.02), page=45)

    - **Core Principle**: A business operator is liable for damages if their emitted pollutants cause harm to another person's health, **even without intent or negligence**. This is known as 'strict liability' or 'no-fault liability'.
    - **Burden of Proof**: The victim only needs to prove a causal link between the pollutant emission and their health damage, without needing to prove the operator's negligence.

    > **Note: Exception to Application**
    > This strict liability provision does not apply to occupational injuries, illnesses, or deaths of workers that occur during business activities. Those cases are covered by other laws, such as the Industrial Accident Compensation Insurance Act.
    """

#     if context_chunks:
#         context_text = "\n\n".join(
#             f"(출처: 제목={chunk['metadata'].get('title', chunk['metadata'].get('source_file_name', '알 수 없음'))}, "
#             f"페이지={chunk.get('page_number', '?')})\n{chunk['content']}"
#             for chunk in context_chunks
#         )

    if context_chunks:
        context_text = "\n\n".join(chunk["content"] for chunk in context_chunks)

        prompt = f"""너는 ESG 문서를 참고하여 사용자 질문에 답변하는 AI야.

You are an AI assistant that answers questions based on ESG documents.

- 문서에 관련된 정보가 있다면 해당 내용을 바탕으로 답변해.
  If relevant information exists in the documents, base your answer on it.

- 답변할 때 참고한 문서의 제목, 페이지 번호를 반드시 포함해.
  When referencing, always include the document title and page number.

- 문서에도 없고 너의 지식으로도 확실하지 않은 경우에는 사실이 아닌 내용을 지어내지 말고,
  "제공된 문서나 내 지식으로는 정확히 알 수 없습니다."라고 정직하게 답변해줘.
  If the answer is unclear from both the documents and your knowledge, do not fabricate an answer. Respond honestly with "The provided documents or my knowledge do not allow me to answer accurately."

{format_instructions}
{fewshot_examples}

[이전 대화]
{history_text}

[참고 문서]
{context_text}

[사용자 질문]
{user_question}

[답변]
"""
    else:
        prompt = f"""너는 ESG 분야 AI 전문가야.
You are an AI expert in the field of ESG.

- 제공된 문서에 관련된 정보가 없으므로, 너의 지식을 바탕으로 답변해줘.
  Since no document is provided, answer using your own knowledge.

- 답변 시에는 너의 지식을 기반으로 한다는 점을 명시해줘.
  Please indicate that your answer is based on your general knowledge.

- 확실하지 않은 부분은 "정확히 알 수 없습니다."라고 답해줘.
  For uncertain parts, simply reply, "I cannot say for sure."

{format_instructions}
{fewshot_examples}

[이전 대화]
{history_text}

[사용자 질문]
{user_question}

[답변]
"""
    return prompt
