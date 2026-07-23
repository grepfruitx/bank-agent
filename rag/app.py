"""
app.py
FastAPI /chat: retrieval (pgvector) + generation (Claude) + базовые guardrails.
"""

import os
import re

import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()
llm = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
LLM_MODEL = os.environ["LLM_MODEL"]
EMBED_MODEL = os.environ["EMBED_MODEL"]

# управление reasoning-моделями (OpenRouter unified reasoning API) — оба необязательные,
# если не заданы в .env, параметр вообще не отправляется (обычные модели его просто не увидят)
LLM_REASONING_ENABLED = os.environ.get("LLM_REASONING_ENABLED")  # "true" / "false"
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT")  # "low" / "medium" / "high"


def build_reasoning_param() -> dict | None:
    reasoning = {}
    if LLM_REASONING_ENABLED is not None:
        reasoning["enabled"] = LLM_REASONING_ENABLED.lower() == "true"
    if LLM_REASONING_EFFORT:
        reasoning["effort"] = LLM_REASONING_EFFORT
    return reasoning or None

DISTANCE_THRESHOLD = 1.1
TOP_K = 5

INJECTION_MARKERS = ["ignore previous instructions", "забудь инструкции", "ты теперь", "system:", "system прompt"]

# бесплатные модели иногда глючат и вставляют случайные символы из левых
# алфавитов (бенгальский, деванагари, CJK и т.п.) — для русскоязычного
# ассистента это верный признак брака в ответе, а не легитимный текст
FOREIGN_SCRIPT_RE = re.compile(
    r"[ऀ-ॿঀ-৿؀-ۿ一-鿿぀-ヿ가-힯]"
)


def looks_corrupted(text: str) -> bool:
    return bool(FOREIGN_SCRIPT_RE.search(text))


def get_conn():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def looks_like_injection(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in INJECTION_MARKERS)


def dedup_key(content: str) -> str:
    # у разных карт (Black / Black Premium и т.п.) один и тот же вопрос
    # часто дублируется почти дословно — дедупим по тексту "## вопрос"
    match = re.search(r"##\s*(.+)", content)
    question = match.group(1) if match else content
    return question.strip().lower()


def search(query: str, top_k: int = TOP_K) -> list[str] | None:
    q_vec = llm.embeddings.create(model=EMBED_MODEL, input=query, encoding_format="float").data[0].embedding
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT content, embedding <-> %s::vector AS distance FROM chunks ORDER BY distance LIMIT %s",
        (q_vec, top_k * 3),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows or rows[0][1] > DISTANCE_THRESHOLD:
        return None

    seen = set()
    result = []
    for content, _distance in rows:
        key = dedup_key(content)
        if key in seen:
            continue
        seen.add(key)
        result.append(content)
        if len(result) >= top_k:
            break
    return result


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
    if looks_like_injection(req.question):
        return {"answer": "Не могу выполнить этот запрос."}

    context_chunks = search(req.question)
    if context_chunks is None:
        return {"answer": "Не нашёл информации по этому вопросу в базе знаний Т-Банка."}

    context = "\n\n---\n\n".join(context_chunks)
    messages = [
        {
            "role": "system",
            "content": (
                "Ты — ассистент поддержки по продуктам Т-Банка. Отвечай ТОЛЬКО на основе "
                f"контекста ниже. Если ответа нет в контексте — скажи, что не знаешь.\n\nКонтекст:\n{context}"
            ),
        },
        {"role": "user", "content": req.question},
    ]

    extra_body = {}
    reasoning_param = build_reasoning_param()
    if reasoning_param is not None:
        extra_body["reasoning"] = reasoning_param

    answer = None
    for _ in range(2):
        response = llm.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=1024,
            temperature=0.2,
            messages=messages,
            extra_body=extra_body,
        )
        answer = response.choices[0].message.content
        if not looks_corrupted(answer):
            break

    return {"answer": answer}


@app.get("/health")
def health():
    return {"status": "ok"}
