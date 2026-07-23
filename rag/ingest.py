"""
ingest.py
Чанкинг + эмбеддинги данных из scraper/data/*/_index.json -> pgvector.

Каждый .md файл резан по "## вопрос" — один Q&A блок = один чанк.
Эмбеддинги — через OpenAI-совместимый API (OpenRouter), без локальной модели.
"""

import json
import os
import time
from pathlib import Path

import httpx
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "scraper" / "data"
# лимит модели ~4096 токенов суммарно на батч — берём чанки по символам
# с запасом (~3.5 символа/токен для рус+англ текста), чтобы не ловить 422
MAX_BATCH_CHARS = 12000
MAX_RETRIES = 6
EMBED_DIM = 2048 

EMBED_MODEL = os.environ["EMBED_MODEL"]
EMBEDDINGS_URL = os.environ["LLM_BASE_URL"].rstrip("/") + "/embeddings"
HEADERS = {"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"}


def embed_batch(texts: list[str]) -> list[list[float]]:
    for attempt in range(MAX_RETRIES):
        resp = httpx.post(
            EMBEDDINGS_URL,
            headers=HEADERS,
            json={"model": EMBED_MODEL, "input": texts, "encoding_format": "float"},
            timeout=30,
        )
        data = resp.json()

        if "data" in data:
            return [item["embedding"] for item in data["data"]]

        message = str(data.get("error", ""))
        if "exceeds model maximum" in message:
            if len(texts) == 1:
                return embed_batch([texts[0][:3000]])
            mid = len(texts) // 2
            return embed_batch(texts[:mid]) + embed_batch(texts[mid:])

        wait = 2 ** attempt
        print(f"    ошибка API ({message or resp.status_code}), попытка {attempt + 1}/{MAX_RETRIES}, жду {wait}с...")
        time.sleep(wait)
    raise RuntimeError("эмбеддинг-API не отвечает после всех попыток")


def chunks_from_markdown(md_text: str, title: str) -> list[str]:
    parts = md_text.split("\n## ")
    qa_blocks = parts[1:]
    return [f"Тема: {title}\n\n## {block}" for block in qa_blocks]


def load_all_chunks() -> list[tuple[str, str, str, str]]:
    """Возвращает список (chunk_key, текст_чанка, url, category).
    chunk_key = url + порядковый номер чанка внутри документа — стабильный
    уникальный ключ, по нему делаем resume при перезапуске."""
    result = []
    for index_file in DATA_DIR.glob("*/_index.json"):
        entries = json.loads(index_file.read_text(encoding="utf-8"))
        for entry in entries:
            md_path = index_file.parent / entry["file"]
            md_text = md_path.read_text(encoding="utf-8")
            for i, chunk_text in enumerate(chunks_from_markdown(md_text, entry["title"])):
                chunk_key = f"{entry['url']}#{i}"
                result.append((chunk_key, chunk_text, entry["url"], entry["category"]))
    return result


def make_batches(chunks: list[tuple[str, str, str, str]]) -> list[list[tuple[str, str, str, str]]]:
    """Группирует чанки в батчи так, чтобы суммарная длина текста не превышала лимит."""
    batches = []
    current: list[tuple[str, str, str, str]] = []
    current_chars = 0
    for chunk in chunks:
        chunk_len = len(chunk[1])
        if current and current_chars + chunk_len > MAX_BATCH_CHARS:
            batches.append(current)
            current, current_chars = [], 0
        current.append(chunk)
        current_chars += chunk_len
    if current:
        batches.append(current)
    return batches


def main():
    all_chunks = load_all_chunks()
    print(f"чанков всего: {len(all_chunks)}")

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            chunk_key TEXT UNIQUE,
            content TEXT,
            source_url TEXT,
            category TEXT,
            embedding VECTOR({EMBED_DIM})
        )
    """)
    conn.commit()

    cur.execute("SELECT chunk_key FROM chunks")
    already_done = {row[0] for row in cur.fetchall()}
    pending = [c for c in all_chunks if c[0] not in already_done]
    print(f"уже загружено: {len(already_done)}, осталось: {len(pending)}")

    if not pending:
        print("готово (нечего догружать)")
        return

    batches = make_batches(pending)
    print(f"батчей к отправке: {len(batches)}")

    done_count = 0
    for batch in batches:
        texts = [c[1] for c in batch]
        vectors = embed_batch(texts)

        for (chunk_key, content, url, category), vec in zip(batch, vectors):
            cur.execute(
                "INSERT INTO chunks (chunk_key, content, source_url, category, embedding) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (chunk_key) DO NOTHING",
                (chunk_key, content, url, category, vec),
            )
        conn.commit()
        done_count += len(batch)
        print(f"  загружено {done_count}/{len(pending)}")

    print("готово")


if __name__ == "__main__":
    main()
