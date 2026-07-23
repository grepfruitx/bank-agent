"""
tbank_scraper.py
Рекурсивный скрейпер справочного центра Т-Банка (розничный банкинг /bank/help/)

Логика: глубина вложенности категорий непостоянна (2-4 уровня).
Поэтому вместо жёсткой схемы URL используем рекурсивный обход:
- заходим на страницу
- если на ней есть <article> теги — это конечная страница с контентом,
  парсим и сохраняем
- если <article> нет — это промежуточная страница-список
  (question-container/li), собираем ссылки глубже текущего URL
  и рекурсивно заходим в каждую
"""

import requests
from bs4 import BeautifulSoup
import time
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse
import json

BASE_URL = "https://www.tbank.ru"

SITES = {
    "bank": ("https://www.tbank.ru/bank/help/", Path("data/tbank_help_articles")),
    "business": ("https://www.tbank.ru/business/help/", Path("data/tbank_business_help_articles")),
    "insurance": ("https://www.tbank.ru/insurance/help/", Path("data/tbank_insurance_help_articles")),
}

SITE_KEY = sys.argv[1] if len(sys.argv) > 1 else "bank"
if SITE_KEY not in SITES:
    raise SystemExit(f"неизвестный сайт '{SITE_KEY}', доступно: {', '.join(SITES)}")
START_URL, OUTPUT_DIR = SITES[SITE_KEY]

DELAY_SECONDS = 1.5
MAX_DEPTH = 6
HEADERS = {
    "User-Agent": "Mozilla/5.0 (educational RAG project; contact: your-email@example.com)"
}

visited_urls: set[str] = set()
all_articles_meta: list[dict] = []


def get_soup(url: str) -> tuple[BeautifulSoup, str]:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    time.sleep(DELAY_SECONDS)
    return BeautifulSoup(resp.text, "html.parser"), clean_url(resp.url)


def clean_url(href: str) -> str:
    full_url = urljoin(BASE_URL, href)
    parsed = urlparse(full_url)
    result = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if not result.endswith("/"):
        result += "/"
    return result


def get_subcategory_urls() -> list[str]:
    soup, _ = get_soup(START_URL)
    links = set()
    for a in soup.find_all("a", href=True):
        url = clean_url(a["href"])
        if url != START_URL and url.startswith(START_URL) and re.match(r"^[a-z0-9\-/]*$", url[len(START_URL):-1]):
            links.add(url)
    return sorted(links)


def parse_article_page(url: str, soup: BeautifulSoup, category: str) -> None:
    """Страница содержит один или несколько <article> блоков — сохраняем"""
    h1_tag = soup.find("h1")
    page_title = h1_tag.get_text(strip=True) if h1_tag else None

    articles = soup.find_all("article")
    qa_blocks = []
    first_question = None

    for art in articles:
        for tag in art.find_all(["script", "style"]):
            tag.decompose()

        question_tag = art.find("h2")
        question = question_tag.get_text(strip=True) if question_tag else None
        if question and first_question is None:
            first_question = question

        paragraphs = []
        for el in art.find_all(["p", "li"]):
            text = el.get_text(strip=True)
            if not text:
                continue
            paragraphs.append(f"- {text}" if el.name == "li" else text)

        body = "\n".join(paragraphs)
        qa_blocks.append(f"\n## {question}\n\n{body}" if question else body)

    title = page_title or first_question or url
    markdown = f"# {title}\n\n" + "\n".join(qa_blocks)

    filename = "-".join(urlparse(url).path.strip("/").split("/")) + ".md"

    filepath = OUTPUT_DIR / filename
    filepath.write_text(markdown, encoding="utf-8")

    all_articles_meta.append({
        "url": url,
        "title": title,
        "file": filename,
        "category": category,
        "questions_count": len(articles),
    })
    print(f"    сохранено: {filename} ({len(articles)} вопросов)")


def parse_guide_page(url: str, soup: BeautifulSoup, category: str) -> bool:
    """Страница-гайд без <article>: контент лежит в h2-секциях (например onboarding-страницы).
    Возвращает False, если секций не нашлось (реальный тупик)."""
    h1_tag = soup.find("h1")
    title = h1_tag.get_text(strip=True) if h1_tag else url

    body = soup.body or soup
    sections: list[tuple[str, list[str]]] = []
    current_h2 = None
    current_paras: list[str] = []

    for el in body.find_all(["h2", "p", "li"]):
        if el.name == "h2":
            text = el.get_text(strip=True)
            if text == "На этой странице":
                continue
            if current_h2 is not None and current_paras:
                sections.append((current_h2, current_paras))
            current_h2 = text
            current_paras = []
        else:
            text = el.get_text(strip=True)
            if not text:
                continue
            current_paras.append(f"- {text}" if el.name == "li" else text)
    if current_h2 is not None and current_paras:
        sections.append((current_h2, current_paras))

    if not sections:
        return False

    qa_blocks = [f"\n## {h2}\n\n" + "\n".join(paras) for h2, paras in sections]
    markdown = f"# {title}\n\n" + "\n".join(qa_blocks)

    filename = "-".join(urlparse(url).path.strip("/").split("/")) + ".md"
    filepath = OUTPUT_DIR / filename
    filepath.write_text(markdown, encoding="utf-8")

    all_articles_meta.append({
        "url": url,
        "title": title,
        "file": filename,
        "category": category,
        "questions_count": len(sections),
    })
    print(f"    сохранено (guide): {filename} ({len(sections)} секций)")
    return True


def crawl(url: str, category: str, depth: int = 0) -> None:
    if url in visited_urls or depth > MAX_DEPTH:
        return
    visited_urls.add(url)

    try:
        soup, final_url = get_soup(url)
    except requests.RequestException as e:
        print(f"  ! ошибка загрузки {url}: {e}")
        return

    if final_url != url:
        if final_url in visited_urls:
            return
        visited_urls.add(final_url)
    url = final_url

    articles = soup.find_all("article")
    if articles:
        parse_article_page(url, soup, category)
        return

    child_links = set()
    for a in soup.find_all("a", href=True):
        raw_href = a["href"]
        if "?card=" in raw_href:
            continue
        child_url = clean_url(raw_href)
        if child_url.startswith(url) and child_url != url:
            child_links.add(child_url)

    if not child_links:
        if parse_guide_page(url, soup, category):
            return
        print(f"  ! тупик (нет ни article, ни дочерних ссылок): {url}")
        return

    for child_url in sorted(child_links):
        crawl(child_url, category, depth + 1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Сайт: {SITE_KEY} ({START_URL}) -> {OUTPUT_DIR}")
    print("Шаг 1: собираем подкатегории...")
    subcategories = get_subcategory_urls()
    print(f"  найдено подкатегорий: {len(subcategories)}")

    for i, subcat_url in enumerate(subcategories, 1):
        print(f"[{i}/{len(subcategories)}] подкатегория: {subcat_url}")
        crawl(subcat_url, category=subcat_url)

    with open(OUTPUT_DIR / "_index.json", "w", encoding="utf-8") as f:
        json.dump(all_articles_meta, f, ensure_ascii=False, indent=2)

    total_questions = sum(a["questions_count"] for a in all_articles_meta)
    print(f"\nГотово. Страниц с контентом: {len(all_articles_meta)}, вопросов всего: {total_questions}")


if __name__ == "__main__":
    main()