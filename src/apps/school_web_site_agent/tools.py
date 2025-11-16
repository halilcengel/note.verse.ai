import os

from langchain.tools import tool, ToolRuntime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from playwright.sync_api import sync_playwright
from apps.school_web_site_agent.context import Context
import time
import re
from datetime import datetime, timedelta
from typing import Optional, Literal
from core.vector_store import store


@tool
def get_document_from_url(runtime: ToolRuntime[Context],url: str):
    """
    Download and extract text content from a PDF document at the given URL.

    This tool fetches a PDF file from a URL, splits it into smaller text chunks,
    and returns the processed document segments. Useful for analyzing PDF documents
    such as exam schedules, announcements, academic calendars, or course materials.

    Args:
        url (str): Direct URL to the PDF file. Must be a valid HTTP/HTTPS URL
                   ending in .pdf or pointing to a PDF resource.

    Returns:
        list: A list of document chunks/splits, each containing:
            - page_content: The text content of the chunk
            - metadata: Information about the source document

    Example usage:
        - Extract exam schedule from a PDF announcement
        - Parse course syllabus or academic calendar
        - Analyze official documents shared on the school website

    Note: This tool bypasses SSL verification for institutional websites
          that may have certificate issues.
    """
    import tempfile
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    response = requests.get(url.replace("/Duyurular", "/"), verify=False, timeout=30)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )

    doc_splits = text_splitter.split_documents(docs)

    os.unlink(tmp_path)

    runtime.state["related_announcement_doc"] = doc_splits

    return doc_splits


@tool
def scrape_announcement(runtime: ToolRuntime[Context],url: str):
    """
    Extract detailed information from a single announcement page on the school website.

    This tool navigates to a specific announcement URL and extracts structured data
    including the title, publication date, full text content, and any associated links
    or attachments (such as PDFs, forms, or related documents).

    Args:
        url (str): Full URL to the individual announcement page. This should be
                   a direct link to a single announcement, not the announcements list page.

    Returns:
        dict: A dictionary containing:
            - title (str): The announcement headline/title
            - date (str): Publication date in DD.MM.YYYY format
            - content (str): The main text content/body of the announcement
            - links (list): List of dictionaries with 'text' and 'href' keys for
                           any downloadable files or referenced URLs

    Use cases:
        - Get full details of a specific announcement from the list
        - Extract PDF links from announcements (exam schedules, forms, etc.)
        - Read the complete content when the list view only shows summaries
        - Gather all related documents attached to an announcement

    Example:
        Use this after scrape_announcements() to get full details of interesting items,
        especially when you need to access attached PDF files or read complete content.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
            ]
        )

        context = browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
            locale='tr-TR',
        )

        page = context.new_page()

        # Hide webdriver property
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)

        print(f"Navigating to {url}...")
        page.goto(url, timeout=60000, wait_until='domcontentloaded')

        print("Waiting for page to fully load...")
        time.sleep(3)

        page.wait_for_selector('.blog-area .info', timeout=30000, state='visible')

        info = page.locator(".blog-area .info").first

        title = info.locator("h3").inner_text().strip()
        print(f"Found title: {title}\n")

        date = info.locator(".meta li").nth(0).inner_text().strip()
        print(f"Found date: {date}\n")

        content_text = []
        content_selectors = [
            ".info > p",
            ".item .info p",
            ".blog-area .info p"
        ]

        for selector in content_selectors:
            content_paragraphs = page.locator(selector)
            count = content_paragraphs.count()
            if count > 0:
                print(f"Found {count} paragraphs with selector: {selector}")
                for i in range(count):
                    text = content_paragraphs.nth(i).inner_text().strip()
                    if text:
                        content_text.append(text)
                if content_text:
                    break

        links = []
        link_selectors = [
            ".info p a",
            ".item .info a[href*='pdf']",
            ".blog-area .info a"
        ]

        for selector in link_selectors:
            content_links = page.locator(selector)
            count = content_links.count()
            if count > 0:
                print(f"Found {count} links with selector: {selector}")
                for i in range(count):
                    link = content_links.nth(i)
                    href = link.get_attribute('href')
                    text = link.inner_text().strip()
                    if href and 'addtoany' not in href.lower():
                        if href.startswith('../'):
                            href = url.rsplit('/', 2)[0] + '/' + href.replace('../', '')
                        elif not href.startswith('http'):
                            base_url = '/'.join(url.split('/')[:3])
                            href = base_url + '/' + href.lstrip('/')

                        links.append({
                            'text': text,
                            'href': href
                        })
                if links:
                    break

        browser.close()

        related_announcement = {
            "title": title,
            "date": date,
            "content": content_text,
            "links": links
        }

        runtime.state["related_announcement"] = related_announcement

        return {
            "title": title,
            "date": date,
            "content": content_text[0] if content_text else "",
            "links": links
        }


@tool
def query_school_regulations(
        runtime: ToolRuntime[Context],
        query: str,
        k: Optional[int] = 5
):
    """
    Search through school regulations and directives using semantic similarity.

    This tool performs a semantic search across all stored school regulations,
    directives (yönergeler), and official documents that have been added to the
    vector store. It returns the most relevant document chunks based on the query.

    Useful for finding information about:
    - Exam policies and makeup exam procedures (Mazeret sınavları)
    - Course exemption and equivalency rules (Muafiyet ve intibak)
    - Double major and minor programs (Çift anadal ve yan dal)
    - Special student regulations (Özel öğrenci)
    - International student policies (Uluslararası öğrenciler)
    - Diploma and diploma supplement procedures
    - Applied education regulations (Uygulamalı eğitim)
    - Language school directives (DİLMER)
    - Grade conversion tables (Not dönüşüm)
    - Measurement and evaluation principles (Ölçme değerlendirme)
    - Common courses coordination
    - Education commission directives

    Args:
        query (str): The question or search query about school regulations.
                    Can be in Turkish or English.
        k (int, optional): Number of most relevant document chunks to return.
                          Defaults to 5. Range: 1-20.

    Returns:
        dict: A dictionary containing:
            - query (str): The original query
            - num_results (int): Number of results returned
            - results (list): List of relevant document chunks, each containing:
                - content (str): The text content of the chunk
                - metadata (dict): Document metadata including:
                    - source_url (str): Original PDF URL
                    - page (int): Page number in the PDF
                    - source (str): File path/name
                - relevance_score (float): Similarity score (if available)

    Example queries:
        - "Mazeret sınavı nasıl alınır?"
        - "Çift anadal başvuru koşulları nelerdir?"
        - "Muafiyet için gerekli belgeler"
        - "How to apply for course exemption?"
        - "Uluslararası öğrenci kabul kriterleri"

    Note:
        - Results are ranked by semantic similarity to the query
        - The tool searches across all uploaded regulation PDFs
        - Requires OPENAI_API_KEY environment variable for embeddings
    """

    k = max(1, min(k, 20))

    try:

        print(f"Searching for: '{query}' (returning top {k} results)")
        results = store.similarity_search_with_score(query, k=k)

        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": float(score)
            })

        runtime.state["regulation_search_results"] = formatted_results
        runtime.state["last_regulation_query"] = query

        response = {
            "query": query,
            "num_results": len(formatted_results),
            "results": formatted_results
        }

        print(f"✓ Found {len(formatted_results)} relevant document chunks")

        return response

    except Exception as e:
        error_msg = f"Error querying vector store: {str(e)}"
        print(f"✗ {error_msg}")
        return {
            "error": error_msg,
            "query": query,
            "num_results": 0,
            "results": []
        }

@tool
def scrape_announcements(
        runtime: ToolRuntime[Context],
        time_range: Optional[Literal["1d", "1w", "1m", "3m", "6m", "1y", "all"]] = "all"
):
    """
    Scrape announcements from School's department website with optional time filtering.

    Args:
        runtime: The tool runtime context containing the URL
        time_range: Time range to filter announcements. Options:
            - "1d": Last 1 day
            - "1w": Last 1 week
            - "1m": Last 1 month  (default)
            - "3m": Last 3 months
            - "6m": Last 6 months
            - "1y": Last 1 year
            - "all": All announcements

    Returns:
        List of announcements within the specified time range
    """

    def parse_date(date_str: str) -> Optional[datetime]:
        """Parse date string in DD.MM.YYYY format"""
        try:
            return datetime.strptime(date_str, '%d.%m.%Y')
        except (ValueError, AttributeError):
            return None

    def get_cutoff_date(time_range: str) -> Optional[datetime]:
        """Calculate the cutoff date based on time range"""
        if time_range == "all":
            return None

        now = datetime.now()
        time_deltas = {
            "1d": timedelta(days=1),
            "1w": timedelta(weeks=1),
            "1m": timedelta(days=30),
            "3m": timedelta(days=90),
            "6m": timedelta(days=180),
            "1y": timedelta(days=365),
        }

        delta = time_deltas.get(time_range)
        return now - delta if delta else None

    with sync_playwright() as p:
        url = runtime.context.url
        announcement_url = url + "/Duyurular"

        browser = p.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
            ]
        )

        context = browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
            locale='tr-TR',
        )

        page = context.new_page()

        # Hide webdriver property
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)

        page.goto(announcement_url, timeout=60000, wait_until='domcontentloaded')

        time.sleep(3)

        page.wait_for_selector('.trending-courses-items', timeout=30000, state='visible')
        page.wait_for_selector('.trending-courses-items .item', timeout=30000, state='visible')
        page.wait_for_load_state('networkidle', timeout=30000)

        items = page.query_selector_all('.trending-courses-items .item')

        announcements = []
        cutoff_date = get_cutoff_date(time_range)

        for idx, item in enumerate(items):
            title_element = item.query_selector('h5 a')
            if not title_element:
                print(f"Skipping item {idx}: no title element found")
                continue

            title = title_element.inner_text().strip()
            relative_link = title_element.get_attribute('href')

            if not relative_link:
                print(f"Skipping item {idx}: no link found")
                continue

            full_link = f"{url}/{relative_link}"

            meta_element = item.query_selector('.meta')

            if not meta_element:
                print(f"Skipping item {idx}: no meta element found")
                continue

            meta_text = meta_element.inner_text()

            date_match = re.search(r'(\d{2}\.\d{2}\.\d{4})', meta_text)
            date_str = date_match.group(1) if date_match else "N/A"

            if cutoff_date and date_str != "N/A":
                announcement_date = parse_date(date_str)
                if announcement_date and announcement_date < cutoff_date:
                    continue

            announcement = {
                'title': title,
                'url': full_link,
                'date': date_str,
            }

            announcements.append(announcement)

        browser.close()

        runtime.state["announcements"] = announcements

        return {
            'count': len(announcements),
            'time_range': time_range,
            'announcements': announcements
        }