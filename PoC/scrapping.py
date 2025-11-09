import tempfile
from datetime import datetime, timedelta
from typing import Optional

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import re
import time
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests


def scrape_duyurular(url):
    """
    Scrape announcements from Bakırçay University EEM department website
    """
    with sync_playwright() as p:
        # Launch with stealth arguments
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

        page.wait_for_selector('.trending-courses-items', timeout=30000, state='visible')
        print("Found .trending-courses-items container")

        page.wait_for_selector('.trending-courses-items .item', timeout=30000, state='visible')
        print("Found announcement items")

        page.wait_for_load_state('networkidle', timeout=30000)

        items = page.query_selector_all('.trending-courses-items .item')

        print(f"Found {len(items)} announcements\n")

        announcements = []

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

            full_link = f"https://eem.bakircay.edu.tr/{relative_link}"

            meta_element = item.query_selector('.meta')

            if not meta_element:
                print(f"Skipping item {idx}: no meta element found")
                continue

            meta_text = meta_element.inner_text()

            date_match = re.search(r'(\d{2}\.\d{2}\.\d{4})', meta_text)
            date = date_match.group(1) if date_match else "N/A"

            announcement = {
                'title': title,
                'url': full_link,
                'date': date,
            }

            announcements.append(announcement)

            if idx < 5:
                print(f"[{idx + 1}] Title: {title}")
                print(f"    Date: {date}")
                print(f"    URL: {full_link}")
                print("-" * 80)

        browser.close()
        return announcements

def scrape_duyuru(url):
    """
    Scrape a single announcement page to get title, date, content, and links.
    Returns a dictionary.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        print(f"Navigating to {url}...")
        page.goto(url, timeout=60000, wait_until='domcontentloaded')

        print("Waiting for page to fully load...")
        time.sleep(3)

        page.wait_for_selector('.blog-area .info', timeout=30000, state='visible')

        info = page.locator(".blog-area .info").first

        title = info.locator("h3").inner_text().strip()

        print(f"Found {title}\n")

        date = info.locator(".meta li").nth(0).inner_text().strip()

        print(f"Found {date}\n")

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

        return {
            "title": title,
            "date": date,
            "content": content_text[0],
            "links": links
        }

def get_document_from_url(url):
    import tempfile
    import requests

    response = requests.get(url, verify=False)

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

    return doc_splits


def scrape_announcements_direct(url, time_range="all"):
    """Attempt direct scraping without browser"""
    url_for_visit = url + "/Duyurular"

    response = requests.get(url_for_visit, verify=False)

    print(f"Scraping {url}...")
    print(response)
    soup = BeautifulSoup(response.text, 'html.parser')

    print(soup)

    items = soup.select('.trending-courses-items .item')

    if not items:
        print("No items found - site may use JavaScript")
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

    # Parse items similar to your current logic
    announcements = []
    cutoff_date = get_cutoff_date(time_range)

    for item in items:
        title_elem = item.select_one('h5 a')
        # ... rest of parsing logic

    return announcements
if __name__ == "__main__":
    url = 'https://eem.bakircay.edu.tr/Duyurular'
    announcements = scrape_duyurular(url)
    print(announcements)

    #content = scrape_duyuru("https://eem.bakircay.edu.tr/Duyurular/5217/2025-2026-guz-donemi-lisansustu-ara-sinav-programi")
    #print(content)

    #doc_splits = get_document_from_url("https://eem.bakircay.edu.tr/Yuklenenler/MMF/EEM/s%C4%B1nav%20programlar%C4%B1/25_26_Guz/EEM_25_26_Lisans%C3%BCst%C3%BC_Ara_S%C4%B1nav_T.pdf")

    #announcements = scrape_announcements_direct(url)
    #print(announcements)