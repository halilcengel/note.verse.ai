from langchain.tools import tool, ToolRuntime
from playwright.sync_api import sync_playwright
from apps.school_web_site_agent.context import Context
import time
import re

@tool
def scrape_announcements(runtime: ToolRuntime[Context]):
    """
    Scrape announcements from School's department website
    """
    with sync_playwright() as p:
        url = runtime.context.url
        url = url + "/Duyurular"

        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        page.goto(url, timeout=60000, wait_until='domcontentloaded')

        time.sleep(3)

        page.wait_for_selector('.trending-courses-items', timeout=30000, state='visible')

        page.wait_for_selector('.trending-courses-items .item', timeout=30000, state='visible')

        page.wait_for_load_state('networkidle', timeout=30000)

        items = page.query_selector_all('.trending-courses-items .item')

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

                full_link = f"{url}/{relative_link}"

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

        return announcements
