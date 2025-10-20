import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def scrape_page(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.find('h1')
        title_text = title.get_text(strip=True) if title else "Başlık bulunamadı"

        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text(strip=True) for p in paragraphs[:3]])

        return {
            'url': url,
            'title': title_text,
            'content': content[:200]
        }
    except Exception as e:
        print(f"Hata ({url}): {e}")
        return None


base_url = "https://eem.bakircay.edu.tr/Duyurular"
response = requests.get(base_url)

soup = BeautifulSoup(response.content, 'html.parser')

print(soup.prettify())

links = soup.find_all('a', href=True)


for link in links:
    full_url = urljoin(base_url, link['href'])
    print(full_url)
