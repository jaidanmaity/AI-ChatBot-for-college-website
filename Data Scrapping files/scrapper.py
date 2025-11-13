import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time
import fitz  # PyMuPDF
import hashlib # NEW: For creating clean filenames

# --- Configuration ---
START_URL = "https://www.tcetmumbai.in/"
DOMAIN = urlparse(START_URL).netloc
OUTPUT_DIR = "scraped_data"
PROGRESS_FILE = "visited.log"
# NEW: A file to map ugly filenames to pretty URLs
URL_MAP_FILE = "url_map.tsv"

# --- Interactive Mode Settings ---
INTERACTIVE_MODE = True
INTERACTIVE_ASK_ALL = False
INTERACTIVE_EXTENSIONS = {'.pdf'} 

IGNORED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.zip', '.mp3', '.mp4', '.avi', '.mov', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'}

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- State Management Functions ---
def load_visited_urls():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f)
    return set()

def save_visited_url(url):
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(url + "\n")

# NEW: Function to save the filename-to-URL mapping
def save_url_mapping(filename, url):
    with open(URL_MAP_FILE, "a", encoding="utf-8") as f:
        f.write(f"{filename}\t{url}\n")

# --- Main Script ---
urls_to_visit = [START_URL]
visited_urls = load_visited_urls()
if visited_urls:
    print(f"✅ Resuming scrape. Loaded {len(visited_urls)} already visited URLs.")

while urls_to_visit:
    current_url = urls_to_visit.pop(0)
    if current_url in visited_urls:
        continue

    # ... (The interactive skip and ignored extension logic is the same) ...
    if any(current_url.lower().endswith(ext) for ext in IGNORED_EXTENSIONS):
        save_visited_url(current_url)
        continue

    if INTERACTIVE_MODE and any(current_url.lower().endswith(ext) for ext in INTERACTIVE_EXTENSIONS):
        user_input = input(f"❓ Scrape PDF? [Y/n]: {current_url}\n   > ")
        if user_input.lower() == 'n':
            print(f"⏩ User skipped.")
            save_visited_url(current_url)
            continue

    print(f"🕸️  Scraping: {current_url}")
    
    try:
        response = requests.get(current_url, timeout=10)
        response.raise_for_status()

        page_text = ""
        # ... (PDF and HTML extraction logic is the same) ...
        if current_url.lower().endswith('.pdf'):
            with fitz.open(stream=response.content, filetype="pdf") as doc:
                for page in doc:
                    page_text += page.get_text()
        else:
            soup = BeautifulSoup(response.content, "html.parser")
            page_text = soup.get_text(separator=" ", strip=True)
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(current_url, href).split('#')[0]
                if urlparse(full_url).netloc == DOMAIN and full_url not in visited_urls and full_url not in urls_to_visit:
                    urls_to_visit.append(full_url)
        
        if page_text:
            # NEW: Create a clean, unique filename using a hash
            filename = hashlib.md5(current_url.encode()).hexdigest() + ".txt"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(page_text)
            # NEW: Save the mapping from the new filename to the original URL
            save_url_mapping(filename, current_url)
        
        save_visited_url(current_url)
        time.sleep(1)

    except requests.RequestException as e:
        print(f"❗️ Error fetching {current_url}: {e}")
        save_visited_url(current_url)
    except Exception as e:
        print(f"❗️ An error occurred while processing {current_url}: {e}")

print(f"\n✅ Scraping complete or queue is empty.")
