import time
import hashlib
import os
from urllib.parse import urljoin, urlparse, unquote

import trafilatura
from bs4 import BeautifulSoup

# Selenium imports, now including "smart wait" components
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
from selenium.common.exceptions import WebDriverException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- Configuration ---
START_URL = "https://www.tcetmumbai.in/"
DOMAIN = urlparse(START_URL).netloc
OUTPUT_DIR = "scraped_data"
PROGRESS_FILE = "visited.log"
URL_MAP_FILE = "url_map.tsv"
# Max time for Selenium's "smart wait" to wait for a page element
WAIT_TIMEOUT = 10 

# --- Settings ---
IGNORED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.webp', '.zip', '.mp3', '.mp4', '.avi', '.mov', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'}

# ... (The initial setup and WebDriver initialization are the same) ...
# --- Selenium WebDriver Setup for Firefox ---
print("Initializing Selenium WebDriver for Firefox...")
try:
    service = Service(GeckoDriverManager().install())
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")
    driver = webdriver.Firefox(service=service, options=options)
    print("WebDriver initialized successfully.")
except Exception as e:
    print(f"Error initializing WebDriver: {e}")
    exit()

# --- NEW: URL Normalization Function ---
def normalize_url(url):
    """Cleans and standardizes a URL."""
    parsed = urlparse(url)
    # Ensure scheme is https and remove 'www.'
    scheme = 'https'
    netloc = parsed.netloc.replace('www.', '')
    # Remove trailing slashes and default filenames
    path = parsed.path.rstrip('/')
    if path.endswith('/index.html'):
        path = path[:-11]
    # Unquote to handle special characters in URL
    path = unquote(path)
    # Rebuild the URL
    return f"{scheme}://{netloc}{path}"

# --- State Management Functions (Unchanged) ---
def load_visited_urls():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f)
    return set()

def save_visited_url(url):
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(url + "\n")

def save_url_mapping(filename, url):
    with open(URL_MAP_FILE, "a", encoding="utf-8") as f:
        f.write(f"{filename}\t{url}\n")

# --- Main Script ---
# Normalize the starting URL
start_url_normalized = normalize_url(START_URL)
urls_to_visit = [start_url_normalized]
visited_urls = load_visited_urls()
if visited_urls:
    print(f"✅ Resuming scrape. Loaded {len(visited_urls)} already visited URLs.")

# --- The Main Crawling Loop ---
while urls_to_visit:
    current_url = urls_to_visit.pop(0)
    # Skip if the already normalized URL has been visited
    if current_url in visited_urls:
        continue

    if any(current_url.lower().endswith(ext) for ext in IGNORED_EXTENSIONS):
        print(f"🚫 Ignoring file: {current_url}")
        save_visited_url(current_url)
        continue

    print(f"🕸️  Scraping: {current_url}")
    
    try:
        driver.get(current_url)
        
        # --- NEW: "Smart Wait" Logic ---
        # Instead of a dumb sleep, we wait up to 10 seconds for the main body of the page to be loaded.
        # This is much more reliable for pages with heavy JavaScript.
        WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        page_html = driver.page_source
        
        # Automated content extraction with Trafilatura
        page_text = trafilatura.extract(page_html)
        
        if not page_text:
            print("   -> Trafilatura failed. Falling back to full text extraction.")
            soup_fallback = BeautifulSoup(page_html, "html.parser")
            page_text = soup_fallback.get_text(separator=" ", strip=True)
        else:
            print("   -> Successfully extracted main content with Trafilatura.")

        soup = BeautifulSoup(page_html, "html.parser")

        if page_text:
            filename = hashlib.md5(current_url.encode()).hexdigest() + ".txt"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(page_text)
            save_url_mapping(filename, current_url)
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href and not href.startswith(('mailto:', 'tel:')):
                full_url = urljoin(current_url, href).split('#')[0]
                # NORMALIZE the new URL before adding it to the queue
                normalized_new_url = normalize_url(full_url)
                
                if urlparse(normalized_new_url).netloc == DOMAIN and normalized_new_url not in visited_urls and normalized_new_url not in urls_to_visit:
                    urls_to_visit.append(normalized_new_url)
        
        save_visited_url(current_url)

    except TimeoutException:
        print(f"❗️ Page timed out after {WAIT_TIMEOUT} seconds: {current_url}")
        save_visited_url(current_url)
    except WebDriverException as e:
        print(f"❗️ Browser error on {current_url}: {e}")
        save_visited_url(current_url)
    except Exception as e:
        print(f"❗️ An unexpected error occurred: {e}")
        save_visited_url(current_url)

driver.quit()
print(f"\n✅ Scraping complete.")