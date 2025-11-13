# AI-ChatBot-for-college-website
Builded a RAG chatbot from scratch using Python, Ollama (Phi-3), and LangChain for college webite


![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-white?logo=langchain)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit)
![Ollama](https://img.shields.io/badge/Ollama-black?logo=ollama)
![Selenium](https://img.shields.io/badge/Selenium-43B02A?logo=selenium)
![License Vibe Coded]([https://x.com/karpathy/status/1886192184808149383?lang=en])

# AI-Powered RAG Chatbot for my College Website

This project is a hands-on exploration of building a modern, end-to-end **Retrieval-Augmented Generation (RAG)** system from the ground up.

It began as a project for my **AINN** subject, driven by a simple question: "Most major websites have AI chatbots, why not our college?" The goal was to create a 24/7 "online reception person" for the TCET website,a bot that could answer any question about admissions, courses, or events, just like a real person on reception,

This repository is the story of that learning journey. That demonstrates a full, working pipeline and my conceptual understanding of how modern AI applications are built.

---

<img width="1920" height="1035" alt="image" src="https://github.com/user-attachments/assets/a83cfccc-6110-4d21-9788-df2d2df7dc0d" />


Here is a screenshot of the final chat bubble, which I built by integrating the college's HTML/CSS with my FastAPI backend.


## Project Concept: The RAG Pipeline

This is just a generic chatbot. **RAG** system, which means in theory it has to only answer questions based on information it finds on the TCET website.



The entire system works in two phases:

### Phase 1: Ingestion (The "Brain")

* **Scrape:** Crawl the `tcetmumbai.in` website and all its PDFs.
* **Clean:** Remove duplicate pages and boilerplate text.
* **Chunk:** Split the clean text into small, overlapping pieces.
* **Embed:** Convert each text chunk into a numerical vector (a list of numbers).
* **Store:** Save all these vectors in a `ChromaDB` vector database.

### Phase 2: Inference (The "Chat")

* **Query:** A user asks a question (e.g., "What is the admission process?").
* **Retrieve:** The system converts the *question* into a vector and searches the database for the most similar text *chunks*.
* **Augment:** The system takes the original question and the retrieved chunks and "augments" a new prompt for the LLM.
* **Generate:** The LLM (e.g.,`Mistral`/ `phi3:mini`) receives this big prompt and generates a final answer, which is streamed back to the user.

---

## Tech Stack

* **Local LLMs:** **Ollama** (serving `phi3:mini`, `mistral`, `gemma2b`)
* **AI Orchestration:** **LangChain**
* **Vector Database:** **ChromaDB** locally
* **Embeddings:** `BAAI/bge-base-en-v1.5`
* **Backend:** **FastAPI** (with WebSockets)
* **Frontend/Demo:** **Streamlit** & `index.html` (with basic JS)
* **Data Scraping:** **Selenium**, **BeautifulSoup**, **PyMuPDF (`fitz`)** (basiclibraries)
* **De-duplication:** **Datasketch (`MinHashLSH`)**

---

## My Learning Journey: File-by-File Breakdown

This project was built step-by-step. My thought process and learnings are captured in the evolution of these files.

### Part 1: The Data Pipeline (The Scrapers)

I quickly learned that just "scraping" a website isn't one simple task.

* `scrapper.py`: **My First Attempt.** This scraper uses `requests` and `BeautifulSoup`. It's very fast for simple, static HTML. however, is that it's the **only one** that uses `fitz` (PyMuPDF) to open and read all the `.pdf` files (like circulars and notices) that it finds.

* `scrapy.py`: **The Evolution.** I realized `requests` can't run JavaScript. Many parts of the college website are dynamic and only load content in a *real browser*. This scraper uses `Selenium` to launch a headless browser, wait for all the content to load, and *then* save it. It also uses the `trafilatura` library to intelligently extract *only* the main article text, giving me much cleaner data.

> **My Takeaway:** You need both. `scrapper.py` is my "PDF Specialist," and `scrapy.py` is my "Dynamic HTML Specialist." Together, they build the complete knowledge base.
>
> *[yes this was all vibe coded]*

### Part 2: LLM’s "Brain" (The Database)

Once I had the text, I had to build the "brain."

* `build_database.py` (The simple version): My first pass at this was straightforward. It loads *all* the `.txt` files, splits them into 1000-character chunks, and saves them to `ChromaDB`.

* `buildDatabse_noCopy.py` (The advanced version): I immediately saw a problem. The website has *tons* of duplicate pages (headers, menus, etc.). This creates "noise" in the database.
    * **My Solution:** I learned about **Locality-Sensitive Hashing (LSH)**. This advanced script uses the `datasketch` library to create a `MinHash` "fingerprint" for every single document. It then compares these fingerprints and **skips any file that is more than 85% similar** to one it has already seen.
    * **Chunking Strategy:** I also refined my chunking. In this file, I set `chunk_size=1500` and `chunk_overlap=300`. This gives the LLM a large, 1500-character piece of context to read, and the 300-character overlap ensures that a sentence isn't cut off between two chunks.

### Part 3: Experimenting (The Three Chatbots)

Vibe coded 3 different RAG pipelines to understand how they work at every level.

1.  **`chatbot.py`: The CLI (Command-Line) Version**
    * **My Mindset:** This was my primary tool for *understanding* and *debugging* the RAG process. It has no UI, but it's the most informative.
    * **Key Feature:** When you ask a question, this script first **prints the source chunks** it retrieved from the database *before* it generates an answer. This let me see *exactly* what context the LLM was given. It also prints the time taken for retrieval and generation, which helped me compare models.

2.  **`app.py`: The Streamlit Demo**
    * **My Mindset:** "How can I build a web UI for this *fast*?" `Streamlit` was the answer by gemini.
    * **Key Feature:** This is an "all-in-one" app. It runs a web server and the RAG pipeline in a single Python file. It's the perfect tool for rapid prototyping and showing a working demo to the professor or a friend.

3.  **`main.py` + `static/index.html`: The "Production" Version**
    * **My Mindset:** This is the "eye-candy" version that mimics a real-world application. a proper frontend/backend architecture.
    * **Key Feature:** `main.py` uses **FastAPI**, a high-performance framework. It creates a **WebSocket** endpoint (`/ws`). This file *only* handles the AI logic.
    * The `static/index.html` file (which I copied from the college website's source) contains the UI. The JavaScript in this file connects to the WebSocket, creating the final, polished chat bubble on the live site. This is the "correct" way to build a scalable app.

---

## How to Run This Project

1.  **Clone the Repo**
    > Dont do it, its a mess

2.  **Install Dependencies**
    * Create a virtual environment: `python -m venv venv`
    * Activate it: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
    * Install requirements: `pip install -r requirements.txt`

3.  **Download & Run Ollama**
    * [Install Ollama](https://ollama.com/)
    * Pull the models:
        ```bash
        ollama pull phi3:mini
        ollama pull mistral
        ```
    * Try different models suiting you and your computing power

4.  **Run the Data Pipeline (One-time setup)**
    * *This will take time. It's scraping the whole website.*
    ```bash
    # Dont Run this scrapers (create one for your website specifically)
    python scrapy.py
    python scrapper.py
    
    # Build the database using your own script which will help you understand your architectures need better
    python buildDatabse_noCopy.py
    ```
    > You can use these files as a reference but I would strongly insist on vibe coding it yourself which will be faster and more educational

5.  **Run Your Chatbot!**
    * **Option 1: The CLI Debugger**
        ```bash
        python chatbot.py #definitely create a cli based version to debug quickly and understand working in bg 
        ```

---

## Key Concepts I Understood in the process

* **RAG:** The entire end-to-end architecture of Retrieval-Augmented Generation.
* **Local LLMs:** Using **Ollama** to serve and switch between multiple open-source models (like `phi3` vs. `mistral`) to compare performance and speed.
* **Vector Embeddings:** The core concept of turning text into numbers so they can be semantically searched.
* **Data Pipelining:** The full workflow of scraping, cleaning, de-duplicating (with **LSH**), and ingesting data.
* **API vs. App:** The architectural difference between a production **FastAPI** backend and an all-in-one **Streamlit** demo.
