# --- LIBRARY IMPORTS ---

from tqdm import tqdm #for progress bar
# `os` is a standard Python library for interacting with the operating system, like creating folders.
import os
# `DirectoryLoader` is a specific tool from the LangChain library designed to load all documents from a folder.
from langchain_community.document_loaders import DirectoryLoader
# `RecursiveCharacterTextSplitter` is LangChain's recommended tool for splitting long texts into smaller chunks.
from langchain.text_splitter import RecursiveCharacterTextSplitter
# `SentenceTransformerEmbeddings` is the class we use to load our open-source embedding model from Hugging Face.
from langchain_community.embeddings import SentenceTransformerEmbeddings
# `Chroma` is the class for the ChromaDB vector database, which will store our vectors.
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
# Purpose: Define constants to make the script easy to read and modify.
DATA_PATH = "scraped_data"
DB_PATH = "db"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5" # Using a more powerful model

# --- STEP 1: LOAD DOCUMENTS ---
# Purpose: To load all the raw text data from our .txt files into memory.
print("Loading documents...")
# Concept Applied: Document Loading.
# We initialize the DirectoryLoader, telling it where our data is (`DATA_PATH`) and what files to look for (`"*.txt"`).
loader = DirectoryLoader(DATA_PATH, glob="*.txt")
# The `.load()` method reads all the files and creates a list of Document objects. Each object contains the text and metadata (like the source filename).
documents = loader.load()
print(f"Loaded {len(documents)} documents.")

# --- STEP 2: SPLIT DOCUMENTS INTO CHUNKS ---
# Purpose: To break down the loaded documents into smaller, searchable pieces for the reasons we discussed (Context Windows and Search Accuracy).
print("Splitting documents into chunks...")
# Concept Applied: Text Splitting / Chunking.
# We initialize the splitter. It's "Recursive" because it tries to split text along logical separators (like newlines `\n\n`, then `\n`, then spaces) to keep related text together.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,    # The target size for each chunk in characters. We're increasing this from 1000.
    chunk_overlap=300   # The number of characters to overlap between chunks. This creates a "sliding window" to ensure context isn't lost between chunks. We're increasing this from 200.
)
# The `.split_documents()` method takes our list of long documents and returns a new list of smaller, chunked documents.
texts = text_splitter.split_documents(documents)
print(f"Created {len(texts)} text chunks.")

# --- STEP 3: CREATE EMBEDDINGS AND STORE IN DATABASE ---
# Purpose: To convert our text chunks into numerical vectors and save them in a searchable database.
print("Generating embeddings and creating the vector database... (This may take a while with the new model)")
# Concept Applied: Vectorization and Indexing.
# We initialize our embedding model. We've switched to `bge-base-en-v1.5`, which is a more powerful model known for better retrieval performance.
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

# This single command does all the heavy lifting:
# 1. It takes our `texts` (the chunks).
# 2. It uses the `embeddings` model to convert each chunk into a vector.
# 3. It creates/connects to a ChromaDB database that will be saved in the `DB_PATH` folder.
# 4. It "indexes" all these vectors so they can be searched quickly.
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=DB_PATH
)

# Define the batch size
batch_size = 100
# Use tqdm to create a progress bar
for i in tqdm(range(0, len(texts), batch_size), desc="Adding documents to DB"):
    # Select a batch of documents
    batch = texts[i:i+batch_size]
    # Add the batch to the vector database
    vectordb.add_documents(documents=batch)

print("\n✅ Database built successfully!")
