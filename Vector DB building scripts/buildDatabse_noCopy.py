import os
import hashlib
from tqdm import tqdm

# NEW: Imports for MinHashing and LSH
from datasketch import MinHash, MinHashLSH

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
DATA_PATH = "scraped_data"
DB_PATH = "db"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# --- STEP 1: LOAD DOCUMENTS ---
print("Loading documents...")
loader = DirectoryLoader(DATA_PATH, glob="*.txt")
documents = loader.load()
print(f"Loaded {len(documents)} total documents from disk.")

# --- NEW STEP 1.5: ADVANCED NEAR-DUPLICATE REMOVAL WITH LSH ---
print("Scanning for and removing near-duplicate content using LSH...")

# Concept Applied: Locality-Sensitive Hashing (LSH) for Near-Duplicate Detection.
# We create an LSH index. This is our "smart filing system."
lsh = MinHashLSH(
    threshold=0.85,      # The similarity threshold (Jaccard similarity). Documents 85% or more similar will be considered duplicates.
    num_perm=128         # The size of the MinHash "fingerprint". 128 is a standard, good-quality value.
)

unique_documents = []
# We use enumerate to get both the index and the document, which LSH can use as a key.
for idx, doc in enumerate(tqdm(documents, desc="Processing documents for duplication")):
    # 1. Create the MinHash fingerprint for the document content.
    #    - We split the text into words to create a set of shingles.
    #    - We create a MinHash object and update it with each word.
    shingles = doc.page_content.split()
    minhash = MinHash(num_perm=128)
    for shingle in shingles:
        minhash.update(shingle.encode('utf-8'))
    
    # 2. Query the LSH index to find any documents that are "close" to this one.
    similar_docs = lsh.query(minhash)
    
    # 3. If the query returns an empty list, it means no similar documents have been seen before.
    if not similar_docs:
        # It's a unique document. Add it to our clean list and insert its fingerprint into the LSH index.
        unique_documents.append(doc)
        lsh.insert(f"doc_{idx}", minhash)

print(f"Removed {len(documents) - len(unique_documents)} near-duplicate documents.")
print(f"Proceeding with {len(unique_documents)} unique documents.")


# --- STEP 2: SPLIT THE (NOW DE-DUPLICATED) DOCUMENTS INTO CHUNKS ---
print("Splitting unique documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)
texts = text_splitter.split_documents(unique_documents)
print(f"Created {len(texts)} text chunks.")


# --- STEP 3: CREATE EMBEDDINGS AND STORE ---
print("Generating embeddings and creating the vector database...")
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

batch_size = 100
for i in tqdm(range(0, len(texts), batch_size), desc="Adding documents to DB"):
    batch = texts[i:i+batch_size]
    vectordb.add_documents(documents=batch)

print("\n✅ Advanced deduplication complete. Clean database built successfully!")