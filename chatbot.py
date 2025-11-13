import time
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
DB_PATH = "db"
MODEL_NAME = "mistral"
# We now use the more powerful embedding model we created the database with.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# --- STEP 1: LOAD DATABASE & LLM ---
print("Loading the vector database...")
# Concept: We must use the *exact same embedding model* that we used to create the database.
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
llm = Ollama(model=MODEL_NAME)
print("Database and LLM loaded successfully.")

# --- STEP 2: CREATE THE RETRIEVER ---
# Concept: The retriever is the component responsible for fetching documents from the database.
# We are asking for the top 5 most relevant documents (`k=5`).
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# --- STEP 3: CREATE A MORE DETAILED PROMPT TEMPLATE ---
# Concept: Prompt Engineering. The quality of your instructions to the LLM dramatically affects the quality of the answer.
# We are giving it a persona ("helpful assistant"), clear instructions (use *only* the context), and a format for the answer.
template = """
You are a helpful and knowledgeable assistant for the Thakur College of Engineering and Technology (TCET).
Your goal is to provide detailed and comprehensive answers based only on the context provided.
Do not make up information. If the context does not contain the answer, say so.

Based on the following context, please provide a detailed answer to the question.

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# --- STEP 4: CREATE THE FINAL RAG CHAIN ---
# Concept: LangChain Expression Language (LCEL). This `|` syntax is called a "pipe," and it chains components together.
# The data flows from left to right through the chain.
chain = (
    # This dictionary prepares the inputs for the prompt.
    # "context": The retriever is called with the user's question.
    # "question": The user's question is passed through unchanged.
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt  # The inputs are fed into the prompt template.
    | llm     # The formatted prompt is sent to the LLM.
    | StrOutputParser() # This component extracts just the text answer from the LLM's response.
)

# --- STEP 5: START THE CONVERSATION LOOP WITH FEEDBACK ---
print("\nChatbot is ready! Type 'exit' to end the conversation.")
while True:
    query = input("\n> ")
    if query.lower() == 'exit':
        break

    start_time = time.time() # Start the timer

    # --- FEEDBACK STEP 1: RETRIEVAL ---
    print("\n[1/3] Searching database for relevant documents...")
    retrieved_docs = retriever.invoke(query)
    retrieval_time = time.time() - start_time
    print(f"   Found {len(retrieved_docs)} documents in {retrieval_time:.2f} seconds.")

    # --- DEBUGGING: PRINT THE CONTEXT ---
    print("\n--- RETRIEVED CONTEXT ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n[DOCUMENT {i+1} SOURCE: {doc.metadata.get('source', 'Unknown')}]")
        print(doc.page_content)
    print("--------------------------\n")
    
    # --- FEEDBACK STEP 2: GENERATION ---
    print("[2/3] Sending context to the LLM for analysis... (This may take a while)")
    
    # --- FEEDBACK STEP 3: STREAMING THE RESPONSE ---
    print("[3/3] Receiving response from LLM:\n")
    
    generation_start_time = time.time()
    # Invoke the full chain to get the final answer and stream it
    for chunk in chain.stream(query):
        print(chunk, end="", flush=True)
    
    generation_time = time.time() - generation_start_time
    print(f"\n\n(LLM generation took {generation_time:.2f} seconds)")