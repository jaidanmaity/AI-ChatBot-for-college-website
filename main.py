import os
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# --- CONFIGURATION ---
DB_PATH = "db"
# UPDATED: Using a much faster model
MODEL_NAME = "phi3:mini" 
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
STATIC_DIR = "static"

# --- INITIALIZE THE FastAPI APP ---
app = FastAPI()

# --- LOAD THE RAG CHAIN (runs only once on startup) ---
print("Loading the RAG chain...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
# Initialize the Ollama LLM with the new, faster model
llm = Ollama(model=MODEL_NAME) 
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
template = """
You are a helpful and knowledgeable assistant for the Thakur College of Engineering and Technology (TCET).
Your goal is to provide detailed and comprehensive answers based only on the context provided.
Do not make up information. If the context does not contain the answer, say so clearly.

Based on the following context, please provide a detailed answer to the question.

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("RAG chain loaded successfully.")


# --- WEBSOCKET ENDPOINT ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles the WebSocket connection for the chatbot."""
    await websocket.accept()
    try:
        while True:
            question = await websocket.receive_text()
            async for chunk in rag_chain.astream(question):
                await websocket.send_text(chunk)
            await websocket.send_text("<END_OF_STREAM>")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await websocket.close()


# --- STATIC FILE SERVING ---
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def read_root():
    """Serves the main index.html file."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
