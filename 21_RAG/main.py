from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

gemini_key = os.getenv("GOOGLE_API_KEY")

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector_db = QdrantVectorStore.from_existing_collection(
    collection_name="learning_rag",
    url="http://localhost:6333",
    embedding=embedding_model
)

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=gemini_key
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    search_results = vector_db.similarity_search(query=req.message)

    context = "\n\n\n".join([
        f"Page Content: {r.page_content}\nPage Number: {r.metadata['page_label']}\nFile Location: {r.metadata['source']}"
        for r in search_results
    ])

    SYSTEM_PROMPT = f"""
    You are a helpful AI assistant who answers based only on context.

    Context:
    {context}
    """

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.message}
        ]
    )

    return {"response": response.choices[0].message.content}