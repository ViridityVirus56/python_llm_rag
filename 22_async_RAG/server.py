from fastapi import FastAPI
from client.rq_client import queue
from queues.process import process_query
from dotenv import load_dotenv
import os



load_dotenv()

gemini_key = os.getenv("GOOGLE_API_KEY")
app = FastAPI()

@app.get("/")
def root():
    return {"status": "Server is up and running"}

@app.post("/chat")
def chat(query:str):
    job = queue.enqueue(process_query, query)

    return {"status": "queued", "job_id": job.id}

@app.post("/result")
def poll(idf:str):
    val = queue.fetch_job(idf).result
    return {"result":val}