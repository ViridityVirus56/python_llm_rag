from server import app
import uvicorn
from dotenv import load_dotenv
import os



load_dotenv()

gemini_key = os.getenv("GOOGLE_API_KEY")

def main():
    uvicorn.run(app, port=8080, host="0.0.0.0")

main()