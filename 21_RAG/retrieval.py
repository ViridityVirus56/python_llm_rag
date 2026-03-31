from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

gemini_key = os.getenv("GOOGLE_API_KEY")

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector_db = QdrantVectorStore.from_existing_collection(collection_name="learning_rag", url="http://localhost:6333", embedding=embedding_model)

client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=gemini_key)
while(True):

    user_query = input("💬 ")

    search_results = vector_db.similarity_search(query=user_query)

    context = "\n\n\n". join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_results])

    SYSTEM_PROMPT =f"""You are a helpful AI assistant who answers user query based on available data context retrieved from a PDF file along with page_contents and number.

    You should only answer user based on following data and navigate the user to open the right page.
    Available data:
    {context}
    """

    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    message_history.append({"role": "user", "content": user_query})

    response = client.chat.completions.create(messages=message_history, model="gemini-2.5-flash")

    print(f"🤖 {response.choices[0].message.content}")
