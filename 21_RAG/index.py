#responsible for indexing data
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

load_dotenv()

pdf_path = Path(__file__).parent/"consti.pdf"

loader = PyPDFLoader(file_path=pdf_path)

docs = loader.load()

splitter_text = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)

chunks = splitter_text.split_documents(documents=docs[:35])

#Vector embeddings

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector_store = QdrantVectorStore.from_documents(chunks,
    embeddings,
    url="http://localhost:6333",
    collection_name="learning_rag")


