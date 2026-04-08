import streamlit as st
import base64
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(layout="wide")
st.title("📄 RAG Chat with PDF Viewer")

# ------------------ INIT ------------------
@st.cache_resource
def init():
    gemini_key = os.getenv("GOOGLE_API_KEY")

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vector_db = QdrantVectorStore.from_existing_collection(
        collection_name="learning_rag",
        url="http://localhost:6333",
        embedding=embedding_model
    )

    client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=gemini_key
    )

    return vector_db, client

vector_db, client = init()

# ------------------ STATE ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

if "current_page" not in st.session_state:
    st.session_state.current_page = 1


# ------------------ PDF VIEWER ------------------
def display_pdf(file_path, page=1):
    if not file_path:
        st.info("No PDF selected")
        return

    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")

        pdf_display = f"""
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}#page={page}" 
            width="100%" height="800px">
        </iframe>
        """

        st.markdown(pdf_display, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading PDF: {e}")


import tempfile

st.sidebar.title("📂 Upload / Select PDF")

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Set as current PDF
    st.session_state.current_pdf = pdf_path
    st.session_state.current_page = 1

    st.sidebar.success("PDF loaded")


# ------------------ LAYOUT ------------------
col1, col2 = st.columns([1, 1])

# ------------------ CHAT ------------------
with col1:
    st.subheader("💬 Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask something...")

    if user_query:
        st.session_state.messages.append(
            {"role": "user", "content": user_query}
        )

        with st.chat_message("user"):
            st.markdown(user_query)

        # ---- RAG SEARCH ----
        search_results = vector_db.similarity_search(user_query, k=5)

        context = "\n\n".join([
            f"Page Content: {r.page_content}\nPage Number: {r.metadata['page_label']}"
            for r in search_results
        ])

        SYSTEM_PROMPT = f"""
        Answer strictly from context.
        Always mention page numbers.

        Context:
        {context}
        """

        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *st.session_state.messages
            ]
        )

        answer = response.choices[0].message.content

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

        # ---- AUTO SELECT TOP RESULT ----
        top = search_results[0]
        st.session_state.current_pdf = top.metadata["source"]
        st.session_state.current_page = int(top.metadata["page_label"])

        st.markdown("### 🔎 Sources")

        for i, r in enumerate(search_results):
            page = int(r.metadata["page_label"])
            source = r.metadata["source"]

            if st.button(f"Open Page {page} (Result {i+1})"):
                st.session_state.current_pdf = source
                st.session_state.current_page = page


# ------------------ PDF PANEL ------------------
with col2:
    st.subheader("📄 PDF Viewer")

    if st.session_state.current_pdf:
        st.write(
            f"Showing page {st.session_state.current_page}"
        )

    display_pdf(
        st.session_state.current_pdf,
        st.session_state.current_page
    )