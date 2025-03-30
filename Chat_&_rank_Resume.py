import os
import streamlit as st
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.chains import RetrievalQA
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai

load_dotenv()
openai_api_key = os.getenv("open_api_key")
openai.api_key = openai_api_key

# === ChromaDB Directory ===
CHROMA_DB_DIR = "chroma_db"
vectorstore = None

# === Load or Initialize Vectorstore ===
def load_vectorstore():
    global vectorstore
    try:
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=OpenAIEmbeddings())
        stored_docs = vectorstore._collection.count()
        st.sidebar.success(f"📌 ChromaDB loaded: {stored_docs} resumes")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading ChromaDB: {e}")

# === Process Full Resumes ===
def process_resumes(uploaded_files):
    global vectorstore

    if not uploaded_files:
        st.warning("⚠️ No resumes uploaded.")
        return

    all_documents = []
    resume_texts = {}

    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())  # Save file locally

        try:
            loader = PyPDFLoader(uploaded_file.name)
            documents = loader.load()
            full_text = "\n".join([doc.page_content for doc in documents])  # Combine all pages
            resume_texts[uploaded_file.name] = full_text  # Store full resume text
            all_documents.append(Document(page_content=full_text, metadata={"source": uploaded_file.name}))
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    # === Embed Full Resumes ===
    embeddings = OpenAIEmbeddings() # "text-embedding-ada-002"
    vectorstore = Chroma.from_documents(
        documents=all_documents,  # Store full resumes, not chunks
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    vectorstore.persist()
    stored_docs = vectorstore._collection.count()
    st.sidebar.success(f"📌 {stored_docs} resumes stored in ChromaDB.")

    # Store full resume texts for ranking
    st.session_state["resume_texts"] = resume_texts

# === Rank Candidates Based on JD ===
def rank_candidates(jd_text):
    global vectorstore

    if vectorstore is None or vectorstore._collection.count() == 0:
        return "⚠️ Vectorstore is empty! Please upload resumes first."

    st.sidebar.info(f"🔍 Ranking {vectorstore._collection.count()} resumes...")

    # Get embeddings
    embeddings = OpenAIEmbeddings()
    jd_embedding = embeddings.embed_query(jd_text)

    # Retrieve stored embeddings (entire resumes)
    all_resumes = vectorstore.get(include=["documents", "embeddings"])
    resume_texts = st.session_state.get("resume_texts", {})

    scores = []
    for i, (resume_text, resume_embedding) in enumerate(zip(all_resumes["documents"], all_resumes["embeddings"])):
        similarity = cosine_similarity([jd_embedding], [resume_embedding])[0][0]
        scores.append((resume_text, similarity))

    # Sort resumes by similarity score (descending)
    ranked_resumes = sorted(scores, key=lambda x: x[1], reverse=True)

    return ranked_resumes

# === Chatbot for Resume Q&A ===
def ask_question(query):
    global vectorstore

    if vectorstore is None or vectorstore._collection.count() == 0:
        return "⚠️ No resumes available! Please upload resumes first."

    # Create Retrieval QA Chain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    response = qa_chain.run(query)
    return response

# === Streamlit UI ===
st.set_page_config(page_title="Resume AI Chat & Ranking", page_icon="📄", layout="wide")

st.title("📄 Resume AI Chat & Ranking")
st.sidebar.header("🔧 Settings")

# === Load Vectorstore on Startup ===
load_vectorstore()

# === Upload Resumes ===
st.sidebar.subheader("📂 Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if st.sidebar.button("Process Resumes"):
    process_resumes(uploaded_files)

# === Chatbot or Ranking Mode ===
mode = st.sidebar.radio("Choose Mode", ["🔍 Ask Questions", "🏆 Rank Candidates"])

if mode == "🔍 Ask Questions":
    st.subheader("🤖 Ask AI About Resumes")
    user_question = st.text_input("❓ Enter your question:")
    
    if st.button("Ask AI"):
        if user_question:
            response = ask_question(user_question)
            st.subheader("🧠 AI Response:")
            st.write(response)
        else:
            st.warning("⚠️ Please enter a question.")

elif mode == "🏆 Rank Candidates":
    st.subheader("📝 Rank Candidates Based on JD")
    jd_text = st.text_area("📜 Enter Job Description:")

    if st.button("Rank Candidates"):
        if jd_text:
            ranked_candidates = rank_candidates(jd_text)
            if isinstance(ranked_candidates, str):  # Error message
                st.warning(ranked_candidates)
            else:
                st.subheader("🏆 Candidate Rankings")
                for i, (resume_text, score) in enumerate(ranked_candidates):
                    st.write(f"**{i+1}. Score: {score:.4f}**")
                    st.text_area(f"🔹 Resume {i+1} Preview", resume_text[:500], height=150)
        else:
            st.warning("⚠️ Please enter a Job Description.")

# === Display Stored Resumes ===
if vectorstore and vectorstore._collection.count() > 0:
    st.sidebar.subheader("📌 ChromaDB Documents")
    st.sidebar.text(f"{vectorstore._collection.count()} resumes stored.")
