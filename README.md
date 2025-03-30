# Resume AI Chat & Ranking

## 📄 Overview
Resume AI Chat & Ranking is a Streamlit-based application that helps recruiters and hiring managers analyze and rank resumes using OpenAI's embeddings and ChromaDB. It allows users to:
- Upload resumes in PDF format.
- Store and process resumes using vector embeddings.
- Rank candidates based on job descriptions (JD) using cosine similarity.
- Ask questions about resumes using an AI-powered chatbot.

## 🚀 Features
- **Resume Upload & Processing**: Users can upload multiple PDF resumes, which are processed and stored in ChromaDB.
- **AI-Powered Resume Chat**: Users can ask questions about stored resumes and get AI-generated responses.
- **Candidate Ranking**: Job descriptions are matched against stored resumes using cosine similarity to rank candidates.
- **Persistent Storage**: Resumes are stored in ChromaDB for future use.

## 🏗️ Tech Stack
- **Python**
- **Streamlit** (UI Framework)
- **OpenAI API** (Embeddings & Chatbot)
- **ChromaDB** (Vector Store for Resumes)
- **LangChain** (For handling AI-based retrieval and embedding models)
- **scikit-learn** (Cosine Similarity for Ranking)
- **PyPDFLoader** (For extracting text from PDFs)

## 🔧 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/resume-ai-chat-ranking.git
cd resume-ai-chat-ranking
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv env
source env/bin/activate  # On macOS/Linux
env\Scripts\activate  # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up Environment Variables
Create a `.env` file in the project root and add your OpenAI API key:
```
open_api_key=your_openai_api_key_here
```

### 5️⃣ Run the Application
```bash
streamlit run Chat_&_rank_Resume.py
```

## 🎯 Usage Guide
### **Uploading Resumes**
1. Click on **Upload Resumes** in the sidebar.
2. Select multiple PDF resumes.
3. Click **Process Resumes** to extract and store the data.

### **Chat with Resumes**
1. Select **Ask Questions** mode.
2. Enter your query about the stored resumes.
3. Click **Ask AI** to get an AI-generated response.

### **Rank Candidates**
1. Select **Rank Candidates** mode.
2. Enter the **Job Description**.
3. Click **Rank Candidates** to see ranked resumes based on similarity.

## 📌 Future Improvements
- Integration with additional document types (e.g., DOCX, TXT).
- Improved ranking with advanced NLP techniques.
- Adding a frontend dashboard for better visualization.

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repo and submit a PR.

## 📜 License
This project is licensed under the MIT License.

