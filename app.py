import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from PyPDF2 import PdfReader

st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: bold; text-align: center; color: #1E88E5; margin-bottom: 1rem; }
    .subtitle { text-align: center; color: #666; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📄 Chat with Your PDF</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload any PDF and ask questions - powered by RAG & LangChain</p>', unsafe_allow_html=True)

# Clé API intégrée automatiquement
groq_api_key = st.secrets.get("GROQ_API_KEY", "")

with st.sidebar:
    st.header("⚙️ Configuration")
    if groq_api_key:
        st.success("✅ API Key configured")
    st.markdown("---")
    st.markdown("### 📚 How to use:")
    st.markdown("1. Upload a PDF file\n2. Wait for processing\n3. Ask questions!")
    st.markdown("---")
    st.markdown("### 🛠️ Built with:")
    st.markdown("• **LangChain** - RAG Framework\n• **Groq** - Fast LLM Inference\n• **FAISS** - Vector Search\n• **Streamlit** - UI")
    st.markdown("---")
    st.markdown("### 👩‍💻 Développé par")
    st.markdown("**Kenewy Diallo**")
    st.markdown("Analyste Data & IA")
    st.markdown("*© 2025 - Projet personnel*")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader("📎 Choose a PDF file", type=['pdf'])

if uploaded_file and groq_api_key:
    file_name = uploaded_file.name
    if st.session_state.processed_file != file_name:
        st.session_state.vector_store = None
        st.session_state.messages = []
        st.session_state.processed_file = file_name
    if st.session_state.vector_store is None:
        with st.spinner("🔄 Processing your PDF..."):
            try:
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                if not text.strip():
                    st.error("❌ Could not extract text from PDF.")
                    st.stop()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(text)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
                st.success(f"✅ Successfully processed **{file_name}**!")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()
    st.markdown("---")
    st.markdown("### 💬 Chat with your PDF")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if question := st.chat_input("Ask anything about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    docs = st.session_state.vector_store.similarity_search(question, k=3)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    prompt_template = """You are a helpful assistant. Answer based only on this context:
{context}
Question: {question}
If the answer is not in the context, say so."""
                    prompt = ChatPromptTemplate.from_template(prompt_template)
                    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
                    chain = ({"context": lambda x: context, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
                    answer = chain.invoke(question)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
else:
    st.info("👆 Upload a PDF file to start chatting!")
