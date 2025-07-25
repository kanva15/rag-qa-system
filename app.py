import streamlit as st
import os
import shutil
import tempfile
from langchain_core.documents import Document
from ask_question import create_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Ask Your Docs", layout="centered")
st.title("🧠 Ask Your PDF")
st.write("Upload a PDF, ask anything about it.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
show_chunks = st.checkbox("Show retrieved chunks")

if uploaded_file:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        os.makedirs("docs", exist_ok=True)
        shutil.copy(temp_pdf_path, os.path.join("docs", uploaded_file.name))

    with st.spinner("Indexing and loading your document..."):
        try:
            qa_chain = create_qa_chain()
            st.success("Ready! Ask a question below.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.stop()

    query = st.text_input("Ask something about the PDF")
    if query:
        with st.spinner("Searching..."):
            result = qa_chain.invoke(query)
            st.markdown("### 💬 Answer:")
            st.write(result["result"])

            if show_chunks:
                docs = qa_chain.retriever.get_relevant_documents(query)
                st.markdown("---")
                st.markdown("### 🔍 Retrieved Chunks")
                for i, doc in enumerate(docs):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.code(doc.page_content.strip())
