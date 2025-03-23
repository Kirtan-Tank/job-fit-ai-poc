import streamlit as st
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec
import io

# -----------------------------------------------------------------------------
# Configuration and Initialization
# -----------------------------------------------------------------------------
# Retrieve API keys and environment from Streamlit secrets.
# Ensure your .streamlit/secrets.toml has something like:
#
# [general]
# HF_API_KEY = "your-huggingface-api-key"
# PINECONE_API_KEY = "your-pinecone-api-key"
# PINECONE_ENV = "us-east-1"  # your desired region
#
HF_API_KEY = st.secrets["general"]["HF_API_KEY"]
PINECONE_API_KEY = st.secrets["general"]["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["general"]["PINECONE_ENV"]

INDEX_NAME = "job-fit-index"
MODEL_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

# Initialize Pinecone using the new SDK pattern.
# Create a Pinecone instance and specify the serverless deployment details.
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

# Create the index if it doesn't exist.
if INDEX_NAME not in pc.list_indexes().names():
    # For the chosen model (all-MiniLM-L6-v2), the embedding dimension is 384.
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",  # or 'dotproduct' depending on your preference
        spec=spec
    )
# Get a reference to the index.
index = pc.Index(INDEX_NAME)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def extract_text(file) -> str:
    """
    Extracts text from an uploaded file.
    Supports PDF, DOCX, and plain text files.
    """
    if file is None:
        return ""
    
    file_bytes = file.read()
    file.seek(0)
    
    if file.name.lower().endswith(".pdf"):
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return ""
    elif file.name.lower().endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        except Exception as e:
            st.error(f"Error processing DOCX: {e}")
            return ""
    else:
        try:
            return file_bytes.decode("utf-8")
        except Exception as e:
            st.error(f"Error decoding text file: {e}")
            return ""

@st.cache_data(show_spinner=False)
def get_embedding(text: str) -> np.ndarray:
    """
    Calls the Hugging Face Inference API to generate and cache the embedding for a given text.
    """
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": text}
    response = requests.post(MODEL_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    # Assume the model returns a list with a single embedding vector.
    return np.array(data[0])

def compute_fit_score(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Computes cosine similarity between two embeddings and maps it to a percentage.
    """
    sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    return ((sim + 1) / 2) * 100  # Convert from [-1, 1] to [0, 100]

def upsert_resume(resume_id: str, resume_emb: np.ndarray):
    """
    Upserts the resume embedding into the Pinecone index.
    """
    index.upsert(vectors=[(resume_id, resume_emb.tolist())])

def query_index(query_emb: np.ndarray, top_k: int = 1):
    """
    Queries the Pinecone index with the provided embedding.
    """
    return index.query(vector=query_emb.tolist(), top_k=top_k)

# -----------------------------------------------------------------------------
# Streamlit User Interface
# -----------------------------------------------------------------------------
def main():
    st.title("Job Fit Score Calculator")
    st.write("Upload a resume (or CV) and a job description document to calculate a fit score.")

    st.subheader("Upload Resume/CV")
    resume_file = st.file_uploader("Choose a PDF, DOCX, or TXT file for the Resume/CV", type=["pdf", "docx", "txt"], key="resume")
    
    st.subheader("Upload Job Description")
    jd_file = st.file_uploader("Choose a PDF, DOCX, or TXT file for the Job Description", type=["pdf", "docx", "txt"], key="jd")

    if st.button("Calculate Fit Score"):
        if resume_file and jd_file:
            with st.spinner("Extracting text and generating embeddings..."):
                resume_text = extract_text(resume_file)
                jd_text = extract_text(jd_file)
                
                if not resume_text or not jd_text:
                    st.error("Could not extract text from one or both of the files.")
                    return
                
                resume_emb = get_embedding(resume_text)
                jd_emb = get_embedding(jd_text)
                
                fit_score = compute_fit_score(resume_emb, jd_emb)
                st.success(f"Job Fit Score: {fit_score:.2f}%")
                
                # Upsert the resume embedding into Pinecone (using a fixed ID for demo purposes)
                resume_id = "resume_1"
                upsert_resume(resume_id, resume_emb)
                
                result = query_index(jd_emb, top_k=1)
                st.write("Pinecone Query Result:", result)
        else:
            st.error("Please upload both a resume and a job description file.")

if __name__ == "__main__":
    main()
