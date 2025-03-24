import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec
import io
from huggingface_hub import InferenceClient

# -----------------------------------------------------------------------------
# Custom CSS for a Light, Creative Theme with Enhanced Readability
# -----------------------------------------------------------------------------
custom_css = """
<style>
/* Apply a soft pastel gradient background for the entire app */
body, .stApp {
    background: linear-gradient(135deg, #f2e8ff, #ffeef9) !important;
    color: #333333; /* Set default text color to dark grey for readability */
}

/* Main container adjustments */
div.block-container {
    background: transparent !important;
    padding: 2rem;
}

/* Style header text with a darker shade for contrast */
h1, h2, h3, h4, h5, h6 {
    color: #4a148c !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
}

/* Style paragraph text */
p {
    color: #333333;
}

/* Style buttons with a pleasant purple tone and sufficient contrast */
div.stButton > button {
    background-color: #8e6bbf;
    color: #ffffff !important;
    border: none;
    border-radius: 10px;
    padding: 0.7em 1.2em;
    font-size: 16px;
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.15);
}
div.stButton > button:hover {
    background-color: #7a5aa9;
}

/* Style the file uploader with a clear border and darker text */
div[data-testid="stFileUploader"] {
    background-color: #ffffff;
    border: 2px dashed #a393d9;
    border-radius: 10px;
    padding: 1rem;
    color: #333333;
}

/* Style the expander header and content for better readability */
.st-expanderHeader {
    background-color: #d1c4e9;
    color: #4a148c;
    border-radius: 8px;
    padding: 0.5rem;
}
.st-expanderContent {
    background-color: #f3e5f5;
    border-radius: 8px;
    padding: 1rem;
    color: #333333;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Configuration and Initialization
# -----------------------------------------------------------------------------
HF_API_KEY = st.secrets["general"]["HF_API_KEY"]
PINECONE_API_KEY = st.secrets["general"]["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["general"]["PINECONE_ENV"]

INDEX_NAME = "job-fit-index"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Returns 768-d embeddings
DESIRED_DIMENSION = 768

# Initialize Pinecone using the new SDK pattern.
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

existing_indexes = pc.list_indexes().names()
if INDEX_NAME in existing_indexes:
    desc = pc.describe_index(INDEX_NAME)
    if desc.dimension != DESIRED_DIMENSION:
        st.warning(f"Index dimension ({desc.dimension}) does not match desired dimension ({DESIRED_DIMENSION}). Recreating index.")
        pc.delete_index(INDEX_NAME)
        pc.create_index(
            name=INDEX_NAME,
            dimension=DESIRED_DIMENSION,
            metric="cosine",
            spec=spec
        )
else:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DESIRED_DIMENSION,
        metric="cosine",
        spec=spec
    )
index = pc.Index(INDEX_NAME)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def extract_text(file) -> str:
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
    client = InferenceClient(api_key=HF_API_KEY)
    try:
        result = client.feature_extraction(text, model=MODEL_NAME)
        embedding_array = np.array(result)
        if embedding_array.ndim == 2:
            pooled_embedding = embedding_array.mean(axis=0)
        elif embedding_array.ndim == 1:
            pooled_embedding = embedding_array
        else:
            st.error("Unexpected embedding dimensions.")
            return np.array([])
        if not isinstance(pooled_embedding.tolist(), list):
            st.error("Embedding is not a list-like structure.")
            return np.array([])
        return pooled_embedding
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return np.array([])

def compute_fit_score(emb1: np.ndarray, emb2: np.ndarray) -> float:
    sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    return ((sim + 1) / 2) * 100

def upsert_resume(resume_id: str, resume_emb: np.ndarray):
    index.upsert(vectors=[(resume_id, resume_emb.tolist())])

def query_index(query_emb: np.ndarray, top_k: int = 1):
    return index.query(vector=query_emb.tolist(), top_k=top_k)

# -----------------------------------------------------------------------------
# Streamlit User Interface
# -----------------------------------------------------------------------------
def main():
    st.title("Job Fit Score Calculator")
    st.write("Upload a resume (or CV) and a job description document to calculate a job fit score based on semantic similarity.")

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
                if resume_emb.size == 0 or jd_emb.size == 0:
                    st.error("Embedding generation failed. Please check your inputs and API configuration.")
                    return
                fit_score = compute_fit_score(resume_emb, jd_emb)
                st.success(f"Job Fit Score: {fit_score:.2f}%")
                resume_id = "resume_1"
                upsert_resume(resume_id, resume_emb)
                with st.expander("Show Pinecone Query Details"):
                    result = query_index(jd_emb, top_k=1)
                    st.write(result)
        else:
            st.error("Please upload both a resume and a job description file.")

if __name__ == "__main__":
    main()
