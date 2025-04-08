import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec
import io
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Custom CSS for a LinkedIn-Inspired Theme with Enhanced Readability and Dark Sidebar
# -----------------------------------------------------------------------------
custom_css = """
<style>
body, .stApp {
    background: linear-gradient(135deg, #E1F5FE, #BBDEFB) !important;
    color: #333333;
}
div.block-container {
    background: transparent !important;
    padding: 2rem;
}
h1, h2, h3, h4, h5, h6 {
    color: #0d6efd !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
}
p { color: #333333; }
div.stButton > button {
    background-color: #0d6efd;
    color: #ffffff !important;
    border: none;
    border-radius: 10px;
    padding: 0.7em 1.2em;
    font-size: 16px;
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.15);
}
div.stButton > button:hover {
    background-color: #0a58ca;
}
div[data-testid="stFileUploader"] {
    background-color: #ffffff;
    border: 2px dashed #90CAF9;
    border-radius: 10px;
    padding: 1rem;
    color: #333333;
}
.st-expanderHeader {
    background-color: #BBDEFB;
    color: #0d6efd;
    border-radius: 8px;
    padding: 0.5rem;
}
.st-expanderContent {
    background-color: #E3F2FD;
    border-radius: 8px;
    padding: 1rem;
    color: #333333;
}
/* Updated dark blue for sidebar using #0d6efd */
[data-testid="stSidebar"] {
    background-color: #0d6efd !important;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
[data-testid="stSidebar"] select {
    background-color: #0d6efd !important;
    color: #ffffff !important;
    border: 1px solid #ffffff !important;
}
/* Styling for large score display */
.score-display {
    font-size: 48px;
    font-weight: bold;
    color: #0d6efd;
    text-align: center;
    margin-top: 20px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: Model Selection (On-Demand mode only)
# -----------------------------------------------------------------------------
st.sidebar.markdown("<span style='color: #ffffff;'>On-Demand mode loads the model locally. This may take a few moments.</span>", unsafe_allow_html=True)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ondemand_status = st.sidebar.empty()
ondemand_status.info("Downloading model...")

@st.cache_resource(show_spinner=False)
def load_ondemand_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)

ondemand_model = load_ondemand_model()
ondemand_status.success("Model ready to use!")

# -----------------------------------------------------------------------------
# Configuration and Initialization for Pinecone
# -----------------------------------------------------------------------------
HF_API_KEY = st.secrets["general"]["HF_API_KEY"]
PINECONE_API_KEY = st.secrets["general"]["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["general"]["PINECONE_ENV"]

INDEX_NAME = "job-fit-index"
DESIRED_DIMENSION = 384

pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
existing_indexes = pc.list_indexes().names()
if INDEX_NAME in existing_indexes:
    desc = pc.describe_index(INDEX_NAME)
    if desc.dimension != DESIRED_DIMENSION:
        st.warning(f"Index dimension ({desc.dimension}) does not match desired dimension ({DESIRED_DIMENSION}). Recreating index.")
        pc.delete_index(INDEX_NAME)
        pc.create_index(name=INDEX_NAME, dimension=DESIRED_DIMENSION, metric="cosine", spec=spec)
else:
    pc.create_index(name=INDEX_NAME, dimension=DESIRED_DIMENSION, metric="cosine", spec=spec)
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

def get_embedding_ondemand(text: str) -> np.ndarray:
    # Use the locally loaded model to generate the embedding
    emb = ondemand_model.encode(text)
    return np.array(emb)

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
    st.title("Pro Connect AI Job Fit Score Calculator")
    st.write("Upload a job description document and a resume (or CV) to calculate a job fit score based on semantic similarity.")

    st.subheader("Upload Job Description")
    jd_file = st.file_uploader("Choose a PDF, DOCX, or TXT file for the Job Description", type=["pdf", "docx", "txt"], key="jd")
    
    st.subheader("Upload Resume/CV")
    resume_file = st.file_uploader("Choose a PDF, DOCX, or TXT file for the Resume/CV", type=["pdf", "docx", "txt"], key="resume")
    
    if jd_file:
        with st.expander("Review Extracted Job Description Text"):
            st.write(extract_text(jd_file))
    if resume_file:
        with st.expander("Review Extracted Resume Text"):
            st.write(extract_text(resume_file))

    if st.button("Calculate Fit Score"):
        if jd_file and resume_file:
            with st.spinner("Extracting text and generating embeddings..."):
                jd_text = extract_text(jd_file)
                resume_text = extract_text(resume_file)
                if not jd_text or not resume_text:
                    st.error("Could not extract text from one or both of the files.")
                    return
                # Always use on-demand mode for embeddings
                jd_emb = get_embedding_ondemand(jd_text)
                resume_emb = get_embedding_ondemand(resume_text)
                if jd_emb.size == 0 or resume_emb.size == 0:
                    st.error("Embedding generation failed. Please check your inputs and configuration.")
                    return
                fit_score = compute_fit_score(resume_emb, jd_emb)
                # Display the score in a larger, more prominent style
                st.markdown(f"<div class='score-display'>Job Fit Score: {fit_score:.2f}%</div>", unsafe_allow_html=True)
                resume_id = "resume_1"
                upsert_resume(resume_id, resume_emb)
                with st.expander("Show Pinecone Query Details"):
                    result = query_index(jd_emb, top_k=1)
                    st.write(result)
        else:
            st.error("Please upload both a job description and a resume (or CV).")

if __name__ == "__main__":
    main()
