import os
import io
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------- 
# Custom CSS Theme 
# ----------------------------------------------------------------------------- 
custom_css = """ 
<style>
/* Gradient background and modern look */
body, .stApp { background: linear-gradient(135deg, #f2e8ff, #ffeef9) !important; color: #333333; }
div.block-container { background: transparent !important; padding: 2rem; }
h1, h2, h3, h4, h5, h6 {
    color: #4a148c !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
}
p { color: #333333; }
div.stButton > button {
    background-color: #8e6bbf; color: #ffffff !important;
    border: none; border-radius: 10px; padding: 0.7em 1.2em;
    font-size: 16px; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.15);
}
div.stButton > button:hover { background-color: #7a5aa9; }
div[data-testid="stFileUploader"] {
    background-color: #ffffff; border: 2px dashed #a393d9;
    border-radius: 10px; padding: 1rem; color: #333333;
}
.st-expanderHeader {
    background-color: #d1c4e9; color: #4a148c;
    border-radius: 8px; padding: 0.5rem;
}
.st-expanderContent {
    background-color: #f3e5f5; border-radius: 8px;
    padding: 1rem; color: #333333;
}
[data-testid="stSidebar"] { background-color: #2e003e !important; }
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] select {
    background-color: #2e003e !important;
    color: #ffffff !important;
    border: 1px solid #ffffff !important;
}
</style> 
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ----------------------------------------------------------------------------- 
# Sidebar: Mode Selection 
# ----------------------------------------------------------------------------- 
mode = st.sidebar.radio("Select Mode", ["Online", "Offline"])
if mode == "Online":
    st.sidebar.markdown("<span style='color: #ffffff;'>Online mode uses the Hugging Face Inference API for embeddings.</span>", unsafe_allow_html=True)
    model_options = {
        "Model 1": "sentence-transformers/all-MiniLM-L6-v2",
        "Model 2": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "Model 3": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "Model 4": "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "Model 5": "sentence-transformers/all-MiniLM-L12-v2"
    }
    selected_model_label = st.sidebar.selectbox("Select a model", list(model_options.keys()))
    MODEL_NAME = model_options[selected_model_label]
else:
    st.sidebar.markdown("<span style='color: #ffffff;'>Offline mode loads the model locally from GitHub folder.</span>", unsafe_allow_html=True)
    MODEL_NAME = "./downloads/all-MiniLM-L6-v2"  # Your local model path
    offline_status = st.sidebar.empty()
    offline_status.info("Loading local model from: " + MODEL_NAME)

    @st.cache_resource(show_spinner=False)
    def load_offline_model() -> SentenceTransformer:
        return SentenceTransformer(MODEL_NAME)
    offline_model = load_offline_model()
    offline_status.success("Model loaded successfully!")

# ----------------------------------------------------------------------------- 
# Pinecone Setup 
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
        st.warning(f"Index dimension mismatch. Recreating index.")
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
            return "\n".join([p.extract_text() or "" for p in pdf_reader.pages]).strip()
        except Exception as e:
            st.error(f"PDF Error: {e}")
            return ""
    elif file.name.lower().endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            return "\n".join([para.text for para in doc.paragraphs]).strip()
        except Exception as e:
            st.error(f"DOCX Error: {e}")
            return ""
    else:
        try:
            return file_bytes.decode("utf-8")
        except Exception as e:
            st.error(f"Text file error: {e}")
            return ""

@st.cache_data(show_spinner=False)
def get_embedding_online(text: str) -> np.ndarray:
    client = InferenceClient(api_key=HF_API_KEY)
    try:
        result = client.feature_extraction(text, model=MODEL_NAME)
        embedding_array = np.array(result)
        return embedding_array.mean(axis=0) if embedding_array.ndim == 2 else embedding_array
    except Exception as e:
        st.error(f"Online embedding error: {e}")
        return np.array([])

def get_embedding_offline(text: str) -> np.ndarray:
    try:
        embedding = offline_model.encode(text)
        return embedding.mean(axis=0) if embedding.ndim == 2 else embedding
    except Exception as e:
        st.error(f"Offline embedding error: {e}")
        return np.array([])

def compute_fit_score(emb1: np.ndarray, emb2: np.ndarray) -> float:
    sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    return ((sim + 1) / 2) * 100

def upsert_resume(resume_id: str, resume_emb: np.ndarray):
    index.upsert(vectors=[(resume_id, resume_emb.tolist())])

def query_index(query_emb: np.ndarray, top_k: int = 1):
    return index.query(vector=query_emb.tolist(), top_k=top_k)

# ----------------------------------------------------------------------------- 
# Streamlit UI 
# ----------------------------------------------------------------------------- 
def main():
    st.title("Job Fit Score Calculator")
    st.write("Upload a job description and resume to compute a job-fit percentage.")
    st.warning("Model availability may depend on API uptime. Use Offline mode for reliability.")

    st.subheader("Upload Job Description")
    jd_file = st.file_uploader("Upload JD (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"], key="jd")

    st.subheader("Upload Resume")
    resume_file = st.file_uploader("Upload Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"], key="resume")

    if jd_file:
        with st.expander("View Job Description Text"):
            st.write(extract_text(jd_file))
    if resume_file:
        with st.expander("View Resume Text"):
            st.write(extract_text(resume_file))

    if st.button("Calculate Fit Score"):
        if jd_file and resume_file:
            with st.spinner("Processing..."):
                jd_text = extract_text(jd_file)
                resume_text = extract_text(resume_file)
                if not jd_text or not resume_text:
                    st.error("Text extraction failed.")
                    return

                if mode == "Online":
                    jd_emb = get_embedding_online(jd_text)
                    resume_emb = get_embedding_online(resume_text)
                else:
                    jd_emb = get_embedding_offline(jd_text)
                    resume_emb = get_embedding_offline(resume_text)

                if jd_emb.size == 0 or resume_emb.size == 0:
                    st.error("Embedding error.")
                    return

                fit_score = compute_fit_score(resume_emb, jd_emb)
                st.success(f"Job Fit Score: {fit_score:.2f}%")

                resume_id = "resume_1"
                upsert_resume(resume_id, resume_emb)
                with st.expander("Pinecone Query Result"):
                    result = query_index(jd_emb, top_k=1)
                    st.write(result)
        else:
            st.error("Please upload both files.")

if __name__ == "__main__":
    main()
