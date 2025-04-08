import os
import io
import shutil
import socket
import ssl
import pathlib

import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import InferenceClient, whoami
from sentence_transformers import SentenceTransformer

# ────────────────────────────────────────────────────────────────────────────────
# 0. Set offline flags so HF calls fail fast if no cache
# ────────────────────────────────────────────────────────────────────────────────
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ────────────────────────────────────────────────────────────────────────────────
# 1. Startup Diagnostics
# ────────────────────────────────────────────────────────────────────────────────
def test_hf_connectivity(host="huggingface.co", port=443, timeout=5):
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host):
                return True, None
    except Exception as e:
        return False, str(e)

def test_hf_token():
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_API_TOKEN")
    if not token:
        return False, "No HF token found in env (HUGGINGFACE_HUB_TOKEN or HF_API_TOKEN)."
    try:
        info = whoami(token=token)
        return True, f"Token valid for user: {info['name']}"
    except Exception as e:
        return False, f"Token invalid or expired: {e}"

def test_local_cache(model_id="sentence-transformers/all-MiniLM-L6-v2"):
    cache_root = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    model_folder = cache_root / model_id.replace("/", "--")
    expected = ["config.json", "pytorch_model.bin"]
    missing = [f for f in expected if not (model_folder / f).exists()]
    if missing:
        return False, f"Missing files in cache: {missing} under {model_folder}"
    return True, f"Found cached model files under {model_folder}"

st.set_page_config(page_title="Startup Diagnostics", layout="wide")
with st.expander("🔧 Startup Diagnostic Report", expanded=True):
    st.write("This diagnostic runs on every cold start to isolate connectivity, auth, and cache issues.")
    ok, msg = test_hf_connectivity()
    st.write("**1. HF Hub Reachability:**", "✅" if ok else "❌", msg)
    ok, msg = test_hf_token()
    st.write("**2. HF Token Check:**", "✅" if ok else "❌", msg)
    ok, msg = test_local_cache()
    st.write("**3. Local Cache Inspection:**", "✅" if ok else "❌", msg)

# ────────────────────────────────────────────────────────────────────────────────
# Monkey Patch: Define init_empty_weights if not defined
# ────────────────────────────────────────────────────────────────────────────────
try:
    init_empty_weights
except NameError:
    try:
        from transformers.modeling_utils import init_empty_weights
    except ImportError:
        def init_empty_weights(*args, **kwargs):
            return None
        import torch
        torch.init_empty_weights = init_empty_weights

# ────────────────────────────────────────────────────────────────────────────────
# Custom CSS Theme 
# ────────────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────────────
# Sidebar: Mode and Model Selection
# ────────────────────────────────────────────────────────────────────────────────
mode = st.sidebar.radio("Select Mode", ["Online", "On-Demand"])
if mode == "Online":
    st.sidebar.markdown(
        "<span style='color: #ffffff;'>Online mode uses the Hugging Face Inference API for embeddings.</span>",
        unsafe_allow_html=True,
    )
    model_options = {
        "Model 1": "sentence-transformers/all-MiniLM-L6-v2",
        "Model 2": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "Model 3": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "Model 4": "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "Model 5": "sentence-transformers/all-MiniLM-L12-v2",
    }
    selected_model_label = st.sidebar.selectbox("Select a model", list(model_options.keys()))
    MODEL_NAME = model_options[selected_model_label]
else:
    st.sidebar.markdown(
        "<span style='color: #ffffff;'>On-Demand mode uses local SentenceTransformer inference.</span>",
        unsafe_allow_html=True,
    )
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    on_demand_status = st.sidebar.empty()
    on_demand_status.info("Loading model from cache...")

    @st.cache_resource(show_spinner=False)
    def load_on_demand_model() -> SentenceTransformer:
        # Will fail fast if not in cache
        return SentenceTransformer(MODEL_NAME)

    with st.spinner("Loading model from cache..."):
        try:
            on_demand_model = load_on_demand_model()
            on_demand_status.success("Model loaded successfully!")
        except Exception as e:
            on_demand_status.error(f"Model load failed: {e}")
            st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# Pinecone Setup 
# ────────────────────────────────────────────────────────────────────────────────
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
        st.warning(
            f"Index dimension ({desc.dimension}) does not match desired dimension ({DESIRED_DIMENSION}). Recreating index."
        )
        pc.delete_index(INDEX_NAME)
        pc.create_index(name=INDEX_NAME, dimension=DESIRED_DIMENSION, metric="cosine", spec=spec)
else:
    pc.create_index(name=INDEX_NAME, dimension=DESIRED_DIMENSION, metric="cosine", spec=spec)
index = pc.Index(INDEX_NAME)

# ────────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ────────────────────────────────────────────────────────────────────────────────
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
def get_embedding_online(text: str) -> np.ndarray:
    client = InferenceClient(api_key=HF_API_KEY)
    try:
        result = client.feature_extraction(text, model=MODEL_NAME)
        arr = np.array(result)
        if arr.ndim == 2:
            return arr.mean(axis=0)
        return arr
    except Exception as e:
        if "503" in str(e):
            st.error("Selected model temporarily unavailable. Try again later.")
        else:
            st.error(f"Online embedding error: {e}")
        return np.array([])

def get_embedding_on_demand(text: str) -> np.ndarray:
    try:
        arr = np.array(on_demand_model.encode(text))
        if arr.ndim == 2:
            return arr.mean(axis=0)
        return arr
    except Exception as e:
        st.error(f"On‑demand embedding error: {e}")
        return np.array([])

def compute_fit_score(emb1: np.ndarray, emb2: np.ndarray) -> float:
    sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    return ((sim + 1) / 2) * 100

def upsert_resume(resume_id: str, resume_emb: np.ndarray):
    index.upsert(vectors=[(resume_id, resume_emb.tolist())])

def query_index(query_emb: np.ndarray, top_k: int = 1):
    return index.query(vector=query_emb.tolist(), top_k=top_k)

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit User Interface
# ────────────────────────────────────────────────────────────────────────────────
def main():
    st.title("Job Fit Score Calculator")
    st.write("Upload a job description and a resume to compute a semantic fit score.")
    st.warning("If the model isn’t loading, check the diagnostic report above.")

    if st.button("Run Similarity Test"):
        test_sents = [
            "That is a happy person",
            "That is a happy dog",
            "That is a very happy person",
            "Today is a sunny day",
        ]
        emb = (
            SentenceTransformer(MODEL_NAME).encode(test_sents)
            if mode == "Online"
            else on_demand_model.encode(test_sents)
        )
        sims = cosine_similarity(emb)
        st.write("Similarity matrix:", sims)

    st.subheader("Upload Job Description")
    jd_file = st.file_uploader("Job Description (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], key="jd")
    st.subheader("Upload Resume/CV")
    resume_file = st.file_uploader("Resume/CV (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], key="resume")

    if jd_file:
        with st.expander("Extracted JD Text"):
            st.write(extract_text(jd_file))
    if resume_file:
        with st.expander("Extracted Resume Text"):
            st.write(extract_text(resume_file))

    if st.button("Calculate Fit Score"):
        if not (jd_file and resume_file):
            st.error("Please upload both JD and Resume.")
            return
        jd_text = extract_text(jd_file)
        resume_text = extract_text(resume_file)
        if not jd_text or not resume_text:
            st.error("Failed to extract text from one of the files.")
            return

        if mode == "Online":
            jd_emb = get_embedding_online(jd_text)
            resume_emb = get_embedding_online(resume_text)
        else:
            jd_emb = get_embedding_on_demand(jd_text)
            resume_emb = get_embedding_on_demand(resume_text)

        if jd_emb.size == 0 or resume_emb.size == 0:
            st.error("Embedding generation failed.")
            return

        score = compute_fit_score(resume_emb, jd_emb)
        st.success(f"Job Fit Score: {score:.2f}%")

        upsert_resume("resume_1", resume_emb)
        with st.expander("Pinecone Query Result"):
            st.write(query_index(jd_emb, top_k=1))

if __name__ == "__main__":
    main()
