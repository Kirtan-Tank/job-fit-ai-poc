import os
import io
import socket
import ssl
import pathlib
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient, whoami, snapshot_download
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# -------------------------------------------------------------------
# Diagnostics
# -------------------------------------------------------------------
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
        return False, "No HF token found."
    try:
        info = whoami(token=token)
        return True, f"Token valid for user: {info['name']}"
    except Exception as e:
        return False, f"Token error: {e}"

def test_local_cache(model_id="sentence-transformers/all-MiniLM-L6-v2"):
    cache_root = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    model_folder = cache_root / model_id.replace("/", "--")
    expected = ["config.json", "pytorch_model.bin"]
    missing = [f for f in expected if not (model_folder / f).exists()]
    return (False, f"Missing: {missing}") if missing else (True, f"Cached at {model_folder}")

st.set_page_config(page_title="Startup Diagnostics", layout="wide")
with st.expander("🔧 Startup Diagnostic Report", expanded=True):
    st.write("Run diagnostics for connectivity, auth, and cache issues.")
    for label, func in zip(
        ["HF Reachability", "HF Token", "Local Cache"],
        [test_hf_connectivity, test_hf_token, test_local_cache]
    ):
        ok, msg = func()
        st.write(f"**{label}:**", "✅" if ok else "❌", msg)

# -------------------------------------------------------------------
# UI Theme
# -------------------------------------------------------------------
st.markdown(open("style.css").read(), unsafe_allow_html=True)  # Externalize CSS for clarity

# -------------------------------------------------------------------
# Sidebar Config
# -------------------------------------------------------------------
mode = st.sidebar.radio("Select Mode", ["Online", "On-Demand"])
if mode == "Online":
    st.sidebar.info("Using HF Inference API.")
    MODEL_NAME = st.sidebar.selectbox(
        "Model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L12-v2"
        ]
    )
else:
    st.sidebar.info("Using local model.")
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    @st.cache_resource(show_spinner=False)
    def load_model():
        path = snapshot_download(
            repo_id=MODEL_NAME,
            token=os.environ.get("HUGGINGFACE_HUB_TOKEN", None),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return SentenceTransformer(str(path))

    with st.spinner("Loading model..."):
        try:
            on_demand_model = load_model()
            st.sidebar.success("Model loaded.")
        except Exception as e:
            st.sidebar.error(f"Load error: {e}")
            st.stop()

# -------------------------------------------------------------------
# Pinecone Setup
# -------------------------------------------------------------------
HF_API_KEY = st.secrets["general"]["HF_API_KEY"]
PINECONE_API_KEY = st.secrets["general"]["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["general"]["PINECONE_ENV"]
INDEX_NAME = "job-fit-index"
DIM = 384

pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
existing = pc.list_indexes().names()
if INDEX_NAME not in existing or pc.describe_index(INDEX_NAME).dimension != DIM:
    if INDEX_NAME in existing:
        pc.delete_index(INDEX_NAME)
    pc.create_index(name=INDEX_NAME, dimension=DIM, metric="cosine", spec=spec)
index = pc.Index(INDEX_NAME)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def extract_text(file) -> str:
    if not file:
        return ""
    data = file.read()
    file.seek(0)
    if file.name.endswith(".pdf"):
        import PyPDF2
        return "\n".join(p.extract_text() or "" for p in PyPDF2.PdfReader(io.BytesIO(data)).pages)
    elif file.name.endswith(".docx"):
        import docx
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", errors="ignore")

@st.cache_data(show_spinner=False)
def get_embedding_online(text: str) -> np.ndarray:
    try:
        client = InferenceClient(api_key=HF_API_KEY)
        out = client.feature_extraction(text, model=MODEL_NAME)
        arr = np.array(out)
        return arr.mean(axis=0) if arr.ndim == 2 else arr
    except Exception as e:
        st.error(f"Online embed error: {e}")
        return np.array([])

def get_embedding_on_demand(text: str) -> np.ndarray:
    try:
        arr = on_demand_model.encode(text)
        return np.array(arr).mean(axis=0)
    except Exception as e:
        st.error(f"Local embed error: {e}")
        return np.array([])

def compute_fit_score(a: np.ndarray, b: np.ndarray) -> float:
    return ((cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0] + 1) / 2) * 100

def upsert_resume(id: str, emb: np.ndarray):
    index.upsert(vectors=[(id, emb.tolist())])

def query_index(emb: np.ndarray, top_k: int = 1):
    return index.query(vector=emb.tolist(), top_k=top_k)

# -------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------
def main():
    st.title("Job Fit Score Calculator")
    st.write("Upload a JD and a resume to compute fit.")

    if st.button("Run Similarity Test"):
        sents = ["happy person", "happy dog", "very happy person", "sunny day"]
        emb = (
            SentenceTransformer(MODEL_NAME).encode(sents)
            if mode == "Online" else on_demand_model.encode(sents)
        )
        st.write("Similarity matrix:", cosine_similarity(emb))

    jd_file = st.file_uploader("Job Description", type=["pdf", "docx", "txt"], key="jd")
    cv_file = st.file_uploader("Resume", type=["pdf", "docx", "txt"], key="cv")

    if jd_file and cv_file:
        jd_text, cv_text = extract_text(jd_file), extract_text(cv_file)
        jd_emb = get_embedding_online(jd_text) if mode == "Online" else get_embedding_on_demand(jd_text)
        cv_emb = get_embedding_online(cv_text) if mode == "Online" else get_embedding_on_demand(cv_text)

        if jd_emb.size and cv_emb.size:
            score = compute_fit_score(jd_emb, cv_emb)
            st.success(f"Semantic Fit Score: {score:.2f}%")
            upsert_resume("resume_001", cv_emb)
            result = query_index(jd_emb, top_k=1)
            st.info(f"Top match in index: {result.matches[0].id} (Score: {result.matches[0].score:.3f})")

if __name__ == "__main__":
    main()
