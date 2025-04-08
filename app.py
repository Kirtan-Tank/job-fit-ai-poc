import os
import io
import shutil
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Monkey Patch: Define init_empty_weights if not defined
# -----------------------------------------------------------------------------
try:
    init_empty_weights  # check if already defined
except NameError:
    try:
        from transformers.modeling_utils import init_empty_weights
    except ImportError:
        def init_empty_weights(*args, **kwargs):
            return None
        import torch
        torch.init_empty_weights = init_empty_weights

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
# Sidebar: Mode and Model Selection
# -----------------------------------------------------------------------------
mode = st.sidebar.radio("Select Mode", ["Online", "On-Demand"])
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
    st.sidebar.markdown("<span style='color: #ffffff;'>On-Demand mode uses the local SentenceTransformer model.</span>", unsafe_allow_html=True)
    # In on-demand mode, we use the SentenceTransformer model ID directly.
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    on_demand_status = st.sidebar.empty()
    on_demand_status.info("Downloading and caching model. Please wait...")

    @st.cache_resource(show_spinner=False)
    def load_on_demand_model() -> SentenceTransformer:
        try:
            # Try loading the model (from cache or downloading if needed)
            model = SentenceTransformer(MODEL_NAME)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}. Clearing cache and retrying...")
            # Clear Hugging Face cache directory to force re-download.
            cache_dir = os.path.expanduser("~/.cache/huggingface")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            model = SentenceTransformer(MODEL_NAME)
            return model

    with st.spinner("Downloading and caching model. This may take several minutes..."):
        on_demand_model = load_on_demand_model()
    if on_demand_model is not None:
        on_demand_status.success("Model loaded successfully!")
    else:
        on_demand_status.error("Model loading failed.")

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

@st.cache_data(show_spinner=False)
def get_embedding_online(text: str) -> np.ndarray:
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
        return pooled_embedding
    except Exception as e:
        if "503" in str(e):
            st.error("The selected model is temporarily unavailable due to third-party service issues. Please try another model or try again later.")
        else:
            st.error(f"Error generating online embedding: {e}")
        return np.array([])

def get_embedding_on_demand(text: str) -> np.ndarray:
    try:
        result = on_demand_model.encode(text)
        embedding_array = np.array(result)
        if embedding_array.ndim == 2:
            pooled_embedding = embedding_array.mean(axis=0)
        elif embedding_array.ndim == 1:
            pooled_embedding = embedding_array
        else:
            st.error("Unexpected embedding dimensions.")
            return np.array([])
        return pooled_embedding
    except Exception as e:
        st.error(f"Error generating on-demand embedding: {e}")
        return np.array([])

def compute_fit_score(emb1: np.ndarray, emb2: np.ndarray) -> float:
    sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    return ((sim + 1) / 2) * 100

def upsert_resume(resume_id: str, resume_emb: np.ndarray):
    index.upsert(vectors=[(resume_id, resume_emb.tolist())])

def query_index(query_emb: np.ndarray, top_k: int = 1):
    return index.query(vector=query_emb.tolist(), top_k=top_k)

# -----------------------------------------------------------------------------
# Robust Similarity Calculation Example (for testing)
# -----------------------------------------------------------------------------
def test_similarity():
    model_to_use = None
    if mode == "Online":
        model_to_use = SentenceTransformer(MODEL_NAME)
    else:
        model_to_use = on_demand_model

    sentences = [
        "That is a happy person",
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day"
    ]
    embeddings = model_to_use.encode(sentences)
    sims = cosine_similarity(embeddings)
    st.write("Similarity matrix shape:", sims.shape)
    st.write(sims)

# -----------------------------------------------------------------------------
# Streamlit User Interface
# -----------------------------------------------------------------------------
def main():
    st.title("Job Fit Score Calculator")
    st.write("Upload a job description document and a resume (or CV) to calculate a job fit score based on semantic similarity.")
    st.warning("Note: Model availability depends on third-party API uptime. If the selected model is unavailable, try another model from the sidebar.")

    # Optionally, show a test similarity matrix.
    if st.button("Run Similarity Test"):
        test_similarity()

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
                if mode == "Online":
                    jd_emb = get_embedding_online(jd_text)
                    resume_emb = get_embedding_online(resume_text)
                else:
                    jd_emb = get_embedding_on_demand(jd_text)
                    resume_emb = get_embedding_on_demand(resume_text)
                if jd_emb.size == 0 or resume_emb.size == 0:
                    st.error("Embedding generation failed. Please check your inputs and API configuration.")
                    return
                fit_score = compute_fit_score(resume_emb, jd_emb)
                st.success(f"Job Fit Score: {fit_score:.2f}%")
                resume_id = "resume_1"  # Modify as needed for unique IDs.
                upsert_resume(resume_id, resume_emb)
                with st.expander("Show Pinecone Query Result"):
                    result = query_index(jd_emb, top_k=1)
                    st.write(result)
        else:
            st.error("Please upload both a job description and a resume (or CV).")

if __name__ == "__main__":
    main()
