import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec
import io
from huggingface_hub import InferenceClient

# -----------------------------------------------------------------------------
# Custom CSS for light purple theme and serene hues
# -----------------------------------------------------------------------------
custom_css = """
<style>
/* Set the main background color */
body {
    background-color: #f7f3ff;
}
/* Style the main container */
.css-1d391kg, .css-18e3th9 {
    background-color: #f7f3ff;
}
/* Style headings */
h1, h2, h3, h4, h5, h6 {
    color: #5e3d99;
}
/* Style Streamlit buttons */
div.stButton > button {
    background-color: #a985e2;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5em 1em;
    font-size: 16px;
}
div.stButton > button:hover {
    background-color: #8a6ac9;
}
/* Style file uploader */
.css-1emrehy.edgvbvh3 {
    background-color: #ffffff;
    border: 1px solid #d3cce3;
    border-radius: 8px;
}
/* Customize the expander header */
.css-1q1n0ol.e1fqkh3o1 {
    background-color: #d8b4f3;
    color: #5e3d99;
    border: none;
    border-radius: 4px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Configuration and Initialization
# -----------------------------------------------------------------------------
# Retrieve API keys and environment from Streamlit secrets.
# Your .streamlit/secrets.toml should contain:
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
# Use the new model:
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Returns 768-d embeddings
DESIRED_DIMENSION = 768  # Correct dimension for the chosen model

# Initialize Pinecone using the new SDK pattern.
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

# Check if the index exists and if its dimension matches.
existing_indexes = pc.list_indexes().names()
if INDEX_NAME in existing_indexes:
    # Get index details.
    desc = pc.describe_index(INDEX_NAME)
    if desc.dimension != DESIRED_DIMENSION:
        st.warning(f"Index dimension ({desc.dimension}) does not match desired dimension ({DESIRED_DIMENSION}). Recreating index.")
        pc.delete_index(INDEX_NAME)
        pc.create_index(
            name=INDEX_NAME,
            dimension=DESIRED_DIMENSION,
            metric="cosine",  # Using cosine similarity
            spec=spec
        )
else:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DESIRED_DIMENSION,
        metric="cosine",  # Using cosine similarity
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
    file.seek(0)  # Reset file pointer for further use

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
    Uses the Hugging Face InferenceClient to perform feature extraction,
    returning an embedding vector for the input text.
    This method caches the result to avoid repeated API calls.
    It also pools token-level embeddings if necessary to obtain a single vector.
    """
    client = InferenceClient(api_key=HF_API_KEY)
    try:
        result = client.feature_extraction(text, model=MODEL_NAME)
        embedding_array = np.array(result)
        # If the returned embedding is 2D (e.g., one embedding per token), pool across tokens.
        if embedding_array.ndim == 2:
            pooled_embedding = embedding_array.mean(axis=0)
        elif embedding_array.ndim == 1:
            pooled_embedding = embedding_array
        else:
            st.error("Unexpected embedding dimensions.")
            return np.array([])
        # Ensure the pooled embedding is list-like.
        if not isinstance(pooled_embedding.tolist(), list):
            st.error("Embedding is not a list-like structure.")
            return np.array([])
        return pooled_embedding
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return np.array([])

def compute_fit_score(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Computes cosine similarity between two embeddings and maps it to a percentage.
    """
    sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    return ((sim + 1) / 2) * 100  # Mapping from [-1, 1] to [0, 100]

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
                
                # Upsert the resume embedding into Pinecone (using a fixed ID for demonstration)
                resume_id = "resume_1"
                upsert_resume(resume_id, resume_emb)
                
                # Use an expander to hide Pinecone query details by default.
                with st.expander("Show Pinecone Query Details"):
                    result = query_index(jd_emb, top_k=1)
                    st.write(result)
        else:
            st.error("Please upload both a resume and a job description file.")

if __name__ == "__main__":
    main()
