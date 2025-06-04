import os
import pandas as pd
from langchain.schema import Document
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------------------
# CONFIG
# --------------------------------------------
DATA_PATH = "/Users/vasanthagullapalli/documents/rewireMe/onet_job_titles_with_descriptions.csv"
VECTOR_STORE_DIR = "/Users/vasanthagullapalli/documents/rewireMe/faiss_store"
EMBED_MODEL_NAME = "text-embedding-3-small"

# --------------------------------------------
# Load data
# --------------------------------------------
df = pd.read_csv(DATA_PATH)  # expects 'title' and 'description' columns
df["text"] = df["Job Title"] + " - " + df["Job Description"]

# --------------------------------------------
# Initialize embedding model
# --------------------------------------------
embedding_model = OpenAIEmbeddings(model=EMBED_MODEL_NAME)

# --------------------------------------------
# Step 1: Build FAISS vector store (Run once)
# --------------------------------------------
def build_vector_store():
    print("Building FAISS store...")
    documents = [
        Document(
            page_content=row["text"],
            metadata={"Job Title": row["Job Title"], "Job Description": row["Job Description"]}
        )
        for _, row in df.iterrows()
    ]
    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local(VECTOR_STORE_DIR)
    print(f"Saved FAISS vector store to {VECTOR_STORE_DIR}")

# --------------------------------------------
# Step 2: Load FAISS from disk and query
# --------------------------------------------
def search_similar_jobs(query_title: str, reason: str, k: int = 5):
    if not os.path.exists(VECTOR_STORE_DIR):
        raise ValueError("Vector store not found. Run build_vector_store() first.")

    vector_store = FAISS.load_local(VECTOR_STORE_DIR, embedding_model, allow_dangerous_deserialization=True)

    query = f"{query_title} - {reason}"
    results = vector_store.similarity_search_with_score(query, k=k)  # <-- use _with_score

    return [
        {
            "Job Title": doc.metadata["Job Title"],
            "Job Description": doc.metadata["Job Description"],
            "score": score  # include the actual similarity score
        }
        for doc, score in results
    ]

# --------------------------------------------
# EXAMPLE USAGE
# --------------------------------------------
if __name__ == "__main__":
    # Run once to build the index
    if not os.path.exists(VECTOR_STORE_DIR):
        build_vector_store()

    # Example search
    query_title = "Director of Technology"
    reason = "This role aligns with experience in technology and executive leadership. Manages large teams and oversees technology strategy."
    matches = search_similar_jobs(query_title, reason)

    print("Top matching job titles:")
    for m in matches:
        print(f"- {m['Job Title']}: {m['Job Description']}: {m['score']}")
