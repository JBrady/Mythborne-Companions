import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # For local embeddings
import time # Optional: to time the process

# --- Configuration ---
KNOWLEDGE_BASE_DIR = "knowledge_base" # Folder containing your documents
CHROMA_PERSIST_DIR = "chroma_db"      # Folder where the DB will be saved
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Efficient local model

# --- 1. Load Documents ---
start_time = time.time()
print(f"Loading documents from: {KNOWLEDGE_BASE_DIR}")
# Using DirectoryLoader with unstructured to handle various file types
# Consider adding specific loaders if DirectoryLoader struggles with certain files
loader = DirectoryLoader(
    KNOWLEDGE_BASE_DIR,
    glob="**/*.*", # Load all files recursively
    show_progress=True,
    use_multithreading=True # Use multiple threads for loading if possible
)
documents = loader.load()

if not documents:
    print(f"No documents found in '{KNOWLEDGE_BASE_DIR}'. Please add some text files.")
    exit()

load_end_time = time.time()
print(f"Loaded {len(documents)} documents in {load_end_time - start_time:.2f} seconds.")

# --- 2. Split Documents into Chunks ---
print("Splitting documents into chunks...")
# RecursiveCharacterTextSplitter tries to split based on common separators like newlines
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # Target size for each chunk
    chunk_overlap=200  # Overlap between chunks to maintain context
)
all_splits = text_splitter.split_documents(documents)
split_end_time = time.time()
print(f"Split into {len(all_splits)} chunks in {split_end_time - load_end_time:.2f} seconds.")

# --- 3. Initialize Local Embedding Model ---
# This will download the model (few hundred MB) from Hugging Face the first time.
print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
# Use device='cpu' explicitly if you don't have a CUDA-enabled GPU or run into issues.
# If you have a compatible GPU and PyTorch with CUDA installed, you might try 'cuda'.
model_kwargs = {'device': 'cpu'}
# Normalize embeddings can sometimes help similarity search, depending on the model.
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
embed_init_end_time = time.time()
print(f"Embedding model initialized in {embed_init_end_time - split_end_time:.2f} seconds.")

# --- 4. Create and Persist ChromaDB Vector Store ---
# This step generates embeddings for all text chunks and saves them.
# It can take significant time depending on CPU/GPU and number of chunks.
print(f"Creating/updating ChromaDB vector store at: {CHROMA_PERSIST_DIR}")
print(f"Embedding {len(all_splits)} chunks... This may take a while...")

# If the directory already exists, Chroma.from_documents typically loads existing data
# and potentially adds new/changed documents. Behavior might vary slightly by version.
# For a guaranteed fresh build, manually delete the CHROMA_PERSIST_DIR folder first.
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    persist_directory=CHROMA_PERSIST_DIR
)

embed_end_time = time.time()
print(f"Embedding completed in {embed_end_time - embed_init_end_time:.2f} seconds.")

print("Ensuring persistence...")
vectorstore.persist()
persist_end_time = time.time()
print(f"ChromaDB index persisted in {persist_end_time - embed_end_time:.2f} seconds.")
total_time = persist_end_time - start_time
print(f"Total indexing time: {total_time:.2f} seconds.")