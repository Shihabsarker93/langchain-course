"""
Documentation Ingestion Pipeline with Free HuggingFace Embeddings
Fetches documentation, chunks it, and stores in vector databases
"""

import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

# Tavily configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
BASE_URL = "https://python.langchain.com/"
MAX_URLS = 50

# Chunking configuration
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200

# Vector store configuration
ENABLE_CHROMA = os.getenv("ENABLE_CHROMA", "0") == "1"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "langchain-doc-huggingface")

# ============================================================
# INITIALIZE EMBEDDINGS (FREE - NO API KEY NEEDED)
# ============================================================

print("=" * 60)
print("🔧 INITIALIZING EMBEDDINGS")
print("=" * 60)
print("ℹ️  Loading HuggingFace embeddings model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("✅ HuggingFace embeddings loaded successfully!")
print("ℹ️  Embedding dimension: 384")

# ============================================================
# INITIALIZE VECTOR STORES
# ============================================================

print("=" * 60)
print("🔧 INITIALIZING VECTOR STORES")
print("=" * 60)

vector_stores = []

# Chroma setup (local)
if ENABLE_CHROMA:
    try:
        from langchain_chroma import Chroma
        
        chroma_store = Chroma(
            collection_name="langchain-docs",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        vector_stores.append(("chroma", chroma_store))
        print("✅ Chroma vector store ready (using HuggingFace embeddings)")
    except ImportError:
        print("⚠️  Chroma not installed (pip install langchain-chroma)")
else:
    print("⚠️  Chroma disabled (set ENABLE_CHROMA=1 to enable)")

# Pinecone setup (cloud)
if PINECONE_API_KEY:
    try:
        from langchain_pinecone import PineconeVectorStore
        
        pinecone_store = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
        )
        vector_stores.append(("pinecone", pinecone_store))
        print(f"✅ Pinecone vector store ready")
        print(f"   • Index: {PINECONE_INDEX_NAME}")
        print(f"   • Embeddings: HuggingFace (dimension 384)")
    except Exception as e:
        print(f"❌ Pinecone initialization failed: {e}")
else:
    print("⚠️  Pinecone disabled (no API key found)")

if not vector_stores:
    raise ValueError("❌ No vector stores configured! Enable at least one.")

# ============================================================
# PHASE 1: DISCOVER URLS using Tavily
# ============================================================

def discover_urls(base_url: str, max_results: int = 20) -> list[str]:
    """Discover documentation URLs using Tavily search"""
    print("=" * 60)
    print("🚀 PHASE 1: DISCOVERING URLS")
    print("=" * 60)
    
    print(f"ℹ️  Searching for documentation pages...")
    
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    
    # Search for documentation pages
    search_query = f"site:{base_url} documentation"
    
    try:
        response = tavily_client.search(
            query=search_query,
            max_results=max_results,
            search_depth="advanced"
        )
        
        # Extract URLs from results
        urls = [result['url'] for result in response.get('results', [])]
        
        # Filter to only include URLs from the base domain
        filtered_urls = [url for url in urls if base_url.replace('https://', '').replace('http://', '') in url]
        
        print(f"✅ Discovered {len(filtered_urls)} URLs from {base_url}")
        
        return filtered_urls
        
    except Exception as e:
        print(f"⚠️  Tavily search failed: {e}")
        # Fallback: return base URL
        return [base_url]

# ============================================================
# PHASE 2: EXTRACT - Load content from URLs
# ============================================================

def extract_content(urls: list[str]) -> list:
    """Extract content from discovered URLs using WebBaseLoader"""
    print("=" * 60)
    print("🚀 PHASE 2: EXTRACTING CONTENT")
    print("=" * 60)
    
    print(f"ℹ️  Loading content from {len(urls)} URLs...")
    
    all_docs = []
    failed = 0
    
    for idx, url in enumerate(urls, 1):
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"   ✓ [{idx}/{len(urls)}] {url[:60]}...")
        except Exception as e:
            failed += 1
            print(f"   ✗ [{idx}/{len(urls)}] Failed: {url[:60]}...")
    
    print(f"✅ Loaded {len(all_docs)} documents ({failed} failed)")
    return all_docs

# ============================================================
# PHASE 3: CHUNK - Split documents
# ============================================================

def chunk_documents(documents: list) -> list:
    """Split documents into smaller chunks"""
    print("=" * 60)
    print("🚀 PHASE 3: CHUNKING DOCUMENTS")
    print("=" * 60)
    
    print(f"ℹ️  Chunk size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks

# ============================================================
# PHASE 4: STORE - Add to vector databases
# ============================================================

def store_in_vectordb(chunks: list, batch_size: int = 100):
    """Store chunks in configured vector databases"""
    print("=" * 60)
    print("🚀 PHASE 4: STORING IN VECTOR DB")
    print("=" * 60)
    
    if not chunks:
        print("⚠️  No chunks to store!")
        return
    
    print(f"ℹ️  Total chunks to store: {len(chunks)}")
    
    num_batches = (len(chunks) + batch_size - 1) // batch_size
    print(f"ℹ️  Processing in {num_batches} batches ({batch_size} chunks/batch)")
    
    for store_name, store in vector_stores:
        print(f"\n📤 Uploading to {store_name}...")
        success_count = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                store.add_documents(batch)
                success_count += 1
                print(f"   ✓ Batch {batch_num}/{num_batches} uploaded ({len(batch)} chunks)")
            except Exception as e:
                print(f"   ✗ Batch {batch_num}/{num_batches} failed: {str(e)[:100]}")
        
        if success_count == num_batches:
            print(f"✅ {store_name}: All {num_batches} batches uploaded successfully!")
        else:
            print(f"⚠️  {store_name}: {success_count}/{num_batches} batches succeeded")

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """Run the complete ingestion pipeline"""
    print("\n" + "=" * 60)
    print("🚀 DOCUMENTATION INGESTION PIPELINE")
    print("=" * 60)
    print()
    
    try:
        # Phase 1: Discover URLs
        urls = discover_urls(BASE_URL, MAX_URLS)
        
        if not urls:
            print("❌ No URLs discovered. Exiting.")
            return
        
        # Phase 2: Extract content
        documents = extract_content(urls)
        
        if not documents:
            print("❌ No documents extracted. Exiting.")
            return
        
        # Phase 3: Chunk documents
        chunks = chunk_documents(documents)
        
        if not chunks:
            print("❌ No chunks created. Exiting.")
            return
        
        # Phase 4: Store in vector DB
        store_in_vectordb(chunks)
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETE")
        print("=" * 60)
        print(f"📊 Summary:")
        print(f"   • URLs discovered: {len(urls)}")
        print(f"   • Documents extracted: {len(documents)}")
        print(f"   • Chunks created: {len(chunks)}")
        print(f"   • Vector stores updated: {len(vector_stores)}")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ PIPELINE FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()