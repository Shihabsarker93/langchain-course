import asyncio
import os
import ssl
from typing import Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (
    Colors,
    log_error,
    log_header,
    log_info,
    log_success,
    log_warning,
)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    log_error("Missing GOOGLE_API_KEY. Set it in your .env file to run ingestion.")
    raise SystemExit(1)

# Base configuration (override with environment variables if needed)
BASE_URL = os.getenv("DOCS_BASE_URL", "https://python.langchain.com/")
TAVILY_INSTRUCTIONS = os.getenv(
    "TAVILY_INSTRUCTIONS", "Documentation relevant to AI agents"
)
MAX_CRAWL_DEPTH = int(os.getenv("TAVILY_MAX_DEPTH", "2"))
MAP_URL_LIMIT = int(os.getenv("TAVILY_URL_LIMIT", "50"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "4000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200"))
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "langchain-docs-2025")

# Embeddings powered by Google GenAI
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
    task_type="retrieval_document",
)

# Vector stores (local + optional Pinecone)
vectorstores: Dict[str, object] = {}
if os.getenv("ENABLE_CHROMA", "").lower() in {"1", "true", "yes"}:
    try:
        from langchain_chroma import Chroma

        vectorstores["chroma"] = Chroma(
            persist_directory=CHROMA_DIR, embedding_function=embeddings
        )
        log_success(f"Chroma vector store ready at '{CHROMA_DIR}'")
    except Exception as exc:  # pragma: no cover - defensive
        log_warning(f"Chroma init skipped: {exc}")
else:
    log_warning("Chroma disabled (set ENABLE_CHROMA=1 to enable)")

if os.getenv("PINECONE_API_KEY"):
    try:
        vectorstores["pinecone"] = PineconeVectorStore(
            index_name=PINECONE_INDEX, embedding=embeddings
        )
        log_success(f"Pinecone vector store ready (index '{PINECONE_INDEX}')")
    except Exception as exc:  # pragma: no cover - defensive
        log_warning(f"Pinecone init skipped: {exc}")
else:
    log_warning("PINECONE_API_KEY not set; skipping Pinecone vector store")

tavily_crawl = TavilyCrawl()
tavily_map = TavilyMap()
tavily_extract = TavilyExtract()


def dedupe_urls(urls: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen = {}
    for url in urls:
        if url not in seen:
            seen[url] = None
    return list(seen.keys())


def map_site_urls() -> List[str]:
    """Use Tavily Map to discover relevant URLs before extraction."""
    log_header("TAVILY MAP PHASE")
    log_info(
        f"🔗 TavilyMap: Mapping {BASE_URL} (depth={MAX_CRAWL_DEPTH}, limit={MAP_URL_LIMIT})",
        Colors.PURPLE,
    )
    map_results = tavily_map.invoke(
        input={
            "url": BASE_URL,
            "instructions": TAVILY_INSTRUCTIONS,
            "max_depth": MAX_CRAWL_DEPTH,
            "limit": MAP_URL_LIMIT,
            "categories": ["Documentation"],
        }
    )

    if map_results.get("error"):
        log_error(f"TavilyMap: {map_results['error']}")
        return []

    urls: List[str] = []
    for item in map_results.get("results", []):
        if isinstance(item, dict):
            url = item.get("url")
        else:
            url = item
        if url:
            urls.append(url)

    urls = dedupe_urls(urls)
    log_success(f"TavilyMap: Discovered {len(urls)} URLs")
    return urls


def extract_documents(urls: List[str]) -> List[Document]:
    """Pull page content with Tavily Extract."""
    if not urls:
        return []

    log_header("TAVILY EXTRACT PHASE")
    log_info(f"📥 TavilyExtract: Fetching content for {len(urls)} URLs", Colors.PURPLE)
    extract_results = tavily_extract.invoke(
        input={
            "urls": urls,
            "extract_depth": "advanced",
            "include_images": False,
            "include_favicon": False,
        }
    )

    if extract_results.get("error"):
        log_error(f"TavilyExtract: {extract_results['error']}")
        return []

    results = extract_results.get("results", [])
    failed = extract_results.get("failed_results", [])

    if failed:
        log_warning(f"TavilyExtract: Failed to extract {len(failed)} URLs")

    documents: List[Document] = []
    for result in results:
        url = result.get("url") or result.get("source") or "unknown"
        content = (
            result.get("raw_content")
            or result.get("content")
            or result.get("markdown")
            or result.get("text")
        )
        if not content:
            log_warning(f"TavilyExtract: No content for {url}")
            continue

        documents.append(Document(page_content=content, metadata={"source": url}))

    log_success(f"TavilyExtract: Created {len(documents)} documents")
    return documents


def crawl_documents() -> List[Document]:
    """Fallback crawl that also retrieves raw content."""
    log_header("TAVILY CRAWL PHASE")
    log_info(
        f"🕷️  TavilyCrawl: Crawling {BASE_URL} (depth={MAX_CRAWL_DEPTH})",
        Colors.PURPLE,
    )
    tavily_crawl_results = tavily_crawl.invoke(
        input={
            "url": BASE_URL,
            "extract_depth": "advanced",
            "instructions": TAVILY_INSTRUCTIONS,
            "max_depth": MAX_CRAWL_DEPTH,
        }
    )

    if tavily_crawl_results.get("error"):
        log_error(f"TavilyCrawl: {tavily_crawl_results['error']}")
        return []

    crawl_results = tavily_crawl_results.get("results", [])
    documents: List[Document] = []
    for item in crawl_results:
        url = item.get("url")
        content = item.get("raw_content")
        if not url or not content:
            continue
        documents.append(Document(page_content=content, metadata={"source": url}))

    log_success(f"TavilyCrawl: Retrieved {len(documents)} documents")
    return documents


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    if not vectorstores:
        log_warning("No vector stores configured. Skipping indexing.")
        return

    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 Preparing to add {len(documents)} documents to vector stores",
        Colors.DARKCYAN,
    )

    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]
    log_info(
        f"📦 Split into {len(batches)} batches of up to {batch_size} documents each"
    )

    async def add_to_store(name: str, store: object, batch: List[Document]) -> bool:
        try:
            # Some stores lack async helpers; push to a thread to avoid blocking.
            await asyncio.to_thread(store.add_documents, batch)  # type: ignore[attr-defined]
            log_success(
                f"VectorStore ({name}): Added batch with {len(batch)} documents"
            )
            return True
        except Exception as exc:
            log_error(f"VectorStore ({name}): Failed to add batch - {exc}")
            return False

    successful_batches = 0
    for batch in batches:
        tasks = [
            add_to_store(name, store, batch) for name, store in vectorstores.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        if all(result is True for result in results):
            successful_batches += 1

    if successful_batches == len(batches):
        log_success(
            f"VectorStore: All batches processed successfully ({successful_batches}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore: {successful_batches}/{len(batches)} batches processed successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    urls = map_site_urls()
    documents = extract_documents(urls)

    if not documents:
        log_warning("Extract returned no documents; falling back to TavilyCrawl.")
        documents = crawl_documents()

    if not documents:
        log_error("No documents available for chunking and indexing.")
        return

    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(documents)} documents "
        f"(chunk={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    splitted_docs = text_splitter.split_documents(documents)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(documents)} documents"
    )

    await index_documents_async(splitted_docs, batch_size=BATCH_SIZE)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • URLs mapped: {len(urls)}")
    log_info(f"   • Documents extracted: {len(documents)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
