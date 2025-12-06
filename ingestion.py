import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
from langchain_core.documents import Document
# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Use Google embeddings instead of OpenAI
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Commented out vector stores for debug mode
# chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
# vectorstore = PineconeVectorStore(
#     index_name="langchain-docs-2025", embedding=embeddings
# )

tavily_crawl = TavilyCrawl()


# Commented out for debug mode
# async def index_documents_async(documents: List[Document], batch_size: int = 50):
#     """Process documents in batches asynchronously."""
#     log_header("VECTOR STORAGE PHASE")
#     log_info(
#         f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
#         Colors.DARKCYAN,
#     )
#
#     # Create batches
#     batches = [
#         documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
#     ]
#
#     log_info(
#         f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
#     )
#
#     # Process all batches concurrently
#     async def add_batch(batch: List[Document], batch_num: int):
#         try:
#             await vectorstore.aadd_documents(batch)
#             log_success(
#                 f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
#             )
#         except Exception as e:
#             log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
#             return False
#         return True
#
#     # Process batches concurrently
#     tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
#     results = await asyncio.gather(*tasks, return_exceptions=True)
#
#     # Count successful batches
#     successful = sum(1 for result in results if result is True)
#
#     if successful == len(batches):
#         log_success(
#             f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
#         )
#     else:
#         log_warning(
#             f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
#         )


async def main():
    """Main async function focused on Tavily crawl debugging."""
    log_header("TAVILY CRAWL DEBUG MODE")

    log_info(
        "🔍 TavilyCrawl: Starting to crawl documentation from https://python.langchain.com/",
        Colors.PURPLE,
    )

    tavily_crawl_results = tavily_crawl.invoke(
        input={
            "url": "https://python.langchain.com/",
            "extract_depth": "advanced",
            "instructions": "Documentation relevant to ai agents",
            "max_depth": 2,
        }
    )
    
    if tavily_crawl_results.get("error"):
        log_error(f"TavilyCrawl: {tavily_crawl_results['error']}")
        return
    else:
        log_success(
            f"TavilyCrawl: Successfully crawled {len(tavily_crawl_results['results'])} URLs from documentation site"
        )

    # Debug: Print detailed information about crawled URLs
    log_header("CRAWL RESULTS DETAILS")
    all_docs = []
    for idx, tavily_crawl_result_item in enumerate(tavily_crawl_results["results"], 1):
        url = tavily_crawl_result_item['url']
        content_length = len(tavily_crawl_result_item['raw_content'])
        
        log_info(
            f"📄 [{idx}/{len(tavily_crawl_results['results'])}] URL: {url}",
            Colors.YELLOW
        )
        log_info(f"    Content length: {content_length} characters")
        
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": url},
            )
        )

    # Commented out for debug mode - Document chunking
    # log_header("DOCUMENT CHUNKING PHASE")
    # log_info(
    #     f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
    #     Colors.YELLOW,
    # )
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    # splitted_docs = text_splitter.split_documents(all_docs)
    # log_success(
    #     f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    # )

    # Commented out for debug mode - Vector indexing
    # await index_documents_async(splitted_docs, batch_size=500)

    log_header("DEBUG SUMMARY")
    log_success("🎉 Crawl debugging complete!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Total URLs crawled: {len(tavily_crawl_results['results'])}")
    log_info(f"   • Documents created: {len(all_docs)}")
    log_info(f"   • Total content size: {sum(len(doc.page_content) for doc in all_docs)} characters")
    
    # Optional: Show first document preview
    if all_docs:
        log_header("SAMPLE DOCUMENT PREVIEW")
        log_info(f"First document URL: {all_docs[0].metadata['source']}")
        log_info(f"First 500 characters:\n{all_docs[0].page_content[:500]}...")


if __name__ == "__main__":
    asyncio.run(main())