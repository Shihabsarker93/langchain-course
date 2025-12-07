import os
from dotenv import load_dotenv

from typing import Any, Dict, List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "langchain-doc-huggingface")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env")

# ============================================================
# EMBEDDINGS (must match ingestion — HuggingFace 384-dim)
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)

# ============================================================
# RETRIEVER (Pinecone)
# ============================================================

docsearch = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

retriever = docsearch.as_retriever()

# ============================================================
# LLM (Google Gemini)
# ============================================================

chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

# ============================================================
# PROMPT TEMPLATE (replacement for Hub prompt)
# ============================================================

retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Use the context below to answer the user question.\n"
         "If the answer is not in the context, say you don't know.\n"
         "Keep the answer short and correct.\n\n"
         "Context:\n{context}"
        ),
        ("human", "{input}")
    ]
)


# ============================================================
# HELPER — format documents
# ============================================================

def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================
# RAG CHAIN
# ============================================================

qa_chain = (
    {
        "context": retriever | RunnableLambda(_format_docs),
        "input": RunnablePassthrough(),
    }
    | retrieval_prompt
    | chat
    | StrOutputParser()
)


def run_llm(query: str) -> str:
    """Run Retrieval-Augmented Generation."""
    return qa_chain.invoke(query)


if __name__ == "__main__":
    print(run_llm("What is a LangChain chain?"))
