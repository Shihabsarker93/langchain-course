"""
Core RAG functionality - 100% FREE VERSION
Uses Ollama (local LLM) + HuggingFace embeddings + Pinecone
NO API KEYS NEEDED FOR LLM!
"""

import os
from typing import Any, Dict, List
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "langchain-doc-huggingface")

# Choose your LLM backend
USE_OLLAMA = True  # Set to True for 100% free local LLM
USE_GOOGLE = False  # Set to True if you want to use Google Gemini

# ============================================================
# INITIALIZE EMBEDDINGS (FREE - NO COST)
# ============================================================

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ Embeddings loaded")

# ============================================================
# INITIALIZE VECTOR STORE (PINECONE)
# ============================================================

print(f"Connecting to Pinecone index: {INDEX_NAME}")
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)
print("✅ Vector store connected")

# ============================================================
# INITIALIZE LLM (Choose one)
# ============================================================

def get_llm():
    """Get the configured LLM"""
    if USE_OLLAMA:
        print("Using Ollama (local, FREE)")
        return Ollama(
            model="llama3.2:latest",  # Using your installed model
            temperature=0
        )
    elif USE_GOOGLE:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("Using Google Gemini")
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    else:
        raise ValueError("No LLM configured! Set USE_OLLAMA=True or USE_GOOGLE=True")

# ============================================================
# PROMPTS
# ============================================================

SYSTEM_TEMPLATE = """You are a helpful assistant that answers questions about LangChain documentation.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep your answer concise and relevant to the question.

Context:
{context}

Question: {input}

Answer:"""

REPHRASE_TEMPLATE = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.

Chat History:
{chat_history}

Follow Up Input: {input}

Standalone question:"""

qa_prompt = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)
rephrase_prompt = ChatPromptTemplate.from_template(REPHRASE_TEMPLATE)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


def format_chat_history(chat_history: List[Dict[str, Any]]) -> str:
    """Format chat history into a string"""
    formatted = []
    for msg in chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


# ============================================================
# RAG FUNCTIONS
# ============================================================

def simple_query(query: str) -> Dict[str, Any]:
    """
    Simple RAG query without chat history
    
    Args:
        query: User's question
    
    Returns:
        Dict with 'answer' and 'context' keys
    """
    
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Build RAG chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    # Get context documents
    docs = retriever.invoke(query)
    
    # Get answer
    answer = rag_chain.invoke(query)
    
    return {
        "answer": answer,
        "context": docs,
        "input": query
    }


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Dict[str, Any]:
    """
    Run RAG query with chat history
    
    Args:
        query: User's question
        chat_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
    
    Returns:
        Dict with 'answer', 'context', and 'input' keys
    """
    
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # If there's chat history, rephrase the question
    if chat_history:
        formatted_history = format_chat_history(chat_history)
        
        rephrase_chain = (
            {
                "chat_history": lambda x: formatted_history,
                "input": RunnablePassthrough(),
            }
            | rephrase_prompt
            | llm
            | StrOutputParser()
        )
        
        standalone_question = rephrase_chain.invoke(query)
    else:
        standalone_question = query
    
    # Retrieve relevant documents
    docs = retriever.invoke(standalone_question)
    
    # Build answer chain
    answer_chain = (
        {
            "context": lambda x: format_docs(docs),
            "input": lambda x: query,
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    answer = answer_chain.invoke({})
    
    return {
        "answer": answer,
        "context": docs,
        "input": query
    }


def run_llm2(query: str, chat_history: List[Dict[str, Any]] = []) -> Dict[str, Any]:
    """
    Alternative LCEL implementation
    """
    
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    retrieve_docs_chain = (lambda x: x["input"]) | retriever
    
    chain = RunnablePassthrough.assign(
        context=retrieve_docs_chain
    ).assign(
        answer=rag_chain
    )
    
    result = chain.invoke({"input": query, "chat_history": chat_history})
    
    return result


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing RAG system...")
    print("=" * 60 + "\n")
    
    # Test 1: Simple query
    test_query = "What is LangChain?"
    
    print(f"Query: {test_query}")
    print("-" * 60)
    
    try:
        result = simple_query(test_query)
        print(f"Answer: {result['answer']}\n")
        print(f"Retrieved {len(result['context'])} documents")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
    
    # Test 2: Query with chat history
    print("Testing with chat history...")
    print("-" * 60)
    
    chat_history = [
        {"role": "user", "content": "What is LangChain?"},
        {"role": "assistant", "content": "LangChain is a framework for building applications with LLMs."}
    ]
    
    followup_query = "What are its main components?"
    
    print(f"Query: {followup_query}")
    print("-" * 60)
    
    try:
        result2 = run_llm(followup_query, chat_history)
        print(f"Answer: {result2['answer']}\n")
        print(f"Retrieved {len(result2['context'])} documents")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Error: {e}\n")
        import traceback
        traceback.print_exc()