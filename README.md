## Contribution check
llams that i have locally :



shihab@Shihabs-MacBook-Air ~ % ollama list
NAME                       ID              SIZE      MODIFIED     
nomic-embed-text:latest    0a109f422b47    274 MB    5 weeks ago     
llama3:latest              365c0bd3c000    4.7 GB    5 weeks ago     
qwen2.5:7b                 845dbda0ea48    4.7 GB    3 months ago    
gemma3:270m                e7d36fb2c3b3    291 MB    3 months ago    
llama3.2:latest            a80c4f17acd5    2.0 GB    4 months ago    
shihab@Shihabs-MacBook-Air ~ % 

## AI Project Context
Use the following context for any AI assistant, collaborator, or future automation working in this repository.

```text
I am building a final-year thesis project: a high-accuracy, bilingual (Bangla-English) Retrieval-Augmented Generation (RAG) chatbot. Treat all future questions and technical decisions in this repository within the context of this system.

PROJECT GOAL:
This is not a casual chatbot. It is an end-to-end intelligent information system focused on factual correctness, reliability, grounded responses, and explainability. Accuracy is more important than fluency or creativity.

CORE SYSTEM REQUIREMENTS:
The system uses Retrieval-Augmented Generation (RAG).
All answers must be grounded in retrieved evidence.
The system must minimize hallucinations and unsupported claims.
Maintain clear separation between:
- document ingestion and preprocessing
- chunking
- embedding
- retrieval
- reranking
- generation
- evaluation

MULTILINGUAL REQUIREMENTS (CRITICAL):
The knowledge base is primarily in Bangla, with some English content.
The system must support:
- Bangla queries
- English queries
- code-mixed Bangla-English queries

Responses should match the user's query language whenever possible.
Cross-lingual retrieval is required (Bangla query to English content, English query to Bangla content).
Do not assume English-only models, embeddings, chunking methods, or benchmarks are suitable.
Embedding and retrieval methods must be evaluated with Bangla as a priority.

DATA CHARACTERISTICS:
The data is real-world, noisy, and heterogeneous, including:
- PDFs
- government websites and service portals
- notices
- mixed-language and semi-structured documents
- potentially OCR-affected or formatting-heavy text

The system must include strong ingestion, cleaning, normalization, chunking, metadata handling, and indexing.
Suggestions should account for real document noise, not only clean tutorial-style text.

SYSTEM PRIORITIES:
Accuracy > Retrieval Quality > Reasoning > Fluency

Additional priorities:
- strong preprocessing to avoid garbage in, garbage out
- retrieval quality is more important than stylish generation
- the system must be debuggable and explainable
- recommendations should support evaluation and ablation
- final choices should be academically defensible

EVALUATION EXPECTATION:
When relevant, frame suggestions in terms of measurable retrieval and RAG evaluation.
Prefer discussion using metrics such as:
- Recall@k
- MRR
- nDCG
- precision and qualitative relevance analysis
- hallucination or faithfulness analysis

When suggesting a method, distinguish clearly whether it is:
- a tutorial baseline
- a reasonable experimental baseline
- a strong candidate for the final thesis system

WHEN ANSWERING QUESTIONS:
Always consider multilingual and cross-lingual constraints, especially Bangla.
Do not default to English-first assumptions.
Be explicit about trade-offs, limitations, and failure modes.
Avoid vague or magical explanations; explain how and why something works.
If something is weak for this use case, say so clearly.
Prefer suggestions that are implementable, testable, and thesis-defensible.
If useful, propose comparisons, baselines, or ablation-friendly alternatives.

ASSUME:
The system is being built step by step.
Correctness matters more than shortcuts.
Questions may be low-level implementation questions or high-level architecture questions.
Act as a technical advisor helping design, implement, evaluate, and justify this system properly.
```
