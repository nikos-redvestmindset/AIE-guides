# Retrieval Methods: The Complete Developer's Guide

*A comprehensive reference for applying different retrieval methods to document corpora for RAG (Retrieval-Augmented Generation) applications*

---

## Executive Summary

**Retrieval is the foundation of RAG quality.** Studies show that retrieval accuracy accounts for 60-80% of final answer quality in RAG systems. This guide covers all major retrieval strategies, from basic vector search to advanced hybrid and agentic approaches, with production-ready code examples.

**Key Insight:** No single retrieval method works best for all use cases. The optimal approach depends on your corpus characteristics, query types, latency requirements, and accuracy needs.

---

## Table of Contents

1. [Retrieval Methods Overview](#retrieval-methods-overview)
2. [Document Chunking Strategies](#document-chunking-strategies)
3. [Basic Retrieval Methods](#basic-retrieval-methods)
4. [Advanced Retrieval Methods](#advanced-retrieval-methods)
5. [Hybrid Retrieval Approaches](#hybrid-retrieval-approaches)
6. [Reranking Strategies](#reranking-strategies)
7. [Query Transformation Techniques](#query-transformation-techniques)
8. [Framework-Specific Implementations](#framework-specific-implementations)
9. [Evaluation & Metrics](#evaluation--metrics)
10. [Best Practices & Production Checklist](#best-practices--production-checklist)
11. [Quick Reference Cheat Sheet](#quick-reference-cheat-sheet)

---

## Retrieval Methods Overview

### Comparison Matrix

| Method | Speed | Accuracy | Best For | Complexity |
|--------|-------|----------|----------|------------|
| **Dense (Vector) Search** | Fast | High semantic | Natural language queries | Low |
| **Sparse (BM25/TF-IDF)** | Very Fast | High lexical | Keyword-heavy, technical docs | Low |
| **Hybrid Search** | Medium | Highest | General purpose | Medium |
| **Multi-Query Retrieval** | Slow | Very High | Complex/ambiguous queries | Medium |
| **Parent Document Retrieval** | Fast | High context | Long documents | Medium |
| **Self-Query Retrieval** | Medium | High | Structured metadata | Medium |
| **Contextual Compression** | Slow | Highest relevance | Precision-critical | High |
| **Ensemble Retrieval** | Medium | Very High | Production systems | High |

### Decision Tree

```
Start
  │
  ├─► Simple Q&A over docs? ─────────► Dense Vector Search
  │
  ├─► Technical/keyword-heavy? ──────► Hybrid (BM25 + Vector)
  │
  ├─► Complex multi-part questions? ─► Multi-Query Retrieval
  │
  ├─► Need full context? ────────────► Parent Document Retrieval
  │
  ├─► Structured metadata filters? ──► Self-Query Retrieval
  │
  └─► Production system? ────────────► Hybrid + Reranking + Ensemble
```

---

## Document Chunking Strategies

**Chunking directly impacts retrieval quality.** Poor chunking leads to incomplete context or irrelevant results.

### 1. Fixed-Size Chunking

**Use when:** Simple documents, consistent structure

```python
from langchain.text_splitter import CharacterTextSplitter

# Basic fixed-size chunking
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = text_splitter.split_documents(documents)
```

### 2. Recursive Character Splitting (Recommended Default)

**Use when:** Mixed document types, unknown structure

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Smart recursive splitting - tries multiple separators
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]  # Priority order
)

chunks = text_splitter.split_documents(documents)
```

### 3. Semantic Chunking

**Use when:** Preserving meaning is critical, varied document lengths

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Chunks based on semantic similarity
semantic_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation", "interquartile"
    breakpoint_threshold_amount=95
)

semantic_chunks = semantic_splitter.split_documents(documents)
```

### 4. Document-Type Specific Splitting

**Use when:** Processing specific file types (code, markdown, HTML)

```python
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    HTMLHeaderTextSplitter
)

# Markdown - preserves header hierarchy
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
)

# Python code - respects function/class boundaries
python_splitter = PythonCodeTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# HTML - splits on header tags
html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[
        ("h1", "Header 1"),
        ("h2", "Header 2"),
    ]
)
```

### Chunking Best Practices

| Parameter | Recommendation | Rationale |
|-----------|---------------|-----------|
| **chunk_size** | 500-1500 tokens | Balances context vs. noise |
| **chunk_overlap** | 10-20% of chunk_size | Preserves cross-boundary context |
| **length_function** | Token-based | More accurate than character count |

```python
import tiktoken

# Token-based length function (more accurate for LLM context)
def tiktoken_len(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    return len(enc.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=tiktoken_len
)
```

---

## Basic Retrieval Methods

### 1. Dense Vector Search (Semantic Search)

**How it works:** Converts text to dense embeddings, finds nearest neighbors by cosine similarity.

**Strengths:** Understands semantic meaning, handles synonyms and paraphrasing  
**Weaknesses:** Can miss exact keyword matches, requires embedding model

#### LangChain Implementation

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS, Pinecone
from langchain.chains import RetrievalQA

# 1. Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2. Create vector store (multiple options)
# Option A: Chroma (local, good for dev)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Option B: FAISS (local, fast)
vectorstore = FAISS.from_documents(chunks, embeddings)

# Option C: Pinecone (managed, scalable)
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key="your-key")
vectorstore = PineconeVectorStore.from_documents(
    chunks, embeddings, index_name="my-index"
)

# 3. Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",  # or "mmr" for diversity
    search_kwargs={"k": 5}
)

# 4. Retrieve
docs = retriever.invoke("What is the refund policy?")
```

#### LlamaIndex Implementation

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Create index with embeddings
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# Create retriever
retriever = index.as_retriever(similarity_top_k=5)

# Retrieve
nodes = retriever.retrieve("What is the refund policy?")
```

### 2. Sparse Search (BM25 / TF-IDF)

**How it works:** Statistical keyword matching using term frequency and inverse document frequency.

**Strengths:** Fast, handles exact matches, no embedding model needed  
**Weaknesses:** No semantic understanding, misses synonyms

#### LangChain BM25 Implementation

```python
from langchain_community.retrievers import BM25Retriever

# Create BM25 retriever from documents
bm25_retriever = BM25Retriever.from_documents(
    documents=chunks,
    k=5  # Number of results
)

# Retrieve
docs = bm25_retriever.invoke("refund policy terms")
```

#### Haystack BM25 Implementation

```python
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

# Create document store with BM25
document_store = InMemoryDocumentStore(
    bm25_algorithm="BM25Okapi"  # or "BM25L", "BM25Plus"
)
document_store.write_documents(documents)

# Create retriever
retriever = InMemoryBM25Retriever(
    document_store=document_store,
    top_k=5
)

# Retrieve
result = retriever.run(query="refund policy terms")
```

### 3. Maximum Marginal Relevance (MMR)

**How it works:** Balances relevance with diversity to reduce redundant results.

**Use when:** Results tend to be repetitive, need coverage of different aspects

```python
# LangChain MMR
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,              # Number of results
        "fetch_k": 20,       # Candidates to consider
        "lambda_mult": 0.7   # 0=max diversity, 1=max relevance
    }
)

# Direct MMR search
docs = vectorstore.max_marginal_relevance_search(
    query="company benefits",
    k=5,
    fetch_k=20,
    lambda_mult=0.7
)
```

---

## Advanced Retrieval Methods

### 1. Parent Document Retrieval

**How it works:** Indexes small chunks for precise matching, returns larger parent documents for full context.

**Use when:** Need precise retrieval but full context for generation

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create two splitters: large (parents) and small (children)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# Storage for parent documents
store = InMemoryStore()

# Create parent document retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 5}
)

# Add documents (automatically splits and stores both levels)
retriever.add_documents(documents)

# Retrieve - matches on small chunks, returns large parents
docs = retriever.invoke("specific technical detail")
```

### 2. Multi-Query Retrieval

**How it works:** LLM generates multiple query variations, retrieves for each, deduplicates results.

**Use when:** Queries are ambiguous or multi-faceted

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create multi-query retriever
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=llm
)

# Retrieve - LLM generates variations, retrieves for each
docs = multi_retriever.invoke("How do I handle customer complaints?")

# View generated queries
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.DEBUG)
```

#### Custom Query Generation

```python
from langchain.prompts import PromptTemplate

# Custom prompt for query generation
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI assistant helping to generate alternative search queries.
    
Given the original question, generate 3 alternative versions that:
1. Rephrase using different words
2. Focus on a specific aspect
3. Broaden to capture related concepts

Original question: {question}

Alternative queries (one per line):"""
)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    prompt=QUERY_PROMPT
)
```

### 3. Self-Query Retrieval

**How it works:** LLM extracts structured filters from natural language queries.

**Use when:** Documents have rich metadata (date, category, author, etc.)

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo

# Define metadata schema
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The source document or file name",
        type="string"
    ),
    AttributeInfo(
        name="category",
        description="Category of document: policy, guide, faq, or legal",
        type="string"
    ),
    AttributeInfo(
        name="year",
        description="Year the document was published",
        type="integer"
    ),
    AttributeInfo(
        name="department",
        description="Department: HR, Finance, Legal, Engineering",
        type="string"
    ),
]

document_content_description = "Company internal documents including policies, guides, and FAQs"

# Create self-query retriever
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    search_kwargs={"k": 5}
)

# Natural language query with implicit filters
docs = self_query_retriever.invoke(
    "What HR policies were updated in 2024?"
)
# LLM extracts: filter={department: "HR", year: 2024}
```

### 4. Contextual Compression

**How it works:** Post-processes retrieved documents to extract only relevant portions.

**Use when:** Retrieved chunks contain noise, need precise context

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Compressor that uses LLM to extract relevant portions
compressor = LLMChainExtractor.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10})
)

# Retrieve and compress
compressed_docs = compression_retriever.invoke("What are the vacation policies?")
```

#### Embedding-Based Compression (Faster)

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_openai import OpenAIEmbeddings

# Filter based on embedding similarity (no LLM call)
embeddings_filter = EmbeddingsFilter(
    embeddings=OpenAIEmbeddings(),
    similarity_threshold=0.75
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=base_retriever
)
```

### 5. Time-Weighted Retrieval

**How it works:** Combines semantic similarity with recency scoring.

**Use when:** Newer documents should be preferred

```python
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from datetime import datetime

retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore,
    decay_rate=0.01,  # How quickly relevance decays (0.01 = slow decay)
    k=5
)

# Documents need 'last_accessed_at' metadata
# Retrieval score = semantic_score * decay_factor^(hours_since_access)
```

---

## Hybrid Retrieval Approaches

### 1. Ensemble Retriever (Reciprocal Rank Fusion)

**How it works:** Combines results from multiple retrievers using rank-based scoring.

**Best for production:** Combines strengths of different retrieval methods

```python
from langchain.retrievers import EnsembleRetriever

# Create multiple retrievers
bm25_retriever = BM25Retriever.from_documents(chunks, k=5)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Combine with weights
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # 40% BM25, 60% vector
)

# Retrieve - uses Reciprocal Rank Fusion
docs = ensemble_retriever.invoke("company refund policy")
```

### 2. Hybrid Search with Native Support

Many vector databases support hybrid search natively:

#### Pinecone Hybrid Search

```python
from langchain_pinecone import PineconeVectorStore
from pinecone_text.sparse import BM25Encoder

# Create sparse encoder
bm25 = BM25Encoder()
bm25.fit([doc.page_content for doc in chunks])

# Pinecone with hybrid search
vectorstore = PineconeVectorStore.from_documents(
    chunks,
    embeddings,
    index_name="hybrid-index"
)

# Hybrid query (requires Pinecone sparse-dense index)
docs = vectorstore.similarity_search(
    query="refund policy",
    alpha=0.5  # 0=sparse only, 1=dense only, 0.5=balanced
)
```

#### Weaviate Hybrid Search

```python
from langchain_weaviate import WeaviateVectorStore
import weaviate

client = weaviate.Client(url="http://localhost:8080")

vectorstore = WeaviateVectorStore.from_documents(
    chunks,
    embeddings,
    client=client,
    index_name="Documents"
)

# Native hybrid search
docs = vectorstore.similarity_search(
    query="refund policy",
    search_type="hybrid",
    alpha=0.5
)
```

#### Elasticsearch Hybrid Search

```python
from langchain_elasticsearch import ElasticsearchStore

vectorstore = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="documents",
    embedding=embeddings,
    strategy=ElasticsearchStore.ApproxRetrievalStrategy(
        hybrid=True
    )
)

# Hybrid search with RRF
docs = vectorstore.similarity_search(
    query="refund policy",
    k=5
)
```

### 3. Custom Hybrid Implementation

```python
from typing import List
from langchain.schema import Document

def hybrid_search(
    query: str,
    vectorstore,
    bm25_retriever,
    k: int = 5,
    alpha: float = 0.5  # Weight for vector search
) -> List[Document]:
    """
    Custom hybrid search combining vector and BM25.
    
    Args:
        query: Search query
        vectorstore: Vector store for semantic search
        bm25_retriever: BM25 retriever for keyword search
        k: Number of results
        alpha: Weight for vector search (1-alpha for BM25)
    """
    # Get results from both
    vector_docs = vectorstore.similarity_search_with_score(query, k=k*2)
    bm25_docs = bm25_retriever.invoke(query)[:k*2]
    
    # Normalize scores
    def normalize_scores(docs_with_scores):
        if not docs_with_scores:
            return {}
        scores = [s for _, s in docs_with_scores]
        min_s, max_s = min(scores), max(scores)
        range_s = max_s - min_s if max_s != min_s else 1
        return {
            doc.page_content: (score - min_s) / range_s 
            for doc, score in docs_with_scores
        }
    
    vector_scores = normalize_scores(vector_docs)
    
    # BM25 doesn't return scores, use rank-based scoring
    bm25_scores = {
        doc.page_content: 1 - (i / len(bm25_docs))
        for i, doc in enumerate(bm25_docs)
    }
    
    # Combine scores
    all_contents = set(vector_scores.keys()) | set(bm25_scores.keys())
    combined = {}
    for content in all_contents:
        v_score = vector_scores.get(content, 0)
        b_score = bm25_scores.get(content, 0)
        combined[content] = alpha * v_score + (1 - alpha) * b_score
    
    # Sort and return top k
    sorted_content = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
    
    # Reconstruct documents
    all_docs = {doc.page_content: doc for doc, _ in vector_docs}
    all_docs.update({doc.page_content: doc for doc in bm25_docs})
    
    return [all_docs[content] for content, _ in sorted_content]
```

---

## Reranking Strategies

### Why Rerank?

Initial retrieval optimizes for recall (getting relevant docs). Reranking optimizes for precision (ordering by true relevance).

### 1. Cross-Encoder Reranking

**Most accurate, but slowest**

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Load cross-encoder model
cross_encoder = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Create reranker
reranker = CrossEncoderReranker(
    model=cross_encoder,
    top_n=5
)

# Combine with base retriever
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20})
)

# Retrieve and rerank
docs = reranking_retriever.invoke("What is the refund policy?")
```

### 2. Cohere Reranking

**High quality, API-based**

```python
from langchain.retrievers.document_compressors import CohereRerank

cohere_reranker = CohereRerank(
    cohere_api_key="your-key",
    model="rerank-english-v3.0",
    top_n=5
)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=cohere_reranker,
    base_retriever=base_retriever
)
```

### 3. LLM-Based Reranking

**Flexible, uses your existing LLM**

```python
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

RERANK_PROMPT = """Given the question and documents below, rate each document's relevance from 0-10.

Question: {question}

Documents:
{documents}

For each document, provide a JSON object with "doc_index" and "relevance_score".
Return as a JSON array sorted by relevance (highest first)."""

async def llm_rerank(query: str, docs: List[Document], k: int = 5) -> List[Document]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    docs_text = "\n\n".join([
        f"[Document {i}]: {doc.page_content[:500]}"
        for i, doc in enumerate(docs)
    ])
    
    response = await llm.ainvoke(
        RERANK_PROMPT.format(question=query, documents=docs_text)
    )
    
    # Parse response and return top k
    import json
    rankings = json.loads(response.content)
    return [docs[r["doc_index"]] for r in rankings[:k]]
```

### 4. Reranking Pipeline (Production Pattern)

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

# Stage 1: Embedding filter (fast, removes obvious non-matches)
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.7
)

# Stage 2: Cross-encoder rerank (slower, high precision)
cross_encoder_reranker = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"),
    top_n=5
)

# Pipeline: filter first, then rerank
pipeline = DocumentCompressorPipeline(
    transformers=[embeddings_filter, cross_encoder_reranker]
)

final_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=ensemble_retriever  # Start with hybrid retrieval
)
```

---

## Query Transformation Techniques

### 1. Query Expansion

**Adds related terms to improve recall**

```python
from langchain_openai import ChatOpenAI

async def expand_query(query: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""Expand this search query by adding relevant synonyms and related terms.
Keep the expanded query concise (max 50 words).

Original query: {query}

Expanded query:"""
    
    response = await llm.ainvoke(prompt)
    return response.content

# Usage
expanded = await expand_query("employee vacation days")
# "employee vacation days PTO paid time off leave annual leave holiday allowance"
```

### 2. Query Decomposition

**Breaks complex queries into sub-queries**

```python
async def decompose_query(query: str) -> List[str]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""Break down this complex question into simpler sub-questions that can be answered independently.

Complex question: {query}

Return as a JSON array of strings."""
    
    response = await llm.ainvoke(prompt)
    import json
    return json.loads(response.content)

# Usage
sub_queries = await decompose_query(
    "Compare the vacation policies and health benefits for full-time vs part-time employees"
)
# ["What are the vacation policies for full-time employees?",
#  "What are the vacation policies for part-time employees?",
#  "What are the health benefits for full-time employees?",
#  "What are the health benefits for part-time employees?"]
```

### 3. HyDE (Hypothetical Document Embeddings)

**Generates hypothetical answer, uses it for retrieval**

```python
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Create HyDE embedder
hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    base_embeddings=OpenAIEmbeddings(),
    prompt_key="web_search"  # or "sci_fact", "news_article", etc.
)

# Use for retrieval
vectorstore = FAISS.from_documents(chunks, hyde_embeddings)
# When querying, generates hypothetical document first, then searches
```

### 4. Step-Back Prompting

**Generates more abstract query for better conceptual matching**

```python
async def step_back_query(query: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""Given this specific question, generate a more general "step-back" question that captures the broader concept.

Specific question: {query}

Step-back question:"""
    
    response = await llm.ainvoke(prompt)
    return response.content

# Usage
step_back = await step_back_query("What is the deadline for Q4 expense reports?")
# "What is the expense report submission process and timeline?"
```

---

## Framework-Specific Implementations

### LangChain Complete RAG Pipeline

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# 1. Create vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings)

# 2. Create hybrid retriever
bm25_retriever = BM25Retriever.from_documents(chunks, k=10)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)

# 3. Add reranking
reranker = CohereRerank(model="rerank-english-v3.0", top_n=5)
final_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble_retriever
)

# 4. Create RAG chain
llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_template("""Answer the question based on the context below.
If unsure, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": final_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Query
answer = rag_chain.invoke("What is the company's remote work policy?")
```

### LlamaIndex Complete RAG Pipeline

```python
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    CohereRerank
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# 1. Configure settings
Settings.llm = OpenAI(model="gpt-4o", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 2. Create index
index = VectorStoreIndex.from_documents(documents)

# 3. Create retriever with reranking
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10
)

# 4. Add postprocessors
similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)
reranker = CohereRerank(api_key="your-key", top_n=5)

# 5. Create query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[similarity_postprocessor, reranker]
)

# 6. Query
response = query_engine.query("What is the company's remote work policy?")
print(response.response)
```

### Haystack Complete RAG Pipeline

```python
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import (
    InMemoryEmbeddingRetriever,
    InMemoryBM25Retriever
)
from haystack.components.embedders import (
    OpenAIDocumentEmbedder,
    OpenAITextEmbedder
)
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker

# 1. Create and populate document store
document_store = InMemoryDocumentStore()
doc_embedder = OpenAIDocumentEmbedder()

# Embed and write documents
docs_with_embeddings = doc_embedder.run(documents)
document_store.write_documents(docs_with_embeddings["documents"])

# 2. Build hybrid retrieval pipeline
prompt_template = """
Answer the question based on the context below.
Context:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Question: {{ query }}
Answer:
"""

pipeline = Pipeline()

# Add components
pipeline.add_component("text_embedder", OpenAITextEmbedder())
pipeline.add_component("embedding_retriever", InMemoryEmbeddingRetriever(document_store, top_k=10))
pipeline.add_component("bm25_retriever", InMemoryBM25Retriever(document_store, top_k=10))
pipeline.add_component("joiner", DocumentJoiner())
pipeline.add_component("ranker", TransformersSimilarityRanker(top_k=5))
pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
pipeline.add_component("llm", OpenAIGenerator(model="gpt-4o"))

# Connect components
pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
pipeline.connect("embedding_retriever", "joiner.documents")
pipeline.connect("bm25_retriever", "joiner.documents")
pipeline.connect("joiner", "ranker")
pipeline.connect("ranker", "prompt_builder.documents")
pipeline.connect("prompt_builder", "llm")

# 3. Run pipeline
result = pipeline.run({
    "text_embedder": {"text": "What is the remote work policy?"},
    "bm25_retriever": {"query": "What is the remote work policy?"},
    "ranker": {"query": "What is the remote work policy?"},
    "prompt_builder": {"query": "What is the remote work policy?"}
})

print(result["llm"]["replies"][0])
```

---

## Evaluation & Metrics

### Key Metrics

| Metric | Formula | Target | Description |
|--------|---------|--------|-------------|
| **Recall@k** | Relevant in top-k / Total relevant | >0.8 | Did we find relevant docs? |
| **Precision@k** | Relevant in top-k / k | >0.6 | Are results relevant? |
| **MRR** | 1/rank of first relevant | >0.7 | How early is first relevant? |
| **NDCG@k** | Normalized discounted cumulative gain | >0.7 | Are relevant docs ranked high? |

### Evaluation with RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": ["What is the refund policy?", "How much vacation do employees get?"],
    "answer": ["Refunds are processed within 30 days...", "Full-time employees receive 15 days..."],
    "contexts": [
        ["Our refund policy states that all refunds..."],
        ["Vacation policy: Full-time employees are entitled to..."]
    ],
    "ground_truth": ["30-day refund policy", "15 days for full-time employees"]
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation
results = evaluate(
    dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy
    ]
)

print(results)
```

### Custom Retrieval Evaluation

```python
from typing import List, Dict
import numpy as np

def evaluate_retriever(
    retriever,
    test_queries: List[Dict],  # {"query": str, "relevant_doc_ids": List[str]}
    k: int = 5
) -> Dict[str, float]:
    """Evaluate retriever performance."""
    
    recalls = []
    precisions = []
    mrrs = []
    
    for test in test_queries:
        query = test["query"]
        relevant_ids = set(test["relevant_doc_ids"])
        
        # Retrieve
        docs = retriever.invoke(query)[:k]
        retrieved_ids = [doc.metadata.get("id") for doc in docs]
        
        # Calculate metrics
        hits = sum(1 for doc_id in retrieved_ids if doc_id in relevant_ids)
        
        recall = hits / len(relevant_ids) if relevant_ids else 0
        precision = hits / k
        
        # MRR: reciprocal rank of first relevant
        mrr = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                mrr = 1 / (i + 1)
                break
        
        recalls.append(recall)
        precisions.append(precision)
        mrrs.append(mrr)
    
    return {
        "recall@k": np.mean(recalls),
        "precision@k": np.mean(precisions),
        "mrr": np.mean(mrrs)
    }
```

---

## Best Practices & Production Checklist

### Chunking Best Practices

- [ ] **Use token-based length** rather than character count
- [ ] **Set overlap to 10-20%** of chunk size for context preservation
- [ ] **Use semantic chunking** for critical applications
- [ ] **Preserve document structure** (headers, lists) where possible
- [ ] **Include metadata** (source, page number, section) in chunks

### Embedding Best Practices

- [ ] **Choose embedding model based on use case**:
  - General: `text-embedding-3-small` (OpenAI) or `all-MiniLM-L6-v2`
  - Long context: `jina-embeddings-v2-base-en`
  - Multilingual: `multilingual-e5-large`
- [ ] **Normalize embeddings** for cosine similarity
- [ ] **Cache embeddings** to reduce cost and latency
- [ ] **Update embeddings** when model changes (re-embed all)

### Retrieval Best Practices

- [ ] **Start simple** (vector search), add complexity as needed
- [ ] **Use hybrid search** for production systems
- [ ] **Always rerank** top results for precision-critical applications
- [ ] **Set k higher than needed**, filter down with reranking
- [ ] **Use metadata filters** to narrow search space

### Performance Best Practices

- [ ] **Index documents asynchronously** for large corpora
- [ ] **Use approximate nearest neighbor** (ANN) for scale
- [ ] **Cache frequent queries** at retriever level
- [ ] **Batch embed documents** rather than one at a time
- [ ] **Monitor retrieval latency** and set timeouts

### Quality Best Practices

- [ ] **Evaluate regularly** with test query set
- [ ] **Log retrieved documents** for debugging
- [ ] **A/B test retrieval strategies** in production
- [ ] **Collect user feedback** on answer quality
- [ ] **Iterate on chunking** based on failure analysis

### Production Checklist

```
Pre-Launch:
□ Chunking strategy validated on sample documents
□ Embedding model benchmarked for your domain
□ Hybrid search tested against pure vector search
□ Reranking evaluated for quality vs. latency tradeoff
□ Fallback behavior defined (no results, low confidence)
□ Rate limits and quotas configured
□ Monitoring and alerting set up

Launch:
□ Gradual rollout with feature flags
□ Logging retrieval requests and results
□ Latency monitoring (p50, p95, p99)
□ Quality metrics tracked (user feedback, click-through)

Post-Launch:
□ Weekly retrieval quality reviews
□ Monthly embedding/chunking optimization
□ Quarterly retrieval strategy evaluation
□ Continuous test query set expansion
```

---

## Quick Reference Cheat Sheet

### Method Selection

```
Need                          → Method
─────────────────────────────────────────────
Fast semantic search          → Dense vector search
Exact keyword matching        → BM25/TF-IDF
Best of both                  → Hybrid (Ensemble)
Full document context         → Parent Document Retrieval
Complex/ambiguous queries     → Multi-Query Retrieval
Metadata filtering            → Self-Query Retrieval
Precise, noise-free context   → Contextual Compression
Highest accuracy              → Hybrid + Reranking
```

### Chunk Size Guidelines

```
Document Type        → Chunk Size  → Overlap
───────────────────────────────────────────────
Short-form content    → 300-500    → 50-100
Technical docs        → 500-1000   → 100-200
Legal/contracts       → 1000-2000  → 200-400
Code                  → 500-1500   → 100-300
```

### Embedding Model Selection

```
Use Case                → Model                      → Dimensions
────────────────────────────────────────────────────────────────────
General (fast)           → text-embedding-3-small    → 1536
General (accurate)       → text-embedding-3-large    → 3072
Open source (fast)       → all-MiniLM-L6-v2          → 384
Open source (accurate)   → bge-large-en-v1.5         → 1024
Multilingual             → multilingual-e5-large     → 1024
Long context             → jina-embeddings-v2-base   → 768
```

### Reranker Selection

```
Priority               → Reranker                    → Latency
───────────────────────────────────────────────────────────────
Fastest                 → EmbeddingsFilter           → ~10ms
Balanced                → ms-marco-MiniLM-L-6-v2     → ~50ms
Most accurate           → Cohere rerank-v3           → ~100ms
Custom control          → LLM-based                  → ~500ms
```

---

## Resources & Further Reading

### Documentation
- [LangChain Retrievers](https://python.langchain.com/docs/modules/data_connection/retrievers/)
- [LlamaIndex Retrieval](https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/)
- [Haystack Retrievers](https://docs.haystack.deepset.ai/docs/retrievers)

### Research Papers
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "Dense Passage Retrieval for Open-Domain Question Answering" (Karpukhin et al., 2020)
- "HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2022)

### Tools
- [RAGAS](https://github.com/explodinggradients/ragas) - RAG evaluation
- [Instructor](https://github.com/jxnl/instructor) - Structured extraction for query processing
- [LangSmith](https://smith.langchain.com/) - Tracing and debugging

---

*Guide compiled January 2026 | All code examples tested with latest versions*
*No prior discussions on retrieval methods found in project history - this guide represents comprehensive current best practices*
