# Text Chunking Methods: Comprehensive Guide for RAG Applications

*A practical reference for developers building retrieval-augmented generation (RAG) systems*

---

## The Core Principle

> **"What gets chunked gets retrieved"** - The quality of your RAG system directly depends on how you split your documents.

Chunking is the process of breaking down large documents into smaller, meaningful pieces that can be embedded and retrieved effectively. Poor chunking leads to poor retrieval, which leads to poor generation quality—regardless of how good your LLM is.

---

## Quick Reference: Chunking Methods Overview

| Method | Best For | Chunk Size Control | Semantic Coherence | Complexity |
|--------|----------|-------------------|-------------------|------------|
| **Fixed Character** | Quick prototypes | ⭐⭐⭐⭐⭐ | ⭐ | Very Easy |
| **Recursive Character** | General purpose | ⭐⭐⭐⭐ | ⭐⭐⭐ | Easy |
| **Token-Based** | LLM context fitting | ⭐⭐⭐⭐⭐ | ⭐⭐ | Easy |
| **Sentence Splitting** | Conversational Q&A | ⭐⭐⭐ | ⭐⭐⭐⭐ | Easy |
| **Semantic Chunking** | High-quality retrieval | ⭐⭐ | ⭐⭐⭐⭐⭐ | Moderate |
| **Document Structure** | Markdown/HTML | ⭐⭐⭐ | ⭐⭐⭐⭐ | Moderate |
| **Code-Aware** | Code repositories | ⭐⭐⭐ | ⭐⭐⭐⭐ | Moderate |
| **Hierarchical** | Complex documents | ⭐⭐ | ⭐⭐⭐⭐⭐ | Complex |
| **Parent Document** | Context-rich retrieval | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Moderate |
| **Agentic Chunking** | Dynamic documents | ⭐⭐ | ⭐⭐⭐⭐⭐ | Complex |

---

## 1. Fixed Character Splitting (Naive)

### When to Use
- Quick prototyping
- Uniform text without structure
- When you just need something working fast

### Implementation

```python
def fixed_character_split(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Simple fixed-size character splitting with overlap.
    
    Args:
        text: The input text to split
        chunk_size: Number of characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Overlap for context continuity
    return chunks

# Usage
text = "Your long document text here..."
chunks = fixed_character_split(text, chunk_size=500, overlap=50)

print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i}: {len(chunk)} chars - '{chunk[:50]}...'")
```

### ⚠️ Limitations
- Splits mid-word and mid-sentence
- No semantic awareness
- Poor retrieval quality for complex queries
- Only suitable for very uniform, simple text

---

## 2. Recursive Character Text Splitting (LangChain Default)

### When to Use
- **General purpose** - best starting point for most applications
- Documents with natural paragraph/sentence structure
- When you want a good balance of simplicity and quality

### How It Works
The splitter tries separators in order, falling back to smaller units only when necessary:
1. `\n\n` (paragraphs) → first choice
2. `\n` (newlines) → if chunks still too large
3. `. ` (sentences) → further breakdown
4. ` ` (words) → near-atomic level
5. `` (characters) → last resort

### Implementation

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Basic usage
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          # Target chunk size in characters
    chunk_overlap=200,        # Overlap between chunks
    length_function=len,      # How to measure length
    separators=["\n\n", "\n", ". ", " ", ""]  # Order matters!
)

# Split raw text
chunks = text_splitter.split_text(text)

# Split with metadata preservation
documents = [Document(page_content=text, metadata={"source": "doc.pdf", "page": 1})]
split_docs = text_splitter.split_documents(documents)

# Each split_doc maintains the original metadata
for doc in split_docs[:3]:
    print(f"Source: {doc.metadata['source']}, Length: {len(doc.page_content)}")
```

### Custom Separators Example

```python
# For legal documents with specific structure
legal_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=[
        "\n\nARTICLE",      # Major sections
        "\n\nSection",      # Subsections
        "\n\n",             # Paragraphs
        "\n",               # Lines
        ". ",               # Sentences
        " ",                # Words
        ""                  # Characters
    ]
)

# For code mixed with documentation
code_doc_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=[
        "\n\n\n",           # Major sections
        "\n\n",             # Paragraphs/functions
        "\n",               # Lines
        "```",              # Code blocks
        ". ",               # Sentences
        " ",
        ""
    ]
)
```

---

## 3. Token-Based Splitting

### When to Use
- **Precise control over LLM context windows**
- When you need to guarantee chunks fit within token limits
- Multi-model pipelines with different tokenizers

### Implementation with LangChain

```python
from langchain.text_splitter import TokenTextSplitter

# Using tiktoken (OpenAI's tokenizer)
splitter = TokenTextSplitter(
    chunk_size=512,                    # tokens, not characters
    chunk_overlap=50,
    encoding_name="cl100k_base"        # GPT-4/3.5-turbo encoding
)
chunks = splitter.split_text(text)

print(f"Created {len(chunks)} chunks")
```

### Manual Implementation with tiktoken

```python
import tiktoken

def token_split(
    text: str, 
    max_tokens: int = 512, 
    overlap_tokens: int = 50, 
    model: str = "gpt-4"
) -> list[str]:
    """
    Split text by token count for precise LLM context control.
    
    Args:
        text: Input text to split
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between chunks
        model: Model name for tokenizer selection
    
    Returns:
        List of text chunks
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end - overlap_tokens
    
    return chunks

# Usage
chunks = token_split(text, max_tokens=512, overlap_tokens=50)

# Verify token counts
encoding = tiktoken.encoding_for_model("gpt-4")
for i, chunk in enumerate(chunks[:3]):
    token_count = len(encoding.encode(chunk))
    print(f"Chunk {i}: {token_count} tokens")
```

### Token Count Reference

| Model Family | Encoding | ~Chars per Token |
|--------------|----------|------------------|
| GPT-4, GPT-3.5-turbo | cl100k_base | ~4 |
| GPT-3 (davinci) | p50k_base | ~4 |
| Codex | p50k_base | ~4 |
| Claude | claude tokenizer | ~3.5 |

---

## 4. Sentence-Based Splitting

### When to Use
- Q&A systems
- Conversational applications
- When semantic completeness of each chunk matters

### Implementation with NLTK

```python
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

def sentence_split(
    text: str, 
    sentences_per_chunk: int = 5, 
    overlap_sentences: int = 1
) -> list[str]:
    """
    Split text by sentences with configurable overlap.
    
    Args:
        text: Input text
        sentences_per_chunk: Number of sentences per chunk
        overlap_sentences: Sentence overlap between chunks
    
    Returns:
        List of text chunks
    """
    sentences = sent_tokenize(text)
    chunks = []
    
    step = sentences_per_chunk - overlap_sentences
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        if chunk:  # Avoid empty chunks
            chunks.append(chunk)
    
    return chunks

# Usage
chunks = sentence_split(text, sentences_per_chunk=5, overlap_sentences=1)
```

### Implementation with spaCy (Better for Complex Text)

```python
import spacy

# Load model (run: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

def spacy_sentence_split(
    text: str, 
    sentences_per_chunk: int = 5,
    overlap_sentences: int = 1,
    max_length: int = 1000000
) -> list[str]:
    """
    Split using spaCy's superior sentence boundary detection.
    Handles abbreviations, URLs, and edge cases better than NLTK.
    """
    # Handle long documents
    nlp.max_length = max_length
    doc = nlp(text)
    
    sentences = [sent.text.strip() for sent in doc.sents]
    chunks = []
    
    step = sentences_per_chunk - overlap_sentences
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        if chunk:
            chunks.append(chunk)
    
    return chunks

# Usage
chunks = spacy_sentence_split(text, sentences_per_chunk=5)
```

---

## 5. Semantic Chunking (Embedding-Based)

### When to Use
- **Quality is critical** and compute cost is acceptable
- Documents with varying topic density
- When you need chunks that represent coherent ideas

### How It Works
1. Split text into sentences
2. Generate embeddings for each sentence
3. Calculate similarity between consecutive sentences
4. Split where similarity drops below threshold

### LangChain Implementation

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Percentile-based breakpoints (recommended)
semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95  # Higher = fewer, larger chunks
)

# Alternative: Standard deviation based
semantic_splitter_std = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.5  # Number of std devs
)

# Alternative: Interquartile range
semantic_splitter_iqr = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="interquartile",
    breakpoint_threshold_amount=1.5
)

# Split text
chunks = semantic_splitter.split_text(text)
```

### Manual Implementation (No Dependencies on LangChain Experimental)

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt', quiet=True)

class SemanticChunker:
    """
    Split text based on semantic similarity between consecutive sentences.
    """
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        similarity_threshold: float = 0.75,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000
    ):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def split(self, text: str) -> list[str]:
        """Split text into semantically coherent chunks."""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 1:
            return [text]
        
        # Get embeddings for all sentences
        embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_length = len(sentences[0])
        
        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(embeddings[i-1], embeddings[i])
            sentence_length = len(sentences[i])
            
            # Decide whether to add to current chunk or start new one
            should_split = (
                similarity < self.similarity_threshold and 
                current_length >= self.min_chunk_size
            ) or (current_length + sentence_length > self.max_chunk_size)
            
            if should_split:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
                current_length = sentence_length
            else:
                current_chunk.append(sentences[i])
                current_length += sentence_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

# Usage
chunker = SemanticChunker(
    similarity_threshold=0.75,
    min_chunk_size=100,
    max_chunk_size=1500
)
chunks = chunker.split(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {len(chunk)} chars")
    print(f"  Preview: {chunk[:100]}...")
    print()
```

---

## 6. Document Structure-Based Splitting

### When to Use
- **Markdown documents** (README files, documentation)
- **HTML content** (web pages, articles)
- Documents with clear hierarchical structure

### Markdown Splitting

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Define headers to split on
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]

# Create markdown splitter
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False  # Keep headers in content
)

# Split by headers first
md_header_splits = markdown_splitter.split_text(markdown_document)

# Each split has metadata about its header hierarchy
for doc in md_header_splits[:3]:
    print(f"Headers: {doc.metadata}")
    print(f"Content: {doc.page_content[:100]}...")
    print()

# Further split large sections if needed
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100
)
final_splits = text_splitter.split_documents(md_header_splits)
```

### HTML Splitting

```python
from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter

# Define HTML headers to split on
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

# Create HTML splitter
html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

# Split HTML document
html_splits = html_splitter.split_text(html_document)

# Or split from URL
html_splits_from_url = html_splitter.split_text_from_url(
    "https://example.com/article"
)

# Further split if needed
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100
)
final_splits = text_splitter.split_documents(html_splits)
```

---

## 7. Code-Aware Splitting

### When to Use
- Code repositories
- Technical documentation with code examples
- Multi-language codebases

### Implementation

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# Python code splitting
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=200
)
python_chunks = python_splitter.split_text(python_code)

# JavaScript/TypeScript splitting
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS,
    chunk_size=2000,
    chunk_overlap=200
)

ts_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.TS,
    chunk_size=2000,
    chunk_overlap=200
)

# Other supported languages
supported_languages = [
    Language.PYTHON,
    Language.JS,
    Language.TS,
    Language.GO,
    Language.RUST,
    Language.JAVA,
    Language.C,
    Language.CPP,
    Language.PHP,
    Language.RUBY,
    Language.SCALA,
    Language.SWIFT,
    Language.KOTLIN,
    Language.MARKDOWN,
    Language.LATEX,
    Language.HTML,
    Language.SOL,  # Solidity
]

# Auto-detect and split
def split_code_by_language(code: str, language: str, chunk_size: int = 2000) -> list[str]:
    """Split code based on detected or specified language."""
    lang_map = {
        'python': Language.PYTHON,
        'py': Language.PYTHON,
        'javascript': Language.JS,
        'js': Language.JS,
        'typescript': Language.TS,
        'ts': Language.TS,
        'java': Language.JAVA,
        'go': Language.GO,
        'rust': Language.RUST,
        'c': Language.C,
        'cpp': Language.CPP,
        'c++': Language.CPP,
    }
    
    lang = lang_map.get(language.lower(), Language.PYTHON)
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=lang,
        chunk_size=chunk_size,
        chunk_overlap=200
    )
    return splitter.split_text(code)
```

---

## 8. LlamaIndex Chunking Methods

### Sentence Splitter (Default)

```python
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

# Create documents
documents = [Document(text=text, metadata={"source": "doc.pdf"})]

# Sentence-based splitting
splitter = SentenceSplitter(
    chunk_size=1024,              # Target size in characters
    chunk_overlap=20,             # Overlap between chunks
    paragraph_separator="\n\n",   # How to identify paragraphs
    secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",  # Sentence pattern
)

nodes = splitter.get_nodes_from_documents(documents)

for node in nodes[:3]:
    print(f"Node ID: {node.node_id}")
    print(f"Text: {node.text[:100]}...")
    print(f"Metadata: {node.metadata}")
    print()
```

### Semantic Splitting

```python
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

# Initialize embedding model
embed_model = OpenAIEmbedding()

# Create semantic splitter
semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1,                          # Sentences to group for embedding
    breakpoint_percentile_threshold=95,     # Higher = fewer splits
    embed_model=embed_model,
)

semantic_nodes = semantic_splitter.get_nodes_from_documents(documents)

print(f"Created {len(semantic_nodes)} semantic chunks")
```

### Hierarchical Splitting

```python
from llama_index.core.node_parser import HierarchicalNodeParser

# Create hierarchical splitter with multiple levels
hierarchical_splitter = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]  # Large → Medium → Small
)

hierarchical_nodes = hierarchical_splitter.get_nodes_from_documents(documents)

# Nodes have parent-child relationships
for node in hierarchical_nodes[:5]:
    print(f"Node: {node.node_id[:20]}...")
    print(f"Parent: {node.parent_node}")
    print(f"Children: {len(node.child_nodes) if node.child_nodes else 0}")
    print()
```

### Token-Based Splitting

```python
from llama_index.core.node_parser import TokenTextSplitter

token_splitter = TokenTextSplitter(
    chunk_size=512,           # Tokens per chunk
    chunk_overlap=50,         # Token overlap
    separator=" ",            # Word separator
)

token_nodes = token_splitter.get_nodes_from_documents(documents)
```

---

## 9. Haystack Document Splitting

### Basic Splitting

```python
from haystack.components.preprocessors import DocumentSplitter
from haystack import Document

# Create documents
documents = [Document(content=text, meta={"source": "doc.pdf"})]

# Split by sentence
sentence_splitter = DocumentSplitter(
    split_by="sentence",      # "word", "sentence", "passage", "page"
    split_length=5,           # Number of units per chunk
    split_overlap=1           # Overlap in units
)

result = sentence_splitter.run(documents=documents)
chunks = result["documents"]

# Split by word
word_splitter = DocumentSplitter(
    split_by="word",
    split_length=200,
    split_overlap=20
)

# Split by passage (paragraphs)
passage_splitter = DocumentSplitter(
    split_by="passage",
    split_length=2,
    split_overlap=0
)
```

### Custom Function-Based Splitting

```python
from haystack.components.preprocessors import DocumentSplitter

def custom_split_function(text: str) -> list[str]:
    """
    Custom splitting logic - split on double newlines
    and section markers.
    """
    import re
    
    # Split on section markers or double newlines
    pattern = r'\n\n|(?=\n#{1,3}\s)'
    chunks = re.split(pattern, text)
    
    # Filter empty chunks and strip whitespace
    return [chunk.strip() for chunk in chunks if chunk.strip()]

# Use custom function
custom_splitter = DocumentSplitter(
    split_by="function",
    splitting_function=custom_split_function
)

result = custom_splitter.run(documents=documents)
```

---

## 10. Parent Document Retrieval Pattern

### When to Use
- You want to **retrieve on small chunks** (precise matching)
- But **pass larger context** to the LLM (better generation)
- Best of both worlds approach

### Implementation

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create two splitters with different chunk sizes
# Small chunks for precise retrieval
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

# Large chunks for context-rich generation
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

# Initialize vector store and document store
vectorstore = Chroma(
    collection_name="parent_child_chunks",
    embedding_function=embeddings
)
docstore = InMemoryStore()

# Create the retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Add documents
retriever.add_documents(documents)

# Retrieve: searches small chunks, returns parent documents
query = "What is the main argument?"
relevant_docs = retriever.get_relevant_documents(query)

# Each returned doc is the larger parent chunk
for doc in relevant_docs:
    print(f"Retrieved parent chunk: {len(doc.page_content)} chars")
    print(f"Content: {doc.page_content[:200]}...")
```

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    Original Document                     │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Parent Chunks (2000 chars)                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Parent 1   │  │  Parent 2   │  │  Parent 3   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │               │
│         ▼               ▼               ▼               │
│  ┌───┬───┬───┐   ┌───┬───┬───┐   ┌───┬───┬───┐        │
│  │C1 │C2 │C3 │   │C4 │C5 │C6 │   │C7 │C8 │C9 │        │
│  └───┴───┴───┘   └───┴───┴───┘   └───┴───┴───┘        │
│  Child Chunks (400 chars) - Embedded in Vector Store   │
└─────────────────────────────────────────────────────────┘

Query: "What is X?" 
  │
  ▼
Search finds Child C5 (high similarity)
  │
  ▼
Return Parent 2 (provides full context)
```

---

## Choosing the Right Method: Decision Tree

```
START
  │
  ├─ Quick prototype needed?
  │   └─ YES ──────────────────────────► Recursive Character (LangChain default)
  │
  ├─ Need precise token control?
  │   └─ YES ──────────────────────────► Token-Based Splitting
  │
  ├─ Document has clear structure?
  │   ├─ Markdown ─────────────────────► MarkdownHeaderTextSplitter
  │   ├─ HTML ─────────────────────────► HTMLHeaderTextSplitter
  │   └─ Code ─────────────────────────► Language-specific Splitter
  │
  ├─ Quality is critical?
  │   ├─ Can afford compute ───────────► Semantic Chunking
  │   └─ Budget constrained ───────────► Sentence-Based + Recursive
  │
  ├─ Need precise retrieval + rich context?
  │   └─ YES ──────────────────────────► Parent Document Retriever
  │
  └─ Complex hierarchical documents?
      └─ YES ──────────────────────────► Hierarchical Chunking
```

---

## Key Parameters Cheat Sheet

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| `chunk_size` | 500-2000 chars / 256-512 tokens | Smaller = more precise retrieval, larger = more context |
| `chunk_overlap` | 10-20% of chunk_size | Prevents context loss at boundaries |
| `similarity_threshold` | 0.7-0.85 | For semantic chunking; higher = fewer splits |
| `breakpoint_percentile` | 85-95 | Higher = fewer, larger semantic chunks |
| `sentences_per_chunk` | 3-10 | For sentence-based splitting |

---

## Performance Comparison

| Method | Speed | Memory | Quality | Cost |
|--------|-------|--------|---------|------|
| Fixed Character | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | Free |
| Recursive Character | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Free |
| Token-Based | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Free |
| Sentence-Based | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Free |
| Semantic (local model) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Free |
| Semantic (API) | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $$$ |
| Structure-Based | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Free |

---

## Common Pitfalls and Solutions

### 1. Chunks Too Small
**Problem:** Lost context, fragmented information
**Solution:** Increase `chunk_size`, use overlap, or try Parent Document Retriever

### 2. Chunks Too Large
**Problem:** Diluted relevance, poor retrieval precision
**Solution:** Decrease `chunk_size`, use semantic chunking

### 3. Splitting Mid-Thought
**Problem:** Semantic boundaries not respected
**Solution:** Use semantic chunking or sentence-based splitting

### 4. Lost Metadata
**Problem:** Can't trace chunks back to source
**Solution:** Always use `split_documents()` to preserve metadata

### 5. Inconsistent Chunk Sizes
**Problem:** Some chunks much smaller than target
**Solution:** Set `min_chunk_size` or post-process to merge small chunks

---

## Installation Requirements

```bash
# Core LangChain
pip install langchain langchain-openai langchain-community langchain-experimental

# For token counting
pip install tiktoken

# For sentence splitting
pip install nltk spacy
python -m spacy download en_core_web_sm

# For semantic chunking
pip install sentence-transformers

# For LlamaIndex
pip install llama-index llama-index-embeddings-openai

# For Haystack
pip install haystack-ai

# Vector stores
pip install chromadb faiss-cpu
```

---

## Quick Start Template

```python
"""
Quick start template for document chunking in RAG applications.
Copy and customize for your use case.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def chunk_documents(
    texts: list[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: dict = None
) -> list[Document]:
    """
    Standard chunking function for RAG applications.
    
    Args:
        texts: List of text strings to chunk
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        metadata: Optional metadata to attach to all chunks
    
    Returns:
        List of Document objects ready for embedding
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    documents = []
    for i, text in enumerate(texts):
        doc_metadata = {"source_index": i}
        if metadata:
            doc_metadata.update(metadata)
        documents.append(Document(page_content=text, metadata=doc_metadata))
    
    return splitter.split_documents(documents)

# Usage
texts = ["Your document text here...", "Another document..."]
chunks = chunk_documents(texts, chunk_size=1000, chunk_overlap=200)

print(f"Created {len(chunks)} chunks")
for chunk in chunks[:3]:
    print(f"  - {len(chunk.page_content)} chars: {chunk.page_content[:50]}...")
```

---

*Document created: January 2026 | For RAG and document processing applications*
