# Text Chunking Methods: Comprehensive Guide for RAG Applications

*A practical reference for developers building retrieval-augmented generation (RAG) systems*

---

## The Core Principle

> **"What gets chunked gets retrieved"** - The quality of your RAG system directly depends on how you split your documents.

Chunking is the process of breaking down large documents into smaller, meaningful pieces that can be embedded and retrieved effectively. Poor chunking leads to poor retrieval, which leads to poor generation quality—regardless of how good your LLM is.

### 🎯 The Golden Rule of Chunking

**Optimize for retrieval first, generation second.** Your chunks need to be:
1. **Semantically coherent** - Each chunk should represent a complete thought or concept
2. **Self-contained** - Understandable without external context (within reason)
3. **Appropriately sized** - Large enough for context, small enough for precision
4. **Consistently structured** - Predictable format aids retrieval

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

### ⚠️ When to Avoid
- Production systems requiring quality retrieval
- Documents with semantic structure (paragraphs, sections)
- Any application where retrieval quality matters
- Legal, medical, or technical documents

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

### ✅ Best Practices (If You Must Use This Method)

1. **Always use overlap** - Minimum 10% of chunk_size to maintain context continuity
2. **Clean text first** - Remove excessive whitespace, normalize line breaks
3. **Post-process for sentence boundaries** - At minimum, try to avoid splitting mid-word
4. **Add metadata** - Track original position for debugging

```python
# Better fixed-size implementation with basic improvements
def improved_fixed_split(text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Fixed splitting with basic quality improvements."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to find a sentence boundary near the end
        if end < len(text):
            # Look for period + space within last 50 chars
            search_start = max(start, end - 50)
            last_period = text.rfind('. ', search_start, end)
            if last_period != -1:
                end = last_period + 1  # Include the period
        
        chunk = text[start:end].strip()
        chunks.append({
            'text': chunk,
            'start_pos': start,
            'end_pos': end,
            'length': len(chunk)
        })
        start = end - overlap
    
    return chunks
```

### ⚠️ Limitations
- Splits mid-word and mid-sentence
- No semantic awareness
- Poor retrieval quality for complex queries
- Only suitable for very uniform, simple text
- Creates confusing chunks that hurt user experience

### 💡 Pro Tip
**Don't use this in production.** Even if you're prototyping, start with Recursive Character Splitting—it's almost as easy and dramatically better. The only valid use case is when you literally need something working in the next 5 minutes and don't care about quality.

---

## 2. Recursive Character Text Splitting (LangChain Default)

### When to Use
- **General purpose** - best starting point for most applications
- Documents with natural paragraph/sentence structure
- When you want a good balance of simplicity and quality
- **Recommended default for 80% of use cases**

### 🎯 Why This Should Be Your Default

The recursive splitter is the Goldilocks solution: simple enough to understand, smart enough for production, and flexible enough for customization. Unless you have a specific reason to use something else, start here.

### How It Works

The splitter tries separators in order, falling back to smaller units only when necessary:
1. `\n\n` (paragraphs) → first choice (preserves semantic units)
2. `\n` (newlines) → if chunks still too large
3. `. ` (sentences) → further breakdown needed
4. ` ` (words) → near-atomic level
5. `` (characters) → last resort (rarely used)

### Implementation

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Basic usage - good defaults for most cases
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          # Target chunk size in characters
    chunk_overlap=200,        # Overlap between chunks (20%)
    length_function=len,      # How to measure length
    separators=["\n\n", "\n", ". ", " ", ""]  # Order matters!
)

# Split raw text
chunks = text_splitter.split_text(text)

# Split with metadata preservation (recommended)
documents = [Document(page_content=text, metadata={"source": "doc.pdf", "page": 1})]
split_docs = text_splitter.split_documents(documents)

# Each split_doc maintains the original metadata
for doc in split_docs[:3]:
    print(f"Source: {doc.metadata['source']}, Length: {len(doc.page_content)}")
```

### ✅ Best Practices

#### 1. **Always Preserve Metadata**
```python
# ❌ BAD: Loses source information
chunks = text_splitter.split_text(text)

# ✅ GOOD: Maintains traceability
documents = [Document(
    page_content=text, 
    metadata={
        "source": filename,
        "page": page_num,
        "section": section_name,
        "timestamp": datetime.now().isoformat()
    }
)]
chunks = text_splitter.split_documents(documents)
```

#### 2. **Optimize Chunk Size for Your Use Case**

```python
# For Q&A systems (prioritize precision)
qa_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,    # Smaller chunks = more precise retrieval
    chunk_overlap=150   # ~19% overlap
)

# For summarization (prioritize context)
summary_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,   # Larger chunks = more context
    chunk_overlap=300   # ~20% overlap
)

# For code documentation (balance both)
code_doc_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)
```

#### 3. **Maintain 10-20% Overlap**
```python
# ❌ BAD: No overlap - context loss at boundaries
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# ❌ BAD: Too much overlap - storage waste, redundant retrieval
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)

# ✅ GOOD: 20% overlap - balance between context and efficiency
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

#### 4. **Customize Separators for Domain-Specific Content**

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
        "\n\n\n",           # Major sections (3+ blank lines)
        "\n\n",             # Paragraphs/functions (2 blank lines)
        "\n```\n",          # Code block boundaries
        "\n",               # Lines
        ". ",               # Sentences
        " ",
        ""
    ]
)

# For scientific papers
paper_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=[
        "\n## ",            # Section headers
        "\n### ",           # Subsection headers
        "\n\n",             # Paragraphs
        "\n",
        ". ",
        " ",
        ""
    ]
)
```

### ⚠️ Common Mistakes to Avoid

```python
# ❌ MISTAKE 1: Ignoring document structure
# Don't use the same splitter for all document types
same_splitter_for_everything = RecursiveCharacterTextSplitter(chunk_size=1000)

# ✅ CORRECT: Adapt to document type
def get_splitter_for_type(doc_type: str):
    if doc_type == "legal":
        return legal_splitter
    elif doc_type == "code":
        return code_splitter
    elif doc_type == "academic":
        return paper_splitter
    else:
        return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# ❌ MISTAKE 2: Using arbitrary chunk sizes
splitter = RecursiveCharacterTextSplitter(chunk_size=873, chunk_overlap=123)

# ✅ CORRECT: Use round numbers based on testing
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# ❌ MISTAKE 3: Not validating chunk quality
chunks = splitter.split_text(text)  # Just trust it worked

# ✅ CORRECT: Validate and log chunk statistics
chunks = splitter.split_documents(documents)
validate_chunks(chunks)  # Check min/max sizes, empty chunks, etc.
```

### 💡 Pro Tips

#### Tip 1: Test Different Chunk Sizes on Your Data
```python
def find_optimal_chunk_size(documents, test_queries, sizes=[500, 1000, 1500, 2000]):
    """Test different chunk sizes to find optimal for your use case."""
    results = {}
    
    for size in sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=int(size * 0.2)  # 20% overlap
        )
        chunks = splitter.split_documents(documents)
        
        # Run test queries and measure retrieval quality
        scores = evaluate_retrieval(chunks, test_queries)
        results[size] = {
            'num_chunks': len(chunks),
            'avg_chunk_size': sum(len(c.page_content) for c in chunks) / len(chunks),
            'retrieval_score': scores
        }
    
    return results
```

#### Tip 2: Add Chunk IDs for Debugging
```python
def add_chunk_ids(chunks):
    """Add unique IDs to chunks for debugging and tracking."""
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = f"{chunk.metadata.get('source', 'unknown')}_{i}"
        chunk.metadata['chunk_index'] = i
        chunk.metadata['total_chunks'] = len(chunks)
    return chunks

chunks = text_splitter.split_documents(documents)
chunks = add_chunk_ids(chunks)
```

#### Tip 3: Monitor Chunk Size Distribution
```python
def analyze_chunk_distribution(chunks):
    """Analyze chunk size distribution to detect issues."""
    sizes = [len(c.page_content) for c in chunks]
    
    return {
        'total_chunks': len(chunks),
        'min_size': min(sizes),
        'max_size': max(sizes),
        'avg_size': sum(sizes) / len(sizes),
        'median_size': sorted(sizes)[len(sizes)//2],
        'std_dev': (sum((s - sum(sizes)/len(sizes))**2 for s in sizes) / len(sizes)) ** 0.5,
        'size_range': max(sizes) - min(sizes)
    }

# Use it
stats = analyze_chunk_distribution(chunks)
print(f"Chunk size stats: {stats}")

# Warning signs:
# - High std_dev (>500): Inconsistent chunking
# - Large size_range (>1500): Some chunks much bigger than others
# - Min_size < 100: Possibly broken chunks
```

### 🎯 Production Checklist

Before deploying recursive character splitting to production:

- [ ] Tested with representative sample of your documents
- [ ] Validated chunk size distribution is reasonable
- [ ] Confirmed chunks maintain semantic coherence
- [ ] Metadata preserved correctly through splitting
- [ ] Overlap configured (10-20% of chunk_size)
- [ ] Separators customized for your document type
- [ ] Monitoring in place to track chunk statistics
- [ ] Fallback handling for edge cases (very short/long docs)

---

## 3. Token-Based Splitting

### When to Use
- **Precise control over LLM context windows**
- When you need to guarantee chunks fit within token limits
- Multi-model pipelines with different tokenizers
- Cost optimization (staying under token limits)
- When character-based estimates aren't reliable enough

### 🎯 Why Tokens Matter More Than Characters

Characters lie. Tokens tell the truth. A 1000-character chunk might be 250 tokens or 400 tokens depending on the content. If you're working with context windows, cost limits, or need precise control, count tokens, not characters.

### Implementation with LangChain

```python
from langchain.text_splitter import TokenTextSplitter

# Using tiktoken (OpenAI's tokenizer)
splitter = TokenTextSplitter(
    chunk_size=512,                    # tokens, not characters
    chunk_overlap=50,                  # token overlap
    encoding_name="cl100k_base"        # GPT-4/3.5-turbo encoding
)
chunks = splitter.split_text(text)

print(f"Created {len(chunks)} chunks")
print(f"Max tokens per chunk: 512 (guaranteed)")
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

### ✅ Best Practices

#### 1. **Choose the Right Token Limit**

```python
# ❌ BAD: Using arbitrary numbers
splitter = TokenTextSplitter(chunk_size=537, chunk_overlap=42)

# ✅ GOOD: Align with your LLM's context window
# GPT-3.5-turbo: 4,096 tokens → use chunks of ~500-1000 tokens
# GPT-4: 8,192 tokens → use chunks of ~1000-2000 tokens  
# GPT-4-turbo/GPT-4o: 128,000 tokens → use chunks based on retrieval count

# For RAG with k=5 retrieved chunks:
context_budget = 8192  # GPT-4 context
overhead = 1000  # System prompt, query, response space
max_chunk_size = (context_budget - overhead) // 5  # ~1400 tokens per chunk

splitter = TokenTextSplitter(
    chunk_size=1400,
    chunk_overlap=140,  # 10%
    encoding_name="cl100k_base"
)
```

#### 2. **Match Tokenizer to Target Model**

```python
# ❌ BAD: Wrong tokenizer for your model
# Using GPT-4 tokenizer but deploying to Claude
splitter = TokenTextSplitter(encoding_name="cl100k_base")  # GPT tokenizer
# → Chunks might overflow Claude's limits!

# ✅ GOOD: Use correct tokenizer
# For OpenAI models
openai_splitter = TokenTextSplitter(
    chunk_size=512,
    encoding_name="cl100k_base"  # GPT-4, GPT-3.5-turbo
)

# For Claude (approximate with OpenAI tokenizer, then reduce ~15%)
claude_splitter = TokenTextSplitter(
    chunk_size=435,  # ~512 Claude tokens ≈ 512 * 0.85 OpenAI tokens
    encoding_name="cl100k_base"
)

# For multi-model pipelines: use smallest common denominator
multi_model_splitter = TokenTextSplitter(
    chunk_size=400,  # Safe for most models
    encoding_name="cl100k_base"
)
```

#### 3. **Consider Token-to-Character Ratio**

```python
# Calculate average token-to-char ratio for your corpus
def analyze_token_ratio(texts: list[str], model: str = "gpt-4"):
    """Understand token density of your documents."""
    encoding = tiktoken.encoding_for_model(model)
    
    ratios = []
    for text in texts:
        char_count = len(text)
        token_count = len(encoding.encode(text))
        ratios.append(char_count / token_count)
    
    avg_ratio = sum(ratios) / len(ratios)
    print(f"Average chars per token: {avg_ratio:.2f}")
    print(f"1000 tokens ≈ {int(1000 * avg_ratio)} characters")
    return avg_ratio

# Use results to choose between token and character splitting
# If ratio is stable (~4 chars/token): character splitting is fine
# If ratio varies widely: use token splitting for safety
```

#### 4. **Budget for Multiple Retrievals**

```python
def calculate_safe_chunk_size(
    model_context: int,
    system_prompt_tokens: int,
    query_tokens: int,
    response_tokens: int,
    num_retrievals: int = 5,
    safety_margin: float = 0.1
) -> int:
    """Calculate safe chunk size for RAG pipeline."""
    
    # Calculate available space
    available = model_context - system_prompt_tokens - query_tokens - response_tokens
    
    # Divide by number of chunks you'll retrieve
    per_chunk = available // num_retrievals
    
    # Apply safety margin
    safe_chunk_size = int(per_chunk * (1 - safety_margin))
    
    return safe_chunk_size

# Example: GPT-4 with 5 retrieved chunks
chunk_size = calculate_safe_chunk_size(
    model_context=8192,
    system_prompt_tokens=500,
    query_tokens=100,
    response_tokens=500,
    num_retrievals=5,
    safety_margin=0.1
)
print(f"Recommended chunk size: {chunk_size} tokens")  # ~1200 tokens
```

### ⚠️ Common Mistakes

```python
# ❌ MISTAKE 1: Forgetting about overlap in token budgets
total_budget = 8192
chunks_needed = 5
chunk_size = total_budget // chunks_needed  # 1638 tokens
# → WRONG! With 10% overlap, 5 chunks * 1638 = 8190 + overlap = OVERFLOW

# ✅ CORRECT: Account for overlap
overlap_ratio = 0.1
effective_tokens_per_chunk = chunk_size * (1 - overlap_ratio)
chunk_size = (total_budget // chunks_needed) / (1 + overlap_ratio)  # ~1489 tokens

# ❌ MISTAKE 2: Not validating actual token counts
splitter = TokenTextSplitter(chunk_size=512)
chunks = splitter.split_text(text)
# Assume all chunks are exactly 512 tokens → WRONG!

# ✅ CORRECT: Validate token counts
encoding = tiktoken.encoding_for_model("gpt-4")
for i, chunk in enumerate(chunks):
    actual_tokens = len(encoding.encode(chunk))
    if actual_tokens > 512:
        print(f"Warning: Chunk {i} has {actual_tokens} tokens (limit: 512)")

# ❌ MISTAKE 3: Ignoring semantic boundaries
# Token splitting can cut mid-sentence more often than character splitting
splitter = TokenTextSplitter(chunk_size=512)  # Pure token count

# ✅ BETTER: Combine with sentence awareness
from langchain.text_splitter import RecursiveCharacterTextSplitter
hybrid_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=512,
    chunk_overlap=50
)
# This respects semantic boundaries while counting tokens
```

### Token Count Reference

| Model Family | Encoding | ~Chars per Token | Typical Use |
|--------------|----------|------------------|-------------|
| GPT-4, GPT-3.5-turbo | cl100k_base | ~4 | Most OpenAI models |
| GPT-3 (davinci, curie) | p50k_base | ~4 | Legacy OpenAI |
| Codex | p50k_base | ~4 | Code models |
| Claude | claude tokenizer | ~3.5 | Anthropic models |
| Llama/Mistral | llama tokenizer | ~4-5 | Open source models |

### 💡 Pro Tips

#### Tip 1: Create a Token Budget Dashboard
```python
def token_budget_report(
    chunks: list[str],
    model: str = "gpt-4",
    retrievals: int = 5,
    context_limit: int = 8192
):
    """Generate token budget report for RAG pipeline."""
    encoding = tiktoken.encoding_for_model(model)
    
    chunk_tokens = [len(encoding.encode(c)) for c in chunks]
    
    max_retrieval_cost = max(chunk_tokens) * retrievals
    avg_retrieval_cost = sum(sorted(chunk_tokens, reverse=True)[:retrievals])
    
    return {
        'total_chunks': len(chunks),
        'min_chunk_tokens': min(chunk_tokens),
        'max_chunk_tokens': max(chunk_tokens),
        'avg_chunk_tokens': sum(chunk_tokens) / len(chunk_tokens),
        'max_retrieval_cost': max_retrieval_cost,
        'avg_retrieval_cost': avg_retrieval_cost,
        'context_remaining': context_limit - avg_retrieval_cost,
        'fits_in_context': max_retrieval_cost < context_limit
    }

# Usage
report = token_budget_report(chunks, model="gpt-4", retrievals=5)
print(f"Token budget analysis: {report}")
```

#### Tip 2: Optimize Chunk Size for Cost
```python
def optimize_for_cost(
    texts: list[str],
    embedding_cost_per_1k: float = 0.0001,  # $0.0001 per 1K tokens
    test_sizes: list[int] = [256, 512, 1024, 2048]
):
    """Find chunk size that minimizes embedding costs while maintaining quality."""
    
    encoding = tiktoken.encoding_for_model("gpt-4")
    
    for size in test_sizes:
        splitter = TokenTextSplitter(chunk_size=size, chunk_overlap=int(size * 0.1))
        
        total_tokens = 0
        total_chunks = 0
        
        for text in texts:
            chunks = splitter.split_text(text)
            total_chunks += len(chunks)
            for chunk in chunks:
                total_tokens += len(encoding.encode(chunk))
        
        cost = (total_tokens / 1000) * embedding_cost_per_1k
        
        print(f"Chunk size {size}:")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Embedding cost: ${cost:.4f}")
        print(f"  Avg tokens/chunk: {total_tokens / total_chunks:.0f}")
        print()
```

### 🎯 Production Checklist

- [ ] Correct tokenizer selected for target model
- [ ] Chunk size accounts for number of retrievals
- [ ] Overlap configured (10-20% of chunk_size)
- [ ] Safety margin applied to prevent context overflow
- [ ] Token counts validated on sample documents
- [ ] Cost implications calculated and acceptable
- [ ] Monitoring for token budget violations
- [ ] Fallback handling for edge cases

---

## 4. Sentence-Based Splitting

### When to Use
- Q&A systems where complete thoughts matter
- Conversational applications
- When semantic completeness of each chunk matters
- Content that should be understood as discrete units
- News articles, FAQs, transcripts

### 🎯 The Power of Complete Thoughts

Sentence splitting ensures each chunk is a complete, understandable unit. Unlike character or token splitting that might cut mid-thought, sentence boundaries create semantically coherent chunks that make sense in isolation.

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
text = """
Your document here. With multiple sentences. Each sentence is preserved intact.
This is the fourth sentence. And this is the fifth. No mid-sentence splits!
"""

chunks = sentence_split(text, sentences_per_chunk=5, overlap_sentences=1)
for i, chunk in enumerate(chunks):
    sentence_count = len(sent_tokenize(chunk))
    print(f"Chunk {i}: {sentence_count} sentences")
```

### Implementation with spaCy (More Accurate)

```python
import spacy

# Load spacy model (more accurate sentence detection)
nlp = spacy.load("en_core_web_sm")

def spacy_sentence_split(
    text: str,
    sentences_per_chunk: int = 5,
    overlap_sentences: int = 1
) -> list[str]:
    """
    Split text using spaCy's sentence detection (more accurate than NLTK).
    """
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
chunks = spacy_sentence_split(text, sentences_per_chunk=5, overlap_sentences=1)
```

### ✅ Best Practices

#### 1. **Adaptive Sentence Grouping**

```python
def adaptive_sentence_split(
    text: str,
    target_chunk_size: int = 1000,  # characters
    min_sentences: int = 2,
    max_sentences: int = 10
) -> list[str]:
    """
    Group sentences dynamically to hit target chunk size.
    Ensures complete sentences while respecting size constraints.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # Would adding this sentence exceed target?
        if current_length + sentence_length > target_chunk_size and len(current_chunk) >= min_sentences:
            # Finalize current chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
            
            # Force split if we hit max_sentences
            if len(current_chunk) >= max_sentences:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
    
    # Add remaining sentences
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Usage
chunks = adaptive_sentence_split(text, target_chunk_size=1000)
```

#### 2. **Handle Edge Cases**

```python
def robust_sentence_split(text: str, sentences_per_chunk: int = 5) -> list[dict]:
    """
    Sentence splitting with edge case handling and metadata.
    """
    # Normalize text first
    text = text.strip()
    if not text:
        return []
    
    sentences = sent_tokenize(text)
    
    # Handle very short documents
    if len(sentences) < sentences_per_chunk:
        return [{
            'text': text,
            'sentence_count': len(sentences),
            'is_complete': True,
            'warning': 'Document shorter than target chunk size'
        }]
    
    chunks = []
    step = sentences_per_chunk - 1  # 1 sentence overlap
    
    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        chunk_text = " ".join(chunk_sentences)
        
        chunks.append({
            'text': chunk_text,
            'sentence_count': len(chunk_sentences),
            'start_sentence': i,
            'end_sentence': i + len(chunk_sentences),
            'is_complete': len(chunk_sentences) == sentences_per_chunk
        })
    
    return chunks
```

#### 3. **Optimize for Question Answering**

```python
def qa_optimized_sentence_split(text: str) -> list[str]:
    """
    Split optimized for Q&A: each chunk is a self-contained answer unit.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    
    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence)
        
        # Check if this completes a thought
        # Heuristics: ends with period, next sentence starts with capital, etc.
        is_complete_thought = (
            sentence.endswith('.') and 
            i + 1 < len(sentences) and 
            sentences[i + 1][0].isupper() and
            len(" ".join(current_chunk)) >= 200  # Minimum chunk size
        )
        
        # Or if we've accumulated enough content
        is_long_enough = len(" ".join(current_chunk)) >= 800
        
        if is_complete_thought or is_long_enough:
            chunks.append(" ".join(current_chunk))
            # Keep last sentence for context
            current_chunk = [sentence]
    
    # Add remaining
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
```

### ⚠️ Common Mistakes

```python
# ❌ MISTAKE 1: Not handling abbreviations
# NLTK will split on "Dr." or "U.S.A." incorrectly
raw_split = sent_tokenize("Dr. Smith works at NASA. He studies climate.")
# → ['Dr.', 'Smith works at NASA.', 'He studies climate.']

# ✅ CORRECT: Use spaCy or custom preprocessing
def preprocess_abbreviations(text: str) -> str:
    """Replace common abbreviations before sentence splitting."""
    replacements = {
        'Dr.': 'Dr',
        'Mr.': 'Mr',
        'Mrs.': 'Mrs',
        'Ms.': 'Ms',
        'U.S.A.': 'USA',
        'U.S.': 'US',
        'etc.': 'etc',
        'vs.': 'vs'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

text = preprocess_abbreviations(text)
sentences = sent_tokenize(text)

# ❌ MISTAKE 2: Ignoring very short/long sentences
chunks = sentence_split(text, sentences_per_chunk=5)
# Some chunks might be way too short or too long

# ✅ CORRECT: Use adaptive approach with size constraints
chunks = adaptive_sentence_split(text, target_chunk_size=1000)

# ❌ MISTAKE 3: No validation of sentence boundaries
sentences = text.split('. ')  # Naive splitting
# Fails on decimals (3.14), abbreviations, etc.

# ✅ CORRECT: Use proper NLP libraries
sentences = sent_tokenize(text)  # or spaCy
```

### 💡 Pro Tips

#### Tip 1: Combine Sentence and Token Limits
```python
def sentence_split_with_token_limit(
    text: str,
    max_tokens: int = 512,
    min_sentences: int = 2,
    model: str = "gpt-4"
) -> list[str]:
    """
    Split by sentences but respect token limits.
    Best of both worlds.
    """
    import tiktoken
    
    encoding = tiktoken.encoding_for_model(model)
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence))
        
        if current_tokens + sentence_tokens > max_tokens and len(current_chunk) >= min_sentences:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
```

#### Tip 2: Preserve Sentence Structure in Metadata
```python
def sentence_split_with_metadata(text: str, sentences_per_chunk: int = 5) -> list[dict]:
    """Track sentence indices for debugging and reconstruction."""
    sentences = sent_tokenize(text)
    chunks = []
    
    for i in range(0, len(sentences), sentences_per_chunk - 1):
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        chunks.append({
            'text': " ".join(chunk_sentences),
            'sentences': chunk_sentences,
            'sentence_indices': list(range(i, i + len(chunk_sentences))),
            'total_sentences': len(sentences),
            'is_first_chunk': i == 0,
            'is_last_chunk': i + sentences_per_chunk >= len(sentences)
        })
    
    return chunks
```

### 🎯 Production Checklist

- [ ] Chose appropriate NLP library (NLTK for speed, spaCy for accuracy)
- [ ] Handled common abbreviations and edge cases
- [ ] Validated sentence boundaries on sample documents
- [ ] Considered combining with token limits
- [ ] Overlap configured (typically 1-2 sentences)
- [ ] Minimum and maximum sentence counts enforced
- [ ] Short document handling implemented
- [ ] Metadata tracking for debugging

---

## 5. Semantic Chunking (Embeddings-Based)

### When to Use
- **Quality is the top priority** over speed/cost
- Documents with varied topics mixed together
- When you need the most semantically coherent chunks possible
- Rich, complex content (research papers, books, transcripts)
- Budget allows for embedding costs during chunking

### 🎯 The Gold Standard for Quality

Semantic chunking uses embeddings to identify natural topic boundaries, creating chunks that group related content together. It's the most sophisticated approach and produces the highest quality results—at the cost of complexity and compute.

### How It Works

1. Split text into sentences
2. Embed each sentence
3. Calculate similarity between consecutive sentences
4. Identify boundary points where similarity drops significantly
5. Create chunks at these natural topic shifts

### Implementation with LangChain Experimental

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Basic semantic chunking
semantic_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation", "interquartile"
    breakpoint_threshold_amount=95  # Higher = fewer, larger chunks
)

semantic_chunks = semantic_splitter.split_text(text)

print(f"Created {len(semantic_chunks)} semantic chunks")
for i, chunk in enumerate(semantic_chunks[:3]):
    print(f"Chunk {i}: {len(chunk)} chars")
```

### Different Breakpoint Strategies

```python
# Strategy 1: Percentile (recommended default)
# Splits at points where similarity is in the bottom N percentile
percentile_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95  # Split at bottom 5% of similarities
)

# Strategy 2: Standard Deviation
# Splits when similarity drops more than N standard deviations below mean
std_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.5  # 1.5 std devs below mean
)

# Strategy 3: Interquartile Range
# Splits at outlier similarity scores
iqr_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="interquartile",
    breakpoint_threshold_amount=1.5  # 1.5 * IQR below Q1
)
```

### Manual Implementation for Full Control

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple

def manual_semantic_chunk(
    text: str,
    embeddings_model,  # Any model with .embed_documents() method
    threshold_percentile: int = 95
) -> List[str]:
    """
    Semantic chunking with full control over the process.
    
    Args:
        text: Input text to chunk
        embeddings_model: Model to generate sentence embeddings
        threshold_percentile: Percentile for determining split points
    
    Returns:
        List of semantically coherent chunks
    """
    # Step 1: Split into sentences
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    
    if len(sentences) < 3:
        return [text]  # Too short to chunk meaningfully
    
    # Step 2: Embed all sentences
    sentence_embeddings = embeddings_model.embed_documents(sentences)
    
    # Step 3: Calculate similarities between consecutive sentences
    similarities = []
    for i in range(len(sentence_embeddings) - 1):
        sim = cosine_similarity(
            [sentence_embeddings[i]], 
            [sentence_embeddings[i + 1]]
        )[0][0]
        similarities.append(sim)
    
    # Step 4: Find breakpoints (low similarity = topic shift)
    threshold = np.percentile(similarities, 100 - threshold_percentile)
    breakpoints = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]
    
    # Step 5: Create chunks
    chunks = []
    start = 0
    for breakpoint in breakpoints:
        chunk = " ".join(sentences[start:breakpoint])
        chunks.append(chunk)
        start = breakpoint
    
    # Add final chunk
    if start < len(sentences):
        chunks.append(" ".join(sentences[start:]))
    
    return chunks

# Usage
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
chunks = manual_semantic_chunk(text, embeddings, threshold_percentile=95)
```

### ✅ Best Practices

#### 1. **Choose the Right Threshold**

```python
def find_optimal_threshold(
    text: str,
    embeddings_model,
    test_percentiles: list = [90, 92, 95, 97, 99]
):
    """
    Test different thresholds to find optimal chunking.
    Higher percentile = fewer, larger chunks.
    """
    from nltk.tokenize import sent_tokenize
    
    results = {}
    
    for percentile in test_percentiles:
        splitter = SemanticChunker(
            embeddings=embeddings_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=percentile
        )
        
        chunks = splitter.split_text(text)
        
        results[percentile] = {
            'num_chunks': len(chunks),
            'avg_chunk_size': sum(len(c) for c in chunks) / len(chunks),
            'min_chunk_size': min(len(c) for c in chunks),
            'max_chunk_size': max(len(c) for c in chunks),
            'avg_sentences_per_chunk': sum(len(sent_tokenize(c)) for c in chunks) / len(chunks)
        }
        
        print(f"\nPercentile {percentile}:")
        print(f"  Chunks: {results[percentile]['num_chunks']}")
        print(f"  Avg size: {results[percentile]['avg_chunk_size']:.0f} chars")
        print(f"  Avg sentences: {results[percentile]['avg_sentences_per_chunk']:.1f}")
    
    return results

# Find what works for your documents
results = find_optimal_threshold(text, OpenAIEmbeddings())
```

#### 2. **Combine Semantic Chunking with Size Constraints**

```python
def semantic_chunk_with_limits(
    text: str,
    embeddings_model,
    min_chunk_size: int = 500,
    max_chunk_size: int = 2000,
    threshold_percentile: int = 95
) -> list[str]:
    """
    Semantic chunking with size constraints.
    Merges too-small chunks, splits too-large ones.
    """
    # Initial semantic chunking
    chunks = manual_semantic_chunk(text, embeddings_model, threshold_percentile)
    
    # Post-process to enforce size constraints
    final_chunks = []
    current_chunk = ""
    
    for chunk in chunks:
        # If chunk is too large, split it recursively
        if len(chunk) > max_chunk_size:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_size,
                chunk_overlap=200
            )
            sub_chunks = splitter.split_text(chunk)
            final_chunks.extend(sub_chunks)
            current_chunk = ""
        
        # If chunk is too small, merge with next
        elif len(chunk) < min_chunk_size:
            current_chunk += " " + chunk if current_chunk else chunk
            
            # If merged chunk is now large enough, finalize it
            if len(current_chunk) >= min_chunk_size:
                final_chunks.append(current_chunk)
                current_chunk = ""
        
        # Chunk is just right
        else:
            if current_chunk:
                final_chunks.append(current_chunk)
                current_chunk = ""
            final_chunks.append(chunk)
    
    # Add any remaining merged content
    if current_chunk:
        final_chunks.append(current_chunk)
    
    return final_chunks
```

#### 3. **Use Local Embeddings to Reduce Cost**

```python
# ❌ EXPENSIVE: Using API for semantic chunking large documents
from langchain_openai import OpenAIEmbeddings
api_embeddings = OpenAIEmbeddings()
# Cost: $0.0001 per 1K tokens, can add up quickly for chunking!

# ✅ COST-EFFECTIVE: Use local embeddings for chunking
from langchain_community.embeddings import HuggingFaceEmbeddings

local_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",  # Fast, good quality
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

semantic_splitter = SemanticChunker(
    embeddings=local_embeddings,  # Free after download!
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95
)

chunks = semantic_splitter.split_text(text)

# Then use higher-quality embeddings (OpenAI) for the actual vector store
```

#### 4. **Cache Embeddings for Repeated Processing**

```python
import hashlib
import pickle
from pathlib import Path

class CachedSemanticChunker:
    """Semantic chunker with embedding caching."""
    
    def __init__(self, embeddings_model, cache_dir: str = ".chunk_cache"):
        self.embeddings_model = embeddings_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text content."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def chunk(self, text: str, threshold_percentile: int = 95) -> list[str]:
        """Chunk with caching."""
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Check cache
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Compute chunks
        chunks = manual_semantic_chunk(text, self.embeddings_model, threshold_percentile)
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(chunks, f)
        
        return chunks

# Usage
chunker = CachedSemanticChunker(OpenAIEmbeddings())
chunks = chunker.chunk(text)  # First run: computes and caches
chunks = chunker.chunk(text)  # Second run: instant retrieval from cache
```

### ⚠️ Common Mistakes

```python
# ❌ MISTAKE 1: Using semantic chunking on everything
# It's overkill for simple, structured documents
all_docs_semantic = SemanticChunker(OpenAIEmbeddings())

# ✅ CORRECT: Use semantic chunking selectively
def choose_chunker(document_type: str):
    if document_type == "research_paper":
        return SemanticChunker(OpenAIEmbeddings())  # Complex, worth the cost
    elif document_type == "markdown_docs":
        return MarkdownHeaderTextSplitter()  # Structure-based is fine
    else:
        return RecursiveCharacterTextSplitter()  # Default

# ❌ MISTAKE 2: Not considering embedding cost
# Semantic chunking embeds every sentence during chunking!
docs = [very_long_document for _ in range(1000)]
for doc in docs:
    chunks = SemanticChunker(OpenAIEmbeddings()).split_text(doc)
# This could cost hundreds of dollars in API calls!

# ✅ CORRECT: Use local embeddings for chunking, API for storage
local_chunker = SemanticChunker(HuggingFaceEmbeddings())
chunks = local_chunker.split_text(doc)  # Free chunking

# Then embed with OpenAI for vector store (better retrieval)
final_embeddings = OpenAIEmbeddings().embed_documents([c.page_content for c in chunks])

# ❌ MISTAKE 3: Ignoring chunk size variability
# Semantic chunks can vary wildly in size
chunks = semantic_splitter.split_text(text)
# Some might be 100 chars, others 5000 chars!

# ✅ CORRECT: Apply size constraints
chunks = semantic_chunk_with_limits(text, embeddings, min_chunk_size=500, max_chunk_size=2000)
```

### 💡 Pro Tips

#### Tip 1: Visualize Semantic Boundaries
```python
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def visualize_semantic_boundaries(text: str, embeddings_model):
    """
    Visualize where semantic chunking will split.
    Helpful for understanding and debugging.
    """
    from nltk.tokenize import sent_tokenize
    
    sentences = sent_tokenize(text)
    embeddings = embeddings_model.embed_documents(sentences)
    
    # Calculate similarities
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        similarities.append(sim)
    
    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(similarities, marker='o')
    plt.axhline(y=np.percentile(similarities, 5), color='r', linestyle='--', 
                label='95th percentile threshold')
    plt.xlabel('Sentence Index')
    plt.ylabel('Similarity to Next Sentence')
    plt.title('Semantic Similarity Between Consecutive Sentences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('semantic_boundaries.png')
    
    print(f"Similarity range: {min(similarities):.3f} - {max(similarities):.3f}")
    print(f"Mean similarity: {np.mean(similarities):.3f}")
    print(f"95th percentile threshold: {np.percentile(similarities, 5):.3f}")

# Usage
visualize_semantic_boundaries(text, OpenAIEmbeddings())
```

#### Tip 2: Hybrid Approach - Semantic + Structure
```python
def hybrid_semantic_structure_split(text: str, embeddings_model) -> list[str]:
    """
    Use document structure as first pass, semantic chunking within sections.
    Best for structured documents with varied content.
    """
    # First: Split by major structure (e.g., sections)
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    structure_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,  # Large chunks for major sections
        separators=["\n\n## ", "\n\n# ", "\n\n"]
    )
    
    major_sections = structure_splitter.split_text(text)
    
    # Second: Semantic chunking within each section
    semantic_splitter = SemanticChunker(
        embeddings=embeddings_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )
    
    all_chunks = []
    for section in major_sections:
        semantic_chunks = semantic_splitter.split_text(section)
        all_chunks.extend(semantic_chunks)
    
    return all_chunks
```

### 🎯 Production Checklist

- [ ] Evaluated cost of embedding all sentences during chunking
- [ ] Considered using local embeddings for chunking phase
- [ ] Tested different threshold percentiles on sample documents
- [ ] Applied size constraints (min/max chunk sizes)
- [ ] Implemented caching to avoid re-computing chunks
- [ ] Validated chunk quality manually on sample outputs
- [ ] Monitoring in place for chunk size distribution
- [ ] Fallback to simpler method for edge cases (very short docs)

---

## 6. Document Structure-Based Splitting

### When to Use
- **Markdown documents** (README files, documentation, wikis)
- **HTML content** (web pages, articles, blogs)
- Documents with clear hierarchical structure
- Technical documentation with sections
- When preserving document hierarchy is important

### 🎯 Respect the Author's Intent

Document authors already told you where the logical boundaries are—they used headers! Structure-based splitting respects these boundaries while preserving the hierarchical context as metadata.

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
    strip_headers=False  # Keep headers in content for context
)

# Example markdown
markdown_doc = """
# User Guide

## Installation

### Prerequisites
You need Python 3.8+

### Setup
Run pip install package

## Usage

### Basic Example
Here's how to use it...
"""

# Split by headers first
md_header_splits = markdown_splitter.split_text(markdown_doc)

# Each split has metadata about its header hierarchy
for doc in md_header_splits:
    print(f"Headers: {doc.metadata}")
    print(f"Content: {doc.page_content[:100]}...")
    print()
    
# Output:
# Headers: {'Header 1': 'User Guide', 'Header 2': 'Installation', 'Header 3': 'Prerequisites'}
# Content: You need Python 3.8+
#
# Headers: {'Header 1': 'User Guide', 'Header 2': 'Installation', 'Header 3': 'Setup'}
# Content: Run pip install package
```

### Further Split Large Sections

```python
# If sections are still too large, split them further
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100
)

final_splits = text_splitter.split_documents(md_header_splits)

# Each final split maintains the header metadata!
for doc in final_splits[:3]:
    print(f"Headers: {doc.metadata}")
    print(f"Content length: {len(doc.page_content)}")
```

### HTML Splitting

```python
from langchain.text_splitter import HTMLHeaderTextSplitter

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
html_document = """
<html>
<body>
<h1>Main Title</h1>
<p>Introduction paragraph</p>
<h2>Section One</h2>
<p>Section one content</p>
<h3>Subsection</h3>
<p>Subsection content</p>
</body>
</html>
"""

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

### ✅ Best Practices

#### 1. **Preserve Header Hierarchy in Metadata**

```python
# ✅ GOOD: Full header path preserved
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False  # Keep headers in content
)

splits = splitter.split_text(markdown_doc)

# Now each chunk knows its full context:
# {'Header 1': 'User Guide', 'Header 2': 'Installation', 'Header 3': 'Prerequisites'}
```

#### 2. **Decide Whether to Strip Headers**

```python
# Option 1: Keep headers (better for context)
splitter_with_headers = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False  # Header text included in chunk
)

# Chunk content: "### Prerequisites\nYou need Python 3.8+"
# Better for: RAG where LLM needs section context

# Option 2: Strip headers (avoid duplication)
splitter_no_headers = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=True  # Headers only in metadata
)

# Chunk content: "You need Python 3.8+"
# Better for: When headers are redundant with metadata
```

#### 3. **Handle Missing or Inconsistent Headers**

```python
def robust_markdown_split(markdown_text: str) -> list:
    """
    Handle markdown with inconsistent header structure.
    """
    # First try structure-based splitting
    try:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        
        chunks = md_splitter.split_text(markdown_text)
        
        # If we got chunks, validate them
        if chunks and len(chunks) > 1:
            return chunks
    
    except Exception as e:
        print(f"Markdown splitting failed: {e}")
    
    # Fallback to recursive character splitting
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return fallback_splitter.split_text(markdown_text)
```

#### 4. **Clean HTML Before Splitting**

```python
from bs4 import BeautifulSoup

def clean_and_split_html(html_content: str) -> list:
    """
    Clean HTML and split by structure.
    """
    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove unwanted elements
    for element in soup(['script', 'style', 'nav', 'footer', 'iframe']):
        element.decompose()
    
    # Extract main content if present
    main_content = soup.find('main') or soup.find('article') or soup.find('body')
    
    if main_content:
        cleaned_html = str(main_content)
    else:
        cleaned_html = str(soup)
    
    # Now split by headers
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
    
    html_splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    
    return html_splitter.split_text(cleaned_html)
```

#### 5. **Combine Structure + Size Constraints**

```python
def structure_split_with_size_limits(
    markdown_text: str,
    max_chunk_size: int = 1500,
    min_chunk_size: int = 300
) -> list:
    """
    Split by structure, then enforce size constraints.
    """
    # First pass: split by headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    
    structural_chunks = md_splitter.split_text(markdown_text)
    
    # Second pass: handle size constraints
    final_chunks = []
    
    for chunk in structural_chunks:
        chunk_size = len(chunk.page_content)
        
        # Too large? Split further
        if chunk_size > max_chunk_size:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_size,
                chunk_overlap=200
            )
            sub_chunks = text_splitter.split_documents([chunk])
            final_chunks.extend(sub_chunks)
        
        # Too small? Consider merging (or keep as-is if it's important)
        elif chunk_size < min_chunk_size:
            # Keep it anyway - structural boundaries are important
            # Add warning in metadata
            chunk.metadata['warning'] = 'below_min_size'
            final_chunks.append(chunk)
        
        # Just right
        else:
            final_chunks.append(chunk)
    
    return final_chunks
```

### ⚠️ Common Mistakes

```python
# ❌ MISTAKE 1: Not preserving header context
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1")],
    strip_headers=True  # Loses important context!
)
# Now you can't tell which section a chunk came from

# ✅ CORRECT: Preserve headers in content or metadata
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1")],
    strip_headers=False  # Keep context
)

# ❌ MISTAKE 2: Assuming all markdown is well-structured
# Some markdown has inconsistent headers or none at all
chunks = markdown_splitter.split_text(markdown)  # Might return 1 giant chunk!

# ✅ CORRECT: Validate and fallback
chunks = robust_markdown_split(markdown)

# ❌ MISTAKE 3: Not handling HTML elements in markdown
# Many markdown processors allow raw HTML
markdown_with_html = "## Section\n<div>Content</div>"
# Structure splitter might not handle this well

# ✅ CORRECT: Clean or convert first
from markdownify import markdownify as md
clean_markdown = md(markdown_with_html)
chunks = markdown_splitter.split_text(clean_markdown)
```

### 💡 Pro Tips

#### Tip 1: Extract and Enrich Metadata
```python
def enrich_markdown_chunks(chunks: list) -> list:
    """
    Add additional metadata from document structure.
    """
    for i, chunk in enumerate(chunks):
        # Add navigation hints
        chunk.metadata['chunk_id'] = i
        chunk.metadata['total_chunks'] = len(chunks)
        
        # Add breadcrumb path
        headers = chunk.metadata
        breadcrumb = " > ".join([
            headers.get('Header 1', ''),
            headers.get('Header 2', ''),
            headers.get('Header 3', '')
        ]).strip(' >')
        chunk.metadata['breadcrumb'] = breadcrumb
        
        # Estimate reading time (250 words per minute)
        word_count = len(chunk.page_content.split())
        chunk.metadata['reading_time_seconds'] = int((word_count / 250) * 60)
        
        # Detect code blocks
        chunk.metadata['has_code'] = '```' in chunk.page_content
    
    return chunks
```

#### Tip 2: Handle Tables and Lists Specially
```python
def split_markdown_preserve_tables(markdown_text: str) -> list:
    """
    Ensure tables and lists aren't split mid-structure.
    """
    import re
    
    # Identify table boundaries
    table_pattern = r'\|.*\|.*\n\|[-:\s|]+\|.*(?:\n\|.*\|.*)*'
    
    # Temporarily replace tables with placeholders
    tables = re.findall(table_pattern, markdown_text)
    for i, table in enumerate(tables):
        markdown_text = markdown_text.replace(table, f"{{{{TABLE_{i}}}}}")
    
    # Split normally
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
        strip_headers=False
    )
    chunks = splitter.split_text(markdown_text)
    
    # Restore tables
    for chunk in chunks:
        for i, table in enumerate(tables):
            chunk.page_content = chunk.page_content.replace(f"{{{{TABLE_{i}}}}}", table)
    
    return chunks
```

### 🎯 Production Checklist

- [ ] Tested with representative sample of markdown/HTML documents
- [ ] Decided on header stripping strategy (keep vs remove)
- [ ] Metadata preservation validated
- [ ] Fallback to character splitting implemented
- [ ] HTML cleaning applied (remove scripts, styles, nav)
- [ ] Size constraints considered for large sections
- [ ] Special elements (tables, code blocks) handled
- [ ] Breadcrumb/hierarchy tracking in metadata

---

## 10. Parent Document Retrieval Pattern

### When to Use
- You want to **retrieve on small chunks** (precise matching)
- But **pass larger context** to the LLM (better generation)
- Best of both worlds approach
- When retrieval precision and generation quality both matter
- Production RAG systems

### 🎯 The Best of Both Worlds

This is arguably the most powerful chunking pattern for production RAG. Small chunks give you precise retrieval (finding the needle), while large parent chunks give your LLM the full context (understanding the haystack).

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

### ✅ Best Practices

#### 1. **Optimize Child/Parent Ratio**

```python
# ❌ BAD: Child and parent too similar in size
child_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# → Defeats the purpose!

# ✅ GOOD: 1:4 to 1:6 ratio (child:parent)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,   # Small for precise retrieval
    chunk_overlap=50
)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # 5x larger for context
    chunk_overlap=200
)

# 💡 PRO TIP: Adjust based on query types
# Specific factual queries → smaller ratio (1:3)
# Complex reasoning queries → larger ratio (1:6)
```

#### 2. **Use Full Documents as Parents for Short Docs**

```python
from langchain.retrievers import ParentDocumentRetriever

# For short documents (< 3000 chars), use full doc as parent
full_doc_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    ),
    # No parent_splitter = full documents are parents
)

# This works great for:
# - Email threads
# - Support tickets
# - Short articles
# - FAQ entries
```

#### 3. **Track Parent-Child Relationships**

```python
def create_parent_document_retriever_with_tracking(
    documents: list,
    child_chunk_size: int = 400,
    parent_chunk_size: int = 2000
):
    """
    Enhanced retriever that tracks relationships.
    """
    from collections import defaultdict
    
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=int(child_chunk_size * 0.1)
    )
    
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=int(parent_chunk_size * 0.1)
    )
    
    vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
    docstore = InMemoryStore()
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    
    retriever.add_documents(documents)
    
    # Track statistics
    parent_count = len(docstore.mget(list(docstore.yield_keys())))
    child_count = vectorstore._collection.count()
    
    stats = {
        'original_docs': len(documents),
        'parent_chunks': parent_count,
        'child_chunks': child_count,
        'avg_children_per_parent': child_count / parent_count if parent_count > 0 else 0,
        'child_size': child_chunk_size,
        'parent_size': parent_chunk_size,
        'ratio': parent_chunk_size / child_chunk_size
    }
    
    print(f"Parent-Child Retriever Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return retriever, stats
```

#### 4. **Implement Hybrid Retrieval Strategy**

```python
class HybridParentChildRetriever:
    """
    Retrieves both child chunks (precise) and parents (context).
    Returns both for maximum flexibility.
    """
    
    def __init__(self, parent_retriever, vectorstore):
        self.parent_retriever = parent_retriever
        self.vectorstore = vectorstore
    
    def get_relevant_documents(self, query: str, k: int = 5):
        """
        Returns both child matches and their parents.
        """
        # Get child chunks (precise matches)
        child_docs = self.vectorstore.similarity_search(query, k=k)
        
        # Get parents through retriever
        parent_docs = self.parent_retriever.get_relevant_documents(query, k=k)
        
        return {
            'child_matches': child_docs,     # For highlighting
            'parent_context': parent_docs,   # For generation
            'query': query
        }
    
    def format_for_llm(self, results: dict) -> str:
        """
        Format results for LLM consumption.
        """
        formatted = f"Query: {results['query']}\n\n"
        
        formatted += "Relevant Excerpts:\n"
        for i, child in enumerate(results['child_matches'], 1):
            formatted += f"{i}. {child.page_content}\n"
        
        formatted += "\nFull Context:\n"
        for i, parent in enumerate(results['parent_context'], 1):
            formatted += f"\n--- Context Block {i} ---\n{parent.page_content}\n"
        
        return formatted

# Usage
hybrid_retriever = HybridParentChildRetriever(
    parent_retriever=retriever,
    vectorstore=vectorstore
)

results = hybrid_retriever.get_relevant_documents("What is the main point?")
prompt = hybrid_retriever.format_for_llm(results)
```

### ⚠️ Common Mistakes

```python
# ❌ MISTAKE 1: Not using persistent storage for production
# InMemoryStore loses everything on restart!
docstore = InMemoryStore()

# ✅ CORRECT: Use persistent storage
from langchain.storage import LocalFileStore
docstore = LocalFileStore("./parent_doc_store")

# Or for production:
from langchain_community.storage import RedisStore
docstore = RedisStore(redis_url="redis://localhost:6379")

# ❌ MISTAKE 2: Forgetting to deduplicate parent chunks
# Multiple child chunks might map to same parent
results = retriever.get_relevant_documents(query, k=10)
# Might get same parent multiple times!

# ✅ CORRECT: Deduplicate parents
def deduplicate_parents(docs: list):
    """Remove duplicate parent chunks."""
    seen = set()
    unique_docs = []
    
    for doc in docs:
        # Use content hash as key
        content_hash = hash(doc.page_content)
        if content_hash not in seen:
            seen.add(content_hash)
            unique_docs.append(doc)
    
    return unique_docs

results = retriever.get_relevant_documents(query, k=10)
unique_results = deduplicate_parents(results)

# ❌ MISTAKE 3: Not monitoring retrieval quality
# You don't know if child chunks are finding right parents

# ✅ CORRECT: Add logging and validation
def retrieve_with_validation(retriever, query: str, k: int = 5):
    """Retrieve with quality checks."""
    results = retriever.get_relevant_documents(query, k=k)
    
    # Log retrieval info
    for i, doc in enumerate(results):
        print(f"Result {i}:")
        print(f"  Length: {len(doc.page_content)} chars")
        print(f"  Source: {doc.metadata.get('source', 'unknown')}")
        print(f"  Snippet: {doc.page_content[:100]}...")
    
    return results
```

### 💡 Pro Tips

#### Tip 1: Optimize Based on Query Complexity
```python
class AdaptiveParentChildRetriever:
    """
    Adjusts child/parent ratio based on query complexity.
    """
    
    def __init__(self, documents):
        self.simple_retriever = self._create_retriever(
            child_size=300, parent_size=1000  # 1:3 for simple queries
        )
        self.complex_retriever = self._create_retriever(
            child_size=400, parent_size=2400  # 1:6 for complex queries
        )
        
        for retriever in [self.simple_retriever, self.complex_retriever]:
            retriever.add_documents(documents)
    
    def _create_retriever(self, child_size, parent_size):
        return ParentDocumentRetriever(
            vectorstore=Chroma(embedding_function=OpenAIEmbeddings()),
            docstore=InMemoryStore(),
            child_splitter=RecursiveCharacterTextSplitter(
                chunk_size=child_size,
                chunk_overlap=int(child_size * 0.1)
            ),
            parent_splitter=RecursiveCharacterTextSplitter(
                chunk_size=parent_size,
                chunk_overlap=int(parent_size * 0.1)
            )
        )
    
    def retrieve(self, query: str, k: int = 5):
        """Choose retriever based on query complexity."""
        # Simple heuristic: short queries = simple, long = complex
        if len(query.split()) <= 5:
            return self.simple_retriever.get_relevant_documents(query, k=k)
        else:
            return self.complex_retriever.get_relevant_documents(query, k=k)
```

#### Tip 2: Create Custom Parent Splitter
```python
def create_semantic_parent_retriever(documents: list):
    """
    Use semantic chunking for parents, character splitting for children.
    Best quality but more expensive.
    """
    from langchain_experimental.text_splitter import SemanticChunker
    
    # Children: small, fast character splits
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    
    # Parents: semantic chunks for coherence
    parent_splitter = SemanticChunker(
        embeddings=OpenAIEmbeddings(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )
    
    retriever = ParentDocumentRetriever(
        vectorstore=Chroma(embedding_function=OpenAIEmbeddings()),
        docstore=InMemoryStore(),
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    
    retriever.add_documents(documents)
    return retriever
```

### 🎯 Production Checklist

- [ ] Child-to-parent ratio optimized (tested 1:3, 1:4, 1:5, 1:6)
- [ ] Persistent storage configured (not InMemoryStore)
- [ ] Parent deduplication implemented
- [ ] Retrieval quality monitored and logged
- [ ] Appropriate for document length (full docs for short content)
- [ ] Memory usage acceptable (storing both child and parent)
- [ ] Performance acceptable (two-stage retrieval adds latency)
- [ ] Fallback handling for edge cases

---

## Choosing the Right Method: Decision Tree

```
START: Chunking Method Selection
  │
  ├─ Quick prototype needed?
  │   └─ YES ─────────────────────────► Recursive Character (LangChain default)
  │                                      - Fast to implement
  │                                      - Good enough for most cases
  │                                      - Easy to customize later
  │
  ├─ Need precise token control?
  │   └─ YES ─────────────────────────► Token-Based Splitting
  │                                      - Guaranteed fit in context window
  │                                      - Cost optimization
  │                                      - Multi-model compatibility
  │
  ├─ Document has clear structure?
  │   ├─ Markdown ────────────────────► MarkdownHeaderTextSplitter
  │   │                                  + RecursiveCharacterTextSplitter
  │   ├─ HTML ────────────────────────► HTMLHeaderTextSplitter  
  │   │                                  + Clean unwanted elements first
  │   └─ Code ────────────────────────► Language-specific Splitter
  │                                      or Code-Aware Custom Splitter
  │
  ├─ Quality is critical?
  │   ├─ Can afford compute ──────────► Semantic Chunking
  │   │                                  - Highest quality
  │   │                                  - Use local embeddings to reduce cost
  │   │                                  - Consider caching
  │   │
  │   └─ Budget constrained ──────────► Sentence-Based + Recursive
  │                                      or Adaptive Sentence Grouping
  │
  ├─ Need both precision and context?
  │   └─ YES ─────────────────────────► Parent Document Retriever
  │                                      - Best of both worlds
  │                                      - Small chunks for retrieval
  │                                      - Large chunks for generation
  │                                      - Production-ready pattern
  │
  ├─ Complex hierarchical documents?
  │   └─ YES ─────────────────────────► Hierarchical Chunking
  │                                      or Structure-Based + Semantic
  │
  └─ Still unsure?
      └─────────────────────────────────► Start with Recursive Character
                                           - Test on sample documents
                                           - Measure retrieval quality
                                           - Iterate based on results
```

---

## Key Parameters Cheat Sheet

| Parameter | Typical Range | Impact | Best Practices |
|-----------|---------------|--------|----------------|
| `chunk_size` | 500-2000 chars<br>256-512 tokens | **Smaller** = more precise retrieval<br>**Larger** = more context | Start with 1000, adjust based on query types |
| `chunk_overlap` | 10-20% of chunk_size | Prevents context loss at boundaries | Use 200 chars for 1000 char chunks |
| `similarity_threshold` | 0.7-0.85 | For semantic chunking;<br>higher = fewer splits | Start at 0.75, increase if too fragmented |
| `breakpoint_percentile` | 85-95 | Higher = fewer, larger semantic chunks | 95 for most cases, 90 for fine-grained |
| `sentences_per_chunk` | 3-10 | For sentence-based splitting | 5 is a good default |
| `max_tokens` | Model dependent | Must fit context window minus overhead | GPT-4: 1400 tokens/chunk for k=5 retrieval |
| `child_chunk_size` | 300-500 | Parent-child retrieval precision | 400 works well for most content |
| `parent_chunk_size` | 1500-2500 | Parent-child context | 2000 (5x child size) is good default |

### Quick Reference: When to Use What

| Chunk Size | Use Case | Example |
|------------|----------|---------|
| **200-500** | Precise factual retrieval | FAQ, definitions, specific facts |
| **500-1000** | Balanced Q&A systems | General knowledge base, documentation |
| **1000-1500** | Rich context retrieval | Technical docs, research papers |
| **1500-2000** | Summarization tasks | Long-form content, articles |
| **2000+** | Parent chunks only | Context for generation, not retrieval |

---

## Performance Comparison

| Method | Speed | Memory | Quality | Cost | Recommended For |
|--------|-------|--------|---------|------|-----------------|
| **Fixed Character** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | Free | Prototypes only |
| **Recursive Character** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Free | **Default choice** |
| **Token-Based** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Free | Precise control needed |
| **Sentence-Based** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Free | Q&A systems |
| **Semantic (local model)** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Free | Best quality, free |
| **Semantic (API)** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $$$ | When compute matters more than cost |
| **Structure-Based** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Free | Markdown/HTML docs |
| **Parent Document** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $$ | **Production RAG** |

### Cost Analysis (Based on 1M characters)

| Method | Chunking Cost | Embedding Cost* | Total Cost |
|--------|---------------|-----------------|------------|
| Recursive Character | $0 | ~$0.04 | **$0.04** |
| Token-Based | $0 | ~$0.04 | **$0.04** |
| Semantic (API) | ~$0.40 | ~$0.04 | **$0.44** |
| Semantic (local) | $0 | ~$0.04 | **$0.04** |
| Parent Document | $0 | ~$0.08** | **$0.08** |

*Assuming OpenAI ada-002 embeddings at $0.0001/1K tokens
**Double cost: embedding both child and parent chunks

---

## Common Pitfalls and Solutions

### 1. Chunks Too Small → Lost Context

**Problem:** Lost context, fragmented information, poor generation quality

**Symptoms:**
- LLM responses are incomplete or lack context
- Retrieval finds the right chunk but answer is wrong
- You need to retrieve 10+ chunks to get a complete answer

**Solutions:**
```python
# ❌ BAD: Chunks too small
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

# ✅ SOLUTION 1: Increase chunk size
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# ✅ SOLUTION 2: Use parent document retriever
retriever = ParentDocumentRetriever(
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=200),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1000)
)

# ✅ SOLUTION 3: Increase overlap significantly
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150  # 30% overlap
)
```

### 2. Chunks Too Large → Poor Retrieval Precision

**Problem:** Diluted relevance, poor retrieval precision, irrelevant content included

**Symptoms:**
- Retrieval brings back too much irrelevant information
- Hard to find specific facts or details
- Similarity scores are all similar (low discrimination)

**Solutions:**
```python
# ❌ BAD: Chunks too large
splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)

# ✅ SOLUTION 1: Decrease chunk size
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# ✅ SOLUTION 2: Use semantic chunking
semantic_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95  # More granular splits
)

# ✅ SOLUTION 3: Use parent document pattern
# Search small, return large
retriever = ParentDocumentRetriever(
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000)
)
```

### 3. Splitting Mid-Thought → Semantic Incoherence

**Problem:** Semantic boundaries not respected, confusing chunks

**Symptoms:**
- Chunks end mid-sentence or mid-thought
- Important information split across chunks
- Chunks don't make sense in isolation

**Solutions:**
```python
# ❌ BAD: Character splitting without regard for content
splitter = CharacterTextSplitter(chunk_size=1000)

# ✅ SOLUTION 1: Use recursive character splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    separators=["\n\n", "\n", ". ", " ", ""]  # Respects boundaries
)

# ✅ SOLUTION 2: Use sentence-based splitting
sentence_splitter = sentence_split(
    text,
    sentences_per_chunk=5,
    overlap_sentences=1
)

# ✅ SOLUTION 3: Use semantic chunking
semantic_splitter = SemanticChunker(embeddings=OpenAIEmbeddings())
```

### 4. Lost Metadata → Can't Trace Sources

**Problem:** Can't trace chunks back to source documents

**Symptoms:**
- Don't know which document a chunk came from
- Can't cite sources in generated responses
- Debugging is impossible

**Solutions:**
```python
# ❌ BAD: Splitting raw text
chunks = text_splitter.split_text(text)

# ✅ SOLUTION: Always use Document objects with metadata
from langchain.schema import Document

documents = [
    Document(
        page_content=text,
        metadata={
            "source": filename,
            "page": page_num,
            "section": section_name,
            "document_id": doc_id,
            "timestamp": datetime.now().isoformat()
        }
    )
]

chunks = text_splitter.split_documents(documents)

# Metadata is automatically preserved in each chunk!
for chunk in chunks:
    print(chunk.metadata)  # All original metadata present
```

### 5. Inconsistent Chunk Sizes → Unpredictable Behavior

**Problem:** Some chunks much smaller/larger than target

**Symptoms:**
- Wide variation in chunk sizes
- Some chunks are just a few words
- Some chunks exceed token limits

**Solutions:**
```python
# ❌ BAD: Not validating chunk sizes
chunks = splitter.split_text(text)

# ✅ SOLUTION: Validate and post-process
def validate_and_fix_chunks(chunks: list, min_size: int = 200, max_size: int = 2000):
    """
    Ensure chunks meet size constraints.
    """
    fixed_chunks = []
    current_merge = ""
    
    for chunk in chunks:
        size = len(chunk.page_content) if hasattr(chunk, 'page_content') else len(chunk)
        
        # Too small? Merge with next
        if size < min_size:
            current_merge += " " + chunk if current_merge else chunk
            if len(current_merge) >= min_size:
                fixed_chunks.append(current_merge)
                current_merge = ""
        
        # Too large? Split further
        elif size > max_size:
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_size,
                chunk_overlap=200
            )
            sub_chunks = sub_splitter.split_text(chunk)
            fixed_chunks.extend(sub_chunks)
        
        # Just right
        else:
            fixed_chunks.append(chunk)
    
    # Add any remaining merged content
    if current_merge:
        fixed_chunks.append(current_merge)
    
    return fixed_chunks

chunks = validate_and_fix_chunks(chunks)
```

### 6. Not Testing on Real Data → Surprises in Production

**Problem:** Chunking works in testing but fails on real documents

**Solutions:**
```python
# ✅ BEST PRACTICE: Test thoroughly before deploying
def test_chunking_strategy(documents: list, splitter) -> dict:
    """
    Comprehensive testing of chunking strategy.
    """
    all_chunks = []
    
    for doc in documents:
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)
    
    chunk_sizes = [len(c.page_content) for c in all_chunks]
    
    report = {
        'total_documents': len(documents),
        'total_chunks': len(all_chunks),
        'chunks_per_doc': len(all_chunks) / len(documents),
        'min_chunk_size': min(chunk_sizes),
        'max_chunk_size': max(chunk_sizes),
        'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
        'median_chunk_size': sorted(chunk_sizes)[len(chunk_sizes)//2],
        'std_deviation': (sum((s - sum(chunk_sizes)/len(chunk_sizes))**2 
                              for s in chunk_sizes) / len(chunk_sizes)) ** 0.5,
        
        # Quality checks
        'very_small_chunks': sum(1 for s in chunk_sizes if s < 200),
        'very_large_chunks': sum(1 for s in chunk_sizes if s > 3000),
        'empty_chunks': sum(1 for c in all_chunks if not c.page_content.strip()),
    }
    
    # Print warnings
    if report['std_deviation'] > 500:
        print("⚠️ WARNING: High size variation - consider reviewing splitter")
    
    if report['very_small_chunks'] > len(all_chunks) * 0.1:
        print("⚠️ WARNING: >10% of chunks are very small (<200 chars)")
    
    if report['empty_chunks'] > 0:
        print(f"⚠️ WARNING: {report['empty_chunks']} empty chunks detected!")
    
    return report

# Run before deploying
report = test_chunking_strategy(sample_documents, splitter)
print(report)
```

---

## Production Checklist

Before deploying your chunking strategy to production:

### Configuration
- [ ] Chunking method selected based on decision tree
- [ ] Chunk size optimized for your use case (tested multiple values)
- [ ] Overlap configured (typically 10-20%)
- [ ] Metadata preservation implemented and tested
- [ ] Token limits respected (if applicable)
- [ ] Separators customized for your document types

### Quality Assurance
- [ ] Tested on representative sample of documents
- [ ] Validated chunk size distribution is acceptable
- [ ] Confirmed chunks maintain semantic coherence
- [ ] Verified metadata preservation through splitting
- [ ] Checked for empty or malformed chunks
- [ ] Tested edge cases (very short/long documents)

### Performance & Cost
- [ ] Chunking latency is acceptable
- [ ] Memory usage is acceptable
- [ ] Embedding costs calculated and approved
- [ ] Considered using local embeddings for semantic chunking
- [ ] Caching strategy implemented (if applicable)

### Monitoring & Observability
- [ ] Chunk statistics tracked (size, count, distribution)
- [ ] Retrieval quality metrics defined
- [ ] Logging implemented for debugging
- [ ] Alerting set up for anomalies (empty chunks, size violations)
- [ ] Regular review process established

### Documentation
- [ ] Chunking strategy documented
- [ ] Parameters and rationale recorded
- [ ] Known limitations documented
- [ ] Fallback strategies defined
- [ ] Team trained on chunking approach

---

## Installation Requirements

```bash
# Core LangChain
pip install langchain langchain-openai langchain-community langchain-experimental

# For token counting
pip install tiktoken

# For sentence splitting
pip install nltk
python -m nltk.downloader punkt punkt_tab

# For spaCy (more accurate sentence detection)
pip install spacy
python -m spacy download en_core_web_sm

# For semantic chunking (local)
pip install sentence-transformers

# For HTML parsing
pip install beautifulsoup4

# Vector stores
pip install chromadb faiss-cpu

# For visualization
pip install matplotlib numpy scikit-learn

# Optional: LlamaIndex
pip install llama-index llama-index-embeddings-openai

# Optional: Haystack
pip install haystack-ai
```

---

## Quick Start Template

```python
"""
Production-ready chunking template.
Copy and customize for your RAG application.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from datetime import datetime
from typing import List

def chunk_documents_for_rag(
    texts: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: dict = None,
    validate: bool = True
) -> List[Document]:
    """
    Production-ready document chunking with validation.
    
    Args:
        texts: List of text strings to chunk
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks (10-20% of chunk_size)
        metadata: Optional metadata to attach to all chunks
        validate: Whether to validate chunk quality
    
    Returns:
        List of Document objects ready for embedding
    """
    # Initialize splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Create documents with metadata
    documents = []
    for i, text in enumerate(texts):
        doc_metadata = {
            "source_index": i,
            "chunk_timestamp": datetime.now().isoformat()
        }
        if metadata:
            doc_metadata.update(metadata)
        
        documents.append(Document(
            page_content=text,
            metadata=doc_metadata
        ))
    
    # Split documents
    chunks = splitter.split_documents(documents)
    
    # Validate if requested
    if validate:
        chunks = _validate_chunks(chunks, min_size=200, max_size=3000)
    
    # Add chunk IDs
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        chunk.metadata['total_chunks'] = len(chunks)
    
    return chunks

def _validate_chunks(chunks: List[Document], min_size: int, max_size: int) -> List[Document]:
    """Internal validation and cleanup."""
    # Remove empty chunks
    chunks = [c for c in chunks if c.page_content.strip()]
    
    # Warn about size issues
    sizes = [len(c.page_content) for c in chunks]
    
    if min(sizes) < min_size:
        print(f"⚠️ Warning: Smallest chunk is {min(sizes)} chars (min: {min_size})")
    
    if max(sizes) > max_size:
        print(f"⚠️ Warning: Largest chunk is {max(sizes)} chars (max: {max_size})")
    
    std_dev = (sum((s - sum(sizes)/len(sizes))**2 for s in sizes) / len(sizes)) ** 0.5
    if std_dev > 500:
        print(f"⚠️ Warning: High size variation (std dev: {std_dev:.0f})")
    
    return chunks

# ============================================================
# USAGE EXAMPLES
# ============================================================

# Example 1: Simple usage
texts = ["Your document text here...", "Another document..."]
chunks = chunk_documents_for_rag(texts)

print(f"✅ Created {len(chunks)} chunks")
for chunk in chunks[:3]:
    print(f"  - {len(chunk.page_content)} chars: {chunk.page_content[:50]}...")

# Example 2: With metadata
chunks = chunk_documents_for_rag(
    texts,
    chunk_size=1000,
    chunk_overlap=200,
    metadata={"source": "user_manual.pdf", "version": "2.1"}
)

# Example 3: For specific use cases
# Q&A system (smaller chunks for precision)
qa_chunks = chunk_documents_for_rag(texts, chunk_size=800, chunk_overlap=150)

# Summarization (larger chunks for context)
summary_chunks = chunk_documents_for_rag(texts, chunk_size=1500, chunk_overlap=300)

# Example 4: Parent-child retrieval
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def create_parent_child_retriever(documents: List[Document]):
    """Create production-ready parent-child retriever."""
    
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    
    vectorstore = Chroma(
        collection_name="rag_chunks",
        embedding_function=OpenAIEmbeddings()
    )
    
    docstore = InMemoryStore()  # Use persistent store in production!
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    
    retriever.add_documents(documents)
    
    print(f"✅ Parent-child retriever created")
    print(f"   Vector store: {vectorstore._collection.count()} child chunks")
    
    return retriever
```

---

## Further Resources

### Documentation
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [LlamaIndex Chunking](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/)
- [Haystack Document Splitters](https://docs.haystack.deepset.ai/docs/documentsplitter)

### Research & Best Practices
- [Chunking Strategies for RAG](https://www.pinecone.io/learn/chunking-strategies/)
- [The Impact of Chunking on RAG Performance](https://arxiv.org/abs/2307.03172)
- [Greg Kamradt's Chunking Visualization](https://github.com/FullStackRetrieval-com/RetrievalTutorials)

### Tools
- [LangSmith](https://smith.langchain.com/) - Tracing and debugging
- [Chunkviz](https://chunkviz.up.railway.app/) - Visualize chunking strategies

---

*Document created: January 2026 | Enhanced with embedded best practices | For production RAG applications*

**Remember: The best chunking strategy is the one that works for YOUR data and YOUR use cases. Test, measure, iterate.**
