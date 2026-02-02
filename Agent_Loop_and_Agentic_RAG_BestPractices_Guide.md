# Agent Loop & Agentic RAG Best Practices Guide

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Quick Reference Overview](#quick-reference-overview)
3. [Decision Tree](#decision-tree)
4. [Agent Fundamentals](#1-agent-fundamentals)
5. [The Agent Loop Pattern](#2-the-agent-loop-pattern)
6. [LangChain Core Concepts](#3-langchain-core-concepts)
7. [LangGraph Architecture](#4-langgraph-architecture)
8. [Tool Definition & Binding](#5-tool-definition--binding)
9. [ReAct Agent Implementation](#6-react-agent-implementation)
10. [Vector Database with Qdrant](#7-vector-database-with-qdrant)
11. [Agentic RAG Implementation](#8-agentic-rag-implementation)
12. [Middleware System](#9-middleware-system)
13. [Memory & Persistence](#10-memory--persistence)
14. [Streaming & Observability](#11-streaming--observability)
15. [Local Models with Ollama](#12-local-models-with-ollama)
16. [Production Considerations](#13-production-considerations)
17. [Installation Requirements](#installation-requirements)
18. [Quick Start Template](#quick-start-template)
19. [Quick Reference Cheat Sheet](#14-quick-reference)

---

## Executive Summary

This guide covers the complete architecture for building **agentic RAG systems** using LangChain and LangGraph. An agent is an LLM-powered system that can reason about tasks, select and execute tools, observe results, and iterate until completion. When combined with RAG (Retrieval-Augmented Generation), agents gain the ability to dynamically retrieve relevant context from knowledge bases.

**Key architectural decisions covered**:
- **High-level API** (`create_react_agent`) vs **low-level control** (custom `StateGraph`)
- **Cloud models** (OpenAI) vs **local models** (Ollama)
- **Middleware patterns** for guardrails, rate limiting, and human-in-the-loop

The guide emphasizes production-ready patterns with embedded best practices in every code example.

---

## Quick Reference Overview

| Approach | Best For | Complexity | Control | Production-Ready |
|----------|----------|------------|---------|------------------|
| **`create_react_agent`** | Quick prototypes, simple agents | Low | Limited | Prototyping |
| **Custom `StateGraph`** | Production systems, complex flows | Medium | Full | ✅ Recommended |
| **Agentic RAG** | Knowledge-based Q&A | Medium | High | ✅ Yes |
| **Local (Ollama)** | Privacy-sensitive, cost-conscious | Medium | Full | ✅ Yes |
| **Cloud (OpenAI)** | Best quality, scalability | Low | Limited | ✅ Yes |

---

## Decision Tree

```
START: What kind of agent do you need?
  │
  ├─► Simple Q&A with tools? ──────────────────► create_react_agent (High-level API)
  │
  ├─► Need custom routing/state? ──────────────► Custom StateGraph (Low-level)
  │
  ├─► Knowledge-based retrieval?
  │   ├─► Single-pass sufficient? ─────────────► Traditional RAG Chain
  │   └─► Multi-step reasoning needed? ────────► Agentic RAG
  │
  ├─► Privacy/cost concerns?
  │   ├─► Full privacy required? ──────────────► Ollama + Local Embeddings
  │   └─► Cost optimization? ──────────────────► Hybrid (local embed, cloud LLM)
  │
  └─► Production deployment?
      ├─► Multi-server? ───────────────────────► PostgreSQL Checkpointer
      └─► Single-server? ──────────────────────► SQLite Checkpointer
```

---

## 1. Agent Fundamentals

### What is an Agent?

An agent is an LLM that can **decide** which actions to take, **execute** those actions via tools, and **iterate** based on observations until the task is complete.

**When to Use**: When tasks require dynamic decision-making, multi-step reasoning, or tool selection based on context.

### ⚠️ When to Avoid

- **Simple, deterministic tasks** - Use a basic chain instead (faster, cheaper, more predictable)
- **Single-step operations** - No need for the overhead of agent loop iterations
- **Highly structured workflows** - If you know the exact sequence, use a chain or graph without LLM routing
- **Budget constraints with predictable tasks** - Agents cost more due to multiple LLM calls

```python
# Conceptual agent structure
from typing import Literal

def agent_decision(state: dict) -> Literal["use_tool", "respond", "clarify"]:
    """
    Agent decides next action based on:
    1. User query
    2. Available tools
    3. Previous observations
    """
    # LLM reasoning happens here
    pass
```

**Key Insight**: Agents are not just chains—they have autonomy to choose their path through a problem.

**Common Pitfall**: Don't use agents for simple, deterministic tasks. A basic chain is faster and more predictable.

### LLM-Based Decision Making

| Aspect | Chain | Agent |
|--------|-------|-------|
| **Control Flow** | Fixed, predetermined | Dynamic, LLM-decided |
| **Tool Usage** | Sequential or parallel | As-needed, iterative |
| **Complexity** | Lower | Higher |
| **Predictability** | High | Variable |
| **Cost** | Lower (fewer calls) | Higher (multiple iterations) |

#### ⚠️ Common Mistakes to Avoid

```python
# ❌ BAD: Using an agent for a simple deterministic task
agent = create_react_agent(llm, [format_date_tool])
result = agent.invoke({"messages": [("user", "Format today's date")]})
# Wasteful: Multiple LLM calls for something predictable

# ✅ GOOD: Use a simple chain for deterministic tasks
from datetime import datetime
formatted = datetime.now().strftime("%Y-%m-%d")  # No LLM needed

# ❌ BAD: No iteration limits - potential infinite loop and runaway costs
agent = create_react_agent(llm, tools)  # Missing max_iterations!

# ✅ GOOD: Always set iteration limits in production
agent = create_react_agent(llm, tools, max_iterations=10)
```

### 💡 Pro Tips

#### Tip 1: Start Simple, Add Agency Later

Begin with a basic chain. Only upgrade to an agent when you genuinely need dynamic decision-making.

```python
# Start with this
chain = prompt | llm | parser

# Upgrade to agent ONLY when you need dynamic tool selection
agent = create_react_agent(llm, tools)
```

#### Tip 2: Use Agents for "Unknown Unknowns"

Agents excel when you can't predict all the steps needed. If you can write a flowchart of the exact steps, you probably don't need an agent.

---

## 2. The Agent Loop Pattern

The agent loop is the core execution pattern: **Reason → Act → Observe → Repeat**.

### Loop Phases

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """State maintained across the agent loop."""
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Best Practice: Use add_messages reducer

# The Loop:
# 1. MODEL CALL PHASE: LLM reasons about state, decides action
# 2. TOOL CALL PHASE: Execute selected tool(s)
# 3. OBSERVATION PHASE: Tool results added to state
# 4. TERMINATION CHECK: Continue or return final response
```

### Loop Termination Conditions

```python
from langchain_core.messages import AIMessage

def should_continue(state: AgentState) -> str:
    """Determine if agent should continue or stop."""
    last_message = state["messages"][-1]

    # Best Practice: Check for tool_calls attribute, not message type
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"  # More tools to execute
    return "end"  # No more tool calls, return response
```

**Key Insight**: The loop terminates when the LLM generates a response without any tool calls.

**Common Pitfall**: Infinite loops. Always implement iteration limits (see Middleware section).

---

## 3. LangChain Core Concepts

### Runnables Interface

All LangChain components implement the **Runnable** interface with standardized methods.

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# Three invocation patterns:
result = llm.invoke("Hello")           # Single input
results = llm.batch(["Hi", "Hello"])   # Multiple inputs
async for chunk in llm.astream("Hi"):  # Streaming
    print(chunk.content, end="")

# Best Practice: Use batch() for multiple independent calls (parallel execution)
```

### LCEL (LangChain Expression Language)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Explain {topic} simply.")
output_parser = StrOutputParser()

# Pipe operator chains components
chain = prompt | llm | output_parser  # Best Practice: Read left-to-right as data flow

# Equivalent to:
# output_parser.invoke(llm.invoke(prompt.invoke({"topic": "agents"})))

response = chain.invoke({"topic": "agents"})
```

**Key Insight**: LCEL enables declarative composition—each `|` passes output to next component's input.

---

## 4. LangGraph Architecture

LangGraph provides fine-grained control over agent execution through a graph-based runtime.

### Core Constructs

#### State

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    State flows through all nodes.
    Best Practice: Use TypedDict for type safety and IDE support.
    """
    messages: Annotated[list, add_messages]  # add_messages = append, don't replace
    context: str  # Custom state fields as needed
```

#### Nodes

```python
from langchain_core.messages import AIMessage

def agent_node(state: AgentState) -> dict:
    """
    Nodes are Python functions that:
    1. Receive current state
    2. Perform computation (LLM call, tool execution, etc.)
    3. Return state updates (partial dict)

    Best Practice: Return only changed fields, not entire state.
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}  # add_messages reducer handles append
```

#### Edges

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(AgentState)

# Normal edge: Fixed routing
graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")  # After tools, always go back to agent

# Conditional edge: Dynamic routing based on state
graph.add_conditional_edges(
    "agent",
    should_continue,  # Function that returns next node name
    {
        "continue": "tools",
        "end": END
    }
)
```

### Complete Graph Building

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

def build_agent_graph(llm, tools: list) -> CompiledStateGraph:
    """Build a complete ReAct agent graph."""

    llm_with_tools = llm.bind_tools(tools)

    def agent(state: AgentState) -> dict:
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def should_continue(state: AgentState) -> str:
        if state["messages"][-1].tool_calls:
            return "tools"
        return END

    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))  # Best Practice: Use prebuilt ToolNode

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()
```

---

## 5. Tool Definition & Binding

### The @tool Decorator

```python
from langchain_core.tools import tool
from typing import Annotated

@tool
def search_knowledge_base(
    query: Annotated[str, "The search query to find relevant documents"]
) -> str:
    """
    Search the knowledge base for relevant information.

    Best Practice: Docstring is CRITICAL - the LLM uses it to decide when to call this tool.
    Be specific about what the tool does and when to use it.
    """
    # Implementation
    results = retriever.invoke(query)
    return format_results(results)

@tool
def calculate_bmi(
    weight_kg: Annotated[float, "Weight in kilograms"],
    height_m: Annotated[float, "Height in meters"]
) -> str:
    """Calculate Body Mass Index (BMI) given weight and height."""
    bmi = weight_kg / (height_m ** 2)
    return f"BMI: {bmi:.1f}"
```

**Key Insight**: The docstring serves as the tool's "instruction manual" for the LLM. Poor docstrings = poor tool selection.

### Binding Tools to LLM

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Best Practice: temperature=0 for tool use

tools = [search_knowledge_base, calculate_bmi]
llm_with_tools = llm.bind_tools(tools)

# Now the LLM can decide to call these tools
response = llm_with_tools.invoke("What's the BMI for someone 70kg and 1.75m tall?")
print(response.tool_calls)  # [{'name': 'calculate_bmi', 'args': {'weight_kg': 70, 'height_m': 1.75}}]
```

#### ⚠️ Common Mistakes to Avoid

```python
# ❌ BAD: Vague or missing docstring
@tool
def search(query: str) -> str:
    """Search."""  # LLM has no idea when to use this!
    return retriever.invoke(query)

# ✅ GOOD: Detailed, specific docstring
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for factual information about the company's products.
    Use this when the user asks about features, pricing, or documentation.
    Do NOT use for general knowledge questions.
    """
    return retriever.invoke(query)

# ❌ BAD: temperature > 0 for tool use (unpredictable)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ✅ GOOD: temperature = 0 for deterministic tool selection
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

### 💡 Pro Tips

#### Tip 1: Test Tool Docstrings Independently

The docstring is your tool's "advertisement" to the LLM. Test it by asking: "Would I know when to use this tool based only on the docstring?"

```python
# Good docstring checklist:
# - What does it do?
# - When should it be used?
# - When should it NOT be used?
# - What format does it return?
```

#### Tip 2: Use Annotated for Complex Arguments

```python
from typing import Annotated

@tool
def book_meeting(
    date: Annotated[str, "Date in ISO format YYYY-MM-DD"],
    duration_mins: Annotated[int, "Meeting duration (15, 30, or 60 minutes)"],
    attendees: Annotated[list[str], "List of email addresses"]
) -> str:
    """Book a meeting on the calendar."""
    # The Annotated descriptions help the LLM format arguments correctly
    pass
```

---

## 6. ReAct Agent Implementation

### High-Level API (create_react_agent)

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [search_knowledge_base, calculate_bmi]

# One-liner agent creation
agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier="You are a helpful assistant."  # System prompt
)

# Best Practice: create_react_agent returns a CompiledStateGraph
# Use for quick prototyping; switch to custom graph for more control

response = agent.invoke({"messages": [("user", "What's my BMI if I'm 70kg, 1.75m?")]})
```

### Low-Level Implementation (Full Control)

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
Always explain your reasoning before using tools.
Cite sources when providing information from the knowledge base."""

def create_custom_agent(llm, tools: list, system_prompt: str = SYSTEM_PROMPT):
    """
    Custom agent with full control over behavior.
    Best Practice: Use this when you need custom routing, state, or middleware.
    """

    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        # Inject system prompt if not present
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()
```

### API Comparison

| Feature | `create_react_agent` | Custom `StateGraph` |
|---------|---------------------|---------------------|
| **Setup** | One line | 15-30 lines |
| **Custom State** | Limited | Full control |
| **Custom Routing** | No | Yes |
| **Middleware** | Yes | Yes |
| **Learning Curve** | Low | Medium |
| **Production Use** | Prototyping | Recommended |

#### ⚠️ Common Mistakes to Avoid

```python
# ❌ BAD: Using create_react_agent in production without customization
agent = create_react_agent(llm, tools)  # No error handling, no limits, no logging!

# ✅ GOOD: Proper production configuration
agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier=SYSTEM_PROMPT,
    checkpointer=PostgresSaver.from_conn_string("postgresql://..."),
)

# ❌ BAD: Not handling empty tool lists
agent = create_react_agent(llm, [])  # Will fail when LLM tries to use tools

# ✅ GOOD: Validate tools before agent creation
if not tools:
    raise ValueError("Agent requires at least one tool")
agent = create_react_agent(llm, tools)
```

### 💡 Pro Tips

#### Tip 1: Prototype with High-Level, Deploy with Low-Level

Start with `create_react_agent` for rapid prototyping. Once you understand the flow, migrate to custom `StateGraph` for production control.

```python
# Development: Quick iteration
dev_agent = create_react_agent(llm, tools)

# Production: Full control with custom state, routing, middleware
prod_agent = build_custom_agent(llm, tools, custom_state_fields)
```

#### Tip 2: Use State Modifier for Dynamic Prompts

```python
def get_system_prompt(user_role: str) -> str:
    """Generate role-specific system prompts."""
    return f"You are a {user_role} assistant with access to tools..."

agent = create_react_agent(
    llm,
    tools,
    state_modifier=get_system_prompt("customer support")
)
```

---

## 7. Vector Database with Qdrant

### Collection Configuration

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Development: In-memory (fast, no persistence)
client = QdrantClient(":memory:")

# Production: Hosted or local server
# client = QdrantClient(url="http://localhost:6333")
# client = QdrantClient(url="https://xyz.qdrant.io", api_key="...")

# Create collection
client.create_collection(
    collection_name="knowledge_base",
    vectors_config=VectorParams(
        size=1536,  # Must match embedding model dimensions
        distance=Distance.COSINE  # Best Practice: Cosine for normalized embeddings
    )
)
```

### Document Ingestion

```python
from qdrant_client.models import PointStruct
from langchain_openai import OpenAIEmbeddings
import uuid

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_documents(documents: list[dict], client: QdrantClient):
    """
    Ingest documents with embeddings and metadata.
    Best Practice: Store source metadata for citations.
    """
    points = []

    for doc in documents:
        embedding = embeddings.embed_query(doc["content"])

        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "content": doc["content"],
                "source": doc.get("source", "unknown"),
                "chunk_index": doc.get("chunk_index", 0)
            }
        ))

    # Best Practice: Batch upsert for performance
    client.upsert(collection_name="knowledge_base", points=points)
```

### Retriever Interface

```python
from langchain_qdrant import QdrantVectorStore

# LangChain integration
vectorstore = QdrantVectorStore(
    client=client,
    collection_name="knowledge_base",
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Best Practice: Start with k=5, tune based on evaluation
)
```

---

## 8. Agentic RAG Implementation

### Traditional RAG vs Agentic RAG

| Aspect | Traditional RAG | Agentic RAG |
|--------|-----------------|-------------|
| **Retrieval** | Always retrieve | Agent decides when |
| **Query** | User query as-is | Agent reformulates |
| **Iterations** | Single pass | Multiple if needed |
| **Control** | Chain-based | Agent-based |
| **Flexibility** | Limited | High |

### RAG Tool Implementation

```python
from langchain_core.tools import tool
from typing import Annotated

@tool
def search_documents(
    query: Annotated[str, "Search query - be specific and detailed"]
) -> str:
    """
    Search the knowledge base for relevant information.
    Use this tool when you need factual information from documents.
    Always cite the source in your response.

    Best Practice: Detailed docstring helps LLM know WHEN to use this tool.
    """
    results = retriever.invoke(query)

    if not results:
        return "No relevant documents found."

    # Format with sources for citation
    formatted = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        formatted.append(f"[{i}] {doc.page_content}\n   Source: {source}")

    return "\n\n".join(formatted)
```

### Complete Agentic RAG Agent

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

AGENTIC_RAG_PROMPT = """You are a research assistant with access to a knowledge base.

When answering questions:
1. Search the knowledge base for relevant information
2. Synthesize information from multiple sources if needed
3. Always cite your sources using [1], [2], etc.
4. If information is not in the knowledge base, say so clearly

Do not make up information. Only use facts from retrieved documents."""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [search_documents]

agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier=AGENTIC_RAG_PROMPT
)

# Usage
response = agent.invoke({
    "messages": [("user", "What are the key benefits of RAG systems?")]
})
```

#### ⚠️ Common Mistakes to Avoid

```python
# ❌ BAD: RAG tool without source citations
@tool
def search_docs(query: str) -> str:
    """Search documents."""
    results = retriever.invoke(query)
    return "\n".join(d.page_content for d in results)  # No way to verify sources!

# ✅ GOOD: Always include source metadata
@tool
def search_docs(query: str) -> str:
    """Search documents with source citations."""
    results = retriever.invoke(query)
    formatted = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        formatted.append(f"[{i}] {doc.page_content}\n   Source: {source}")
    return "\n\n".join(formatted)

# ❌ BAD: Vague RAG tool description
@tool
def search(query: str) -> str:
    """Search for information."""  # When should the LLM use this?
    return retriever.invoke(query)

# ✅ GOOD: Specific instructions about when to use the tool
@tool
def search_product_docs(query: str) -> str:
    """
    Search product documentation for technical specifications and features.
    Use ONLY for questions about our products.
    Do NOT use for pricing, sales, or general knowledge.
    """
    return retriever.invoke(query)
```

### 💡 Pro Tips

#### Tip 1: Multi-Stage RAG with Separate Tools

Create specialized tools for different knowledge bases:

```python
@tool
def search_technical_docs(query: str) -> str:
    """Search technical documentation."""
    return tech_retriever.invoke(query)

@tool
def search_marketing_content(query: str) -> str:
    """Search marketing materials and product descriptions."""
    return marketing_retriever.invoke(query)

# Agent decides which knowledge base to query
agent = create_react_agent(llm, [search_technical_docs, search_marketing_content])
```

#### Tip 2: Implement Query Rewriting in the Tool

```python
@tool
def smart_search(query: str) -> str:
    """
    Search knowledge base with automatic query optimization.
    Handles typos, synonyms, and reformulation.
    """
    # Agent-driven query expansion
    optimized_queries = [
        query,
        query.replace("", "synonym"),  # Domain-specific synonyms
        f"{query} overview"  # Broader search
    ]

    all_results = []
    for q in optimized_queries:
        all_results.extend(retriever.invoke(q))

    # Deduplicate and rank
    unique_results = list({doc.page_content: doc for doc in all_results}.values())
    return format_results(unique_results[:5])
```

### 🎯 Production Checklist

Before deploying agentic RAG to production:

- [ ] Tool docstrings clearly specify when to use each retrieval source
- [ ] Source citations included in all retrieved content
- [ ] Tested with queries that require NO retrieval (agent should respond directly)
- [ ] Tested with queries requiring MULTIPLE retrievals
- [ ] Fallback behavior defined when no relevant documents found
- [ ] Retrieval latency monitored and optimized
- [ ] Cost per query measured (embedding + LLM calls)

---

## 9. Middleware System

Middleware intercepts agent execution at key points for logging, guardrails, and control.

### Middleware Hooks

```python
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.messages import BaseMessage
from typing import Sequence

# before_model: Called before each LLM call
def log_before_model(state: AgentState) -> AgentState:
    """Log incoming messages before LLM processes them."""
    print(f"[BEFORE MODEL] Messages: {len(state['messages'])}")
    return state

# after_model: Called after each LLM response
def log_after_model(state: AgentState) -> AgentState:
    """Log LLM response and any tool calls."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        print(f"[AFTER MODEL] Tool calls: {[tc['name'] for tc in last.tool_calls]}")
    return state
```

### ModelCallLimitMiddleware (Built-in)

```python
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

# Prevent infinite loops with call limits
agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier=system_prompt,
    # Best Practice: Always set limits in production
    max_iterations=10  # Maximum agent loop iterations
)
```

### Custom Middleware: Rate Limiting

```python
import time
from functools import wraps

class RateLimitMiddleware:
    """Rate limit LLM calls to avoid API throttling."""

    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.call_times = []

    def __call__(self, state: AgentState) -> AgentState:
        now = time.time()
        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if now - t < 60]

        if len(self.call_times) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.call_times[0])
            print(f"[RATE LIMIT] Sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)

        self.call_times.append(now)
        return state
```

### Custom Middleware: Content Guardrails

```python
from langchain_core.messages import AIMessage

BLOCKED_PATTERNS = ["password", "api_key", "secret"]

def content_guardrail(state: AgentState) -> AgentState:
    """
    Block responses containing sensitive information.
    Best Practice: Implement guardrails as middleware, not in prompts alone.
    """
    last = state["messages"][-1]

    if isinstance(last, AIMessage):
        content = last.content.lower()
        for pattern in BLOCKED_PATTERNS:
            if pattern in content:
                raise ValueError(f"Response blocked: contains '{pattern}'")

    return state
```

---

## 10. Memory & Persistence

### MemorySaver Checkpointer

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# In-memory persistence (development)
memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory  # Enable conversation memory
)

# Best Practice: Use thread_id to maintain separate conversations
config = {"configurable": {"thread_id": "user-123"}}

# First turn
response1 = agent.invoke(
    {"messages": [("user", "My name is Alice")]},
    config=config
)

# Second turn - agent remembers context
response2 = agent.invoke(
    {"messages": [("user", "What's my name?")]},
    config=config
)
# Agent: "Your name is Alice"
```

### Production Persistence

```python
# For production, use persistent checkpointers:
# - PostgreSQL: langgraph.checkpoint.postgres.PostgresSaver
# - SQLite: langgraph.checkpoint.sqlite.SqliteSaver
# - Redis: Custom implementation

from langgraph.checkpoint.sqlite import SqliteSaver

# SQLite for single-server deployments
memory = SqliteSaver.from_conn_string("checkpoints.db")

# PostgreSQL for multi-server deployments
# from langgraph.checkpoint.postgres import PostgresSaver
# memory = PostgresSaver.from_conn_string("postgresql://...")
```

---

## 11. Streaming & Observability

### Streaming Agent Responses

```python
# Stream node-by-node updates
for event in agent.stream(
    {"messages": [("user", "Search for RAG best practices")]},
    stream_mode="updates"  # Best Practice: Use "updates" for granular control
):
    for node_name, node_output in event.items():
        print(f"--- {node_name} ---")
        if "messages" in node_output:
            for msg in node_output["messages"]:
                if hasattr(msg, "content"):
                    print(msg.content[:100])
```

### LangSmith Integration

```python
import os

# Enable tracing (set environment variables)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "agentic-rag-prod"

# All agent runs are now traced automatically
# View at: https://smith.langchain.com

# Best Practice: Use projects to organize traces
# - Development: "agentic-rag-dev"
# - Staging: "agentic-rag-staging"
# - Production: "agentic-rag-prod"
```

### Graph Visualization

```python
from IPython.display import Image, display

# Export as Mermaid diagram (works in notebooks)
display(Image(agent.get_graph().draw_mermaid_png()))

# Or get ASCII representation
print(agent.get_graph().draw_ascii())

# Output:
#      +-----------+
#      |  __start__|
#      +-----------+
#            |
#            v
#      +-----------+
#      |   agent   |
#      +-----------+
#            |
#      +-----+-----+
#      |           |
#      v           v
# +---------+  +-------+
# |  tools  |  | __end__|
# +---------+  +-------+
#      |
#      +---> (back to agent)
```

---

## 12. Local Models with Ollama

### Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start server
ollama serve

# Pull models
ollama pull llama3.1:8b          # Chat model
ollama pull nomic-embed-text     # Embedding model (768 dimensions)
```

### LangChain Integration

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Chat model
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    # Best Practice: Set timeout for local models
    request_timeout=120.0
)

# Embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"  # 768 dimensions
)

# Use exactly like cloud models
agent = create_react_agent(model=llm, tools=tools)
```

### Trade-offs: Local vs Cloud

| Aspect | Local (Ollama) | Cloud (OpenAI) |
|--------|----------------|----------------|
| **Privacy** | Full control | Data sent externally |
| **Cost** | Hardware only | Per-token pricing |
| **Latency** | Depends on hardware | Generally lower |
| **Model Quality** | Good (Llama 3.1) | Best (GPT-4o) |
| **Setup** | More complex | API key only |
| **Scaling** | Limited by hardware | Unlimited |

---

## 13. Production Considerations

### Error Handling

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def resilient_agent_call(agent, messages: list, config: dict):
    """
    Agent invocation with automatic retry.
    Best Practice: Wrap agent calls with retry logic for transient failures.
    """
    try:
        return agent.invoke({"messages": messages}, config=config)
    except Exception as e:
        logger.warning(f"Agent call failed, retrying: {e}")
        raise
```

### Monitoring Metrics

```python
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentMetrics:
    """Track agent performance metrics."""
    query: str
    total_duration_ms: float
    llm_calls: int
    tool_calls: int
    tokens_used: int
    success: bool
    error: Optional[str] = None

def tracked_invoke(agent, messages, config):
    """Invoke agent with metrics tracking."""
    start = time.perf_counter()
    metrics = AgentMetrics(
        query=messages[-1][1] if messages else "",
        total_duration_ms=0,
        llm_calls=0,
        tool_calls=0,
        tokens_used=0,
        success=False
    )

    try:
        # Stream to count iterations
        for event in agent.stream({"messages": messages}, config=config):
            if "agent" in event:
                metrics.llm_calls += 1
            if "tools" in event:
                metrics.tool_calls += 1

        metrics.success = True
    except Exception as e:
        metrics.error = str(e)
    finally:
        metrics.total_duration_ms = (time.perf_counter() - start) * 1000

    return metrics
```

### Cost Optimization

```python
# 1. Use cheaper models for simple tool selection
tool_selection_llm = ChatOpenAI(model="gpt-4o-mini")  # Cheaper
response_llm = ChatOpenAI(model="gpt-4o")  # Better quality for final response

# 2. Cache embedding results
from functools import lru_cache

@lru_cache(maxsize=10000)
def cached_embed(text: str) -> tuple:
    """Cache embeddings to avoid redundant API calls."""
    return tuple(embeddings.embed_query(text))

# 3. Limit iteration count
agent = create_react_agent(
    model=llm,
    tools=tools,
    max_iterations=5  # Prevent runaway costs
)
```

---

## Installation Requirements

```bash
# Core dependencies
pip install langchain langchain-openai langchain-community langgraph

# Vector database
pip install qdrant-client langchain-qdrant

# Local models (optional)
pip install langchain-ollama

# For production persistence
pip install langgraph-checkpoint-sqlite  # Single server
pip install langgraph-checkpoint-postgres  # Multi-server

# Observability
pip install langsmith

# Retry logic
pip install tenacity

# Development utilities
pip install python-dotenv ipython
```

**Version Compatibility Note**: This guide was tested with:
- `langchain>=0.3.0`
- `langgraph>=0.2.0`
- `qdrant-client>=1.9.0`

---

## Quick Start Template

```python
"""
Agentic RAG Quick Start Template
Copy and customize for your use case.

Requirements:
    pip install langchain langchain-openai langgraph qdrant-client langchain-qdrant
"""

from typing import Annotated
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "knowledge_base"
SYSTEM_PROMPT = """You are a helpful assistant with access to a knowledge base.
Search the knowledge base when you need factual information.
Always cite sources in your responses."""

# =============================================================================
# 2. VECTOR DATABASE SETUP
# =============================================================================

def setup_vectorstore() -> QdrantVectorStore:
    """Initialize Qdrant vector store."""
    client = QdrantClient(":memory:")  # Use url="http://localhost:6333" for production

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

# =============================================================================
# 3. TOOL DEFINITION
# =============================================================================

vectorstore = setup_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

@tool
def search_knowledge_base(
    query: Annotated[str, "Detailed search query to find relevant information"]
) -> str:
    """
    Search the knowledge base for relevant information.
    Use when you need factual data from documents.
    Returns formatted results with source citations.
    """
    results = retriever.invoke(query)

    if not results:
        return "No relevant documents found."

    formatted = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        formatted.append(f"[{i}] {doc.page_content[:500]}\n   Source: {source}")

    return "\n\n".join(formatted)

# =============================================================================
# 4. AGENT CREATION
# =============================================================================

def create_agent():
    """Create production-ready agentic RAG agent."""
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    memory = MemorySaver()  # Use SqliteSaver/PostgresSaver for production

    return create_react_agent(
        model=llm,
        tools=[search_knowledge_base],
        state_modifier=SYSTEM_PROMPT,
        checkpointer=memory,
    )

# =============================================================================
# 5. USAGE
# =============================================================================

if __name__ == "__main__":
    agent = create_agent()
    config = {"configurable": {"thread_id": "session-1"}}

    # Example query
    response = agent.invoke(
        {"messages": [("user", "What information do you have?")]},
        config=config
    )

    print(response["messages"][-1].content)
```

---

## 14. Quick Reference

### Cheat Sheet

| Task | Code |
|------|------|
| **Create basic agent** | `agent = create_react_agent(llm, tools)` |
| **Add system prompt** | `create_react_agent(..., state_modifier="prompt")` |
| **Enable memory** | `create_react_agent(..., checkpointer=MemorySaver())` |
| **Set iteration limit** | `create_react_agent(..., max_iterations=10)` |
| **Stream responses** | `agent.stream(input, stream_mode="updates")` |
| **Use thread** | `agent.invoke(input, {"configurable": {"thread_id": "x"}})` |

### Common Recipes

**Recipe: Quick Agentic RAG**
```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore

# 1. Setup retriever
retriever = QdrantVectorStore(...).as_retriever(search_kwargs={"k": 5})

# 2. Create RAG tool
@tool
def search(query: str) -> str:
    """Search knowledge base."""
    return "\n".join(d.page_content for d in retriever.invoke(query))

# 3. Create agent
agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), [search])
```

**Recipe: Agent with Memory**
```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_react_agent(llm, tools, checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "session-1"}}
agent.invoke({"messages": [("user", "Hi")]}, config)
```

### Production Checklist

**Pre-Launch**
- [ ] Iteration limits configured (`max_iterations`)
- [ ] Error handling with retries implemented
- [ ] Rate limiting middleware added
- [ ] Content guardrails in place
- [ ] Logging and tracing enabled (LangSmith)

**Launch**
- [ ] Persistent checkpointer configured (not MemorySaver)
- [ ] Monitoring dashboards set up
- [ ] Alerting on error rates configured
- [ ] Cost tracking enabled

**Post-Launch**
- [ ] Trace analysis for optimization opportunities
- [ ] User feedback loop established
- [ ] Model/prompt iteration based on real usage

---

## Performance Comparison

| Approach | Speed | Memory | Quality | Cost | Complexity |
|----------|-------|--------|---------|------|------------|
| **`create_react_agent`** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | $$ | ⭐ Easy |
| **Custom `StateGraph`** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $$ | ⭐⭐⭐ Moderate |
| **Traditional RAG Chain** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | $ | ⭐ Easy |
| **Agentic RAG** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $$$ | ⭐⭐⭐ Moderate |
| **Ollama (Local)** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | Free | ⭐⭐⭐ Moderate |
| **OpenAI (Cloud)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $$$$ | ⭐ Easy |

---

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Tools Guide](https://python.langchain.com/docs/modules/tools/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Ollama Model Library](https://ollama.com/library)
- [LangSmith Tracing](https://docs.smith.langchain.com/)

---