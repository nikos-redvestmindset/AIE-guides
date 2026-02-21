# Production-Grade LangGraph Agents with Dependency Injection

## Overview

This document describes a production-grade architecture for building LangGraph agents using dependency injection and the builder pattern. The architecture enables testable, composable agents with explicit dependencies while maintaining flexibility to use either `create_agent()` or custom `StateGraph` implementations. All agents extend `AgentBuilder` — implement `_build()` (protected subclass hook), consumers call `compile()` (public, cached, non-overridable).

> **Deprecation:** `abstract_agent.BaseAgent` (legacy custom LLM framework) is deprecated. Use `AgentBuilder` instead.

## Core Architecture

### Design Principles

1. **Explicit Dependencies**: All agent dependencies injected through constructors
2. **Builder Pattern**: Agents extend `AgentBuilder` ABC — implement `_build()`, consumers call `compile()`
3. **Compile Output**: `compile()` returns `CompiledStateGraph` (full LangChain Runnable API)
4. **Non-overridable compile()**: `compile()` is `@final` with `__init_subclass__` runtime guard
5. **Implementation Flexibility**: Freedom to use `create_agent()` or custom `StateGraph` inside `_build()`
6. **Subgraph Isolation**: Multi-stage agents use the LangGraph "invoke from a node" pattern — each sub-agent keeps its own focused state
7. **Input/Output Schema Separation**: Parent graphs use `StateGraph(State, input=Input, output=Output)` to hide internal bridge fields from callers
8. **Type Safety**: Full typing with TypedDict, Pydantic, and type hints
9. **Testability**: Easy mocking through dependency injection
10. **Stateless Agents**: Per-request isolation, no shared state between requests
11. **External Resource Management**: LLMs, clients, and stores managed by container

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│              AgentBuilder ABC (agents/base.py)                │
│         _build() -> CompiledStateGraph  [subclass hook]      │
│         compile() -> CompiledStateGraph [@final, cached]     │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┬─────────────────┐
         ▼                               ▼                 ▼
┌──────────────────┐        ┌──────────────────┐   ┌──────────────────┐
│ Simple Agents    │        │ Graph Agents     │   │ Multi-Agent      │
│ (create_agent)   │        │ (StateGraph +    │   │ (Supervisor)     │
│                  │        │  subgraphs)      │   │                  │
└────────┬─────────┘        └────────┬─────────┘   └────────┬─────────┘
         │                           │                       │
         └───────────────────────────┴───────────────────────┘
                                     │
                         ┌───────────▼──────────┐
                         │ DI Container         │
                         │ Wires Dependencies   │
                         └──────────────────────┘
```

## Implementation

### 1. Agent Builder Base Class

All agent builders extend `AgentBuilder`, an abstract base class with a sealed public API:

- **`compile()`** — Public entry point. Returns a cached `CompiledStateGraph`. Marked `@final` (static check) and guarded by `__init_subclass__` (runtime check) so subclasses cannot override it.
- **`_build()`** — Protected subclass hook. Constructs and returns the `CompiledStateGraph`. Each agent implements this.
- **`reset()`** — Clears the cached compilation (useful in tests or reconfiguration).

```python
# agents/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, final

from langgraph.graph.state import CompiledStateGraph


class AgentBuilder(ABC):
    """Abstract base for all LangGraph agent builders.

    Subclasses implement ``_build()`` to construct a
    ``CompiledStateGraph``.  Consumers call ``compile()`` which
    caches the result.

    ``compile()`` cannot be overridden:

    * Static: ``@typing.final`` — Pyright / Mypy flag violations.
    * Runtime: ``__init_subclass__`` — ``TypeError`` if a subclass
      defines ``compile`` in its own ``__dict__``.
    """

    _compiled: CompiledStateGraph | None = None

    # ------------------------------------------------------------------
    # Subclass guard — prevent compile() override at class-creation time
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "compile" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override compile(). "
                "Implement _build() instead."
            )

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    @abstractmethod
    def _build(self) -> CompiledStateGraph:
        """Subclass hook: construct and return the agent graph.

        Not called directly by consumers — they call ``compile()``.
        """
        ...

    # ------------------------------------------------------------------
    # Public API (sealed)
    # ------------------------------------------------------------------

    @final
    def compile(self) -> CompiledStateGraph:
        """Compile (and cache) the agent graph.

        Returns a ``CompiledStateGraph`` which implements the full
        LangChain ``Runnable`` API (``invoke``, ``ainvoke``,
        ``stream``, ``astream``, ``batch``, etc.).

        The result is cached — repeated calls return the same
        instance.  Call ``reset()`` to clear the cache.
        """
        if self._compiled is None:
            self._compiled = self._build()
        return self._compiled

    def reset(self) -> None:
        """Clear the cached compilation (for testing / reconfiguration)."""
        self._compiled = None
```

**Why this design:**

| Concern | Mechanism |
|---------|-----------|
| Consumers get a consistent API | `compile()` always returns `CompiledStateGraph` |
| Subclasses can't accidentally break caching | `compile()` is non-overridable |
| Static tooling catches mistakes | `@final` decorator |
| Runtime catches mistakes too | `__init_subclass__` raises `TypeError` |
| Subclasses focus on graph construction only | `_build()` is the single extension point |

### 2. State Definitions

#### Per-stage focused states

Each sub-agent defines its **own** focused TypedDict. Do not create a monolithic "uber-state" shared across unrelated stages — it couples sub-agents and makes them harder to test independently.

```python
# agents/viz_designer/schemas.py (excerpted)
from typing import Annotated, Any
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class VizSelectionState(TypedDict):
    """State for the chart-selection sub-agent only."""
    messages: Annotated[list[AnyMessage], add_messages]
    nlp_query: str
    output_schema: dict[str, Any]
    materialized_data: list[dict[str, Any]]
    structured_response: VizSelectionResponseSchema | None


class VizRefinementState(TypedDict):
    """State for the refinement sub-agent only."""
    messages: Annotated[list[AnyMessage], add_messages]
    selection_result: VizSelectionResponseSchema | None
    nlp_query: str
    output_schema: dict[str, Any]
    materialized_data: list[dict[str, Any]]
    conversation_id: str | None
    structured_response: VizRefinementResponseSchema | None
```

#### Input / Output / Graph State pattern (recommended for multi-stage agents)

For orchestrating parent graphs that wire multiple sub-agents, use the `StateGraph(State, input=Input, output=Output)` pattern (as seen in [open_deep_research](https://github.com/langchain-ai/open_deep_research)):

```python
class VizDesignerInput(TypedDict):
    """What the caller provides when invoking the pipeline."""
    messages: Annotated[list[AnyMessage], add_messages]
    nlp_query: str
    output_schema: dict[str, Any]
    materialized_data: list[dict[str, Any]]
    conversation_id: str | None


class VizDesignerOutput(TypedDict):
    """What the caller receives back from the pipeline."""
    agent_response: AgentChartResponse | None


class VizDesignerGraphState(TypedDict):
    """Internal orchestration state — includes bridge fields hidden from callers.

    Bridge fields (e.g. ``selection_result``) carry intermediate data
    between pipeline stages.  They never appear in the input or output
    schemas.
    """
    # ---- From input (provided by the caller) ----
    messages: Annotated[list[AnyMessage], add_messages]
    nlp_query: str
    output_schema: dict[str, Any]
    materialized_data: list[dict[str, Any]]
    conversation_id: str | None
    # ---- Bridge (internal, hidden from caller) ----
    selection_result: VizSelectionResponseSchema | None
    # ---- Output ----
    agent_response: AgentChartResponse | None
```

**Why three TypedDicts?**

| TypedDict | Purpose |
|-----------|---------|
| `Input` | Narrow contract for callers — only what they must provide |
| `Output` | Narrow contract for callers — only what they receive back |
| `GraphState` | Internal superset — includes bridge fields for inter-node data flow |

This pattern hides internal wiring (bridge fields like `selection_result`) from the public API while giving nodes full access to intermediate state.

### 3. Simple Agent Implementation

Agent using `create_agent()` for a single-stage tool-calling pattern. Even simple agents extend `AgentBuilder` and implement `_build()`:

```python
# agents/chart_selector/agent.py
from __future__ import annotations

from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from agents.base import AgentBuilder
from agents.models import AgentContext
from agents.chart_selector.config import ChartSelectorSettings
from agents.chart_selector.schemas import ChartSelectionState


class ChartSelectionAgent(AgentBuilder):
    """Chart selection agent using simple tool-calling pattern.

    This agent uses ``create_agent()`` for straightforward tool-calling.
    Suitable when:
    - Standard ReAct loop is sufficient
    - No complex control flow needed
    - Tools are independent operations

    Extends ``AgentBuilder`` — consumers call ``compile()`` (not ``_build()``).
    """

    def __init__(
        self,
        settings: ChartSelectorSettings,
        tools: list | None = None,
    ):
        super().__init__()
        self.settings = settings
        self.tools = tools or []

    def _build(self) -> CompiledStateGraph:
        """Build the agent using create_agent().

        Returns:
            CompiledStateGraph ready for invocation.
        """
        return create_agent(
            model=self.settings.model,
            tools=self.tools,
            system_prompt=self.settings.system_prompt,
            state_schema=ChartSelectionState,
            response_format=self.settings.response_format,
            context_schema=AgentContext,
            debug=self.settings.debug,
            name="chart_selection",
        )



# Usage: consumers call compile(), never _build()
selector = ChartSelectionAgent(settings=settings, tools=tools)
pipeline = selector.compile()      # cached CompiledStateGraph
result = await pipeline.ainvoke(
    {"messages": [], "nlp_query": "Show revenue over time", ...},
    context=agent_context,
)
```

### 4. Complex Agent with StateGraph — "Invoke from a Node" Pattern

For multi-stage agents, each sub-agent should have its **own focused state**. The parent graph orchestrates sub-agents using the LangGraph "invoke from a node" pattern: each node transforms parent state → subgraph state, invokes the subgraph, and transforms the result back into a parent state update.

This is the actual pattern used by `VisualizationDesigner` (see `agents/src/agents/viz_designer/agent.py`):

```python
# agents/viz_designer/agent.py (simplified — see actual implementation for full code)
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.base import AgentBuilder
from agents.models import AgentContext
from agents.viz_designer.config import VisualizationDesignerSettings
from agents.viz_designer.schemas import (
    VizDesignerGraphState,
    VizDesignerInput,
    VizDesignerOutput,
    VizRefinementState,
    VizSelectionState,
    # ... response schemas ...
)

if TYPE_CHECKING:
    from xlake.api import XLakeClient


class VisualizationDesigner(AgentBuilder):
    """2-stage visualization pipeline: Selection -> Refinement.

    Uses the LangGraph "invoke from a node" subgraph pattern so each
    sub-agent operates on its own focused state while the parent graph
    uses a thin orchestration state with ``input=``/``output=`` schema
    separation to hide internal bridge fields.

    Pipeline::

        START -> selection -> transform -> refinement -> END
    """

    def __init__(
        self,
        settings: VisualizationDesignerSettings,
        xlake_client: XLakeClient | None = None,
    ):
        super().__init__()
        self.settings = settings
        self._xlake_client = xlake_client
        self._tools = self._build_tools()

    # ------------------------------------------------------------------
    # AgentBuilder implementation
    # ------------------------------------------------------------------

    def _build(self) -> CompiledStateGraph:
        """Build the 2-stage pipeline with input/output schema separation.

        Uses ``StateGraph(State, input=Input, output=Output)`` pattern
        from ``open_deep_research`` to hide bridge fields from callers.
        """
        graph = StateGraph(
            VizDesignerGraphState,          # internal (includes bridges)
            input=VizDesignerInput,         # what the caller provides
            output=VizDesignerOutput,       # what the caller receives
            context_schema=AgentContext,
        )

        graph.add_node("selection", self._run_selection)
        graph.add_node("transform", self._selection_to_refinement)
        graph.add_node("refinement", self._run_refinement)

        graph.add_edge(START, "selection")
        graph.add_edge("selection", "transform")
        graph.add_edge("transform", "refinement")
        graph.add_edge("refinement", END)

        return graph.compile(debug=self.settings.debug)

    # ------------------------------------------------------------------
    # Graph node functions — "invoke from a node" pattern
    # ------------------------------------------------------------------

    async def _run_selection(
        self,
        state: VizDesignerGraphState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """Node: invoke the selection sub-agent.

        Transform parent state -> VizSelectionState,
        invoke sub-agent, transform result -> parent state update.
        Config propagates AgentContext automatically.
        """
        agent = self._create_selection_stage()

        # Parent state -> selection subgraph state
        result = await agent.ainvoke(
            {
                "messages": state["messages"],
                "nlp_query": state["nlp_query"],
                "output_schema": state["output_schema"],
                "materialized_data": state["materialized_data"],
            },
            config=config,
        )

        # Selection output -> parent state update (bridge field)
        return {"selection_result": result["structured_response"]}

    def _selection_to_refinement(
        self,
        state: VizDesignerGraphState,
    ) -> dict[str, Any]:
        """Node: pure transform — format selection output for refinement.

        This is NOT a sub-agent invocation. It reads the bridge field
        and creates a HumanMessage for the refinement LLM.
        """
        selection_result = state["selection_result"]
        formatted_request = format_refinement_request(
            selection_result=selection_result,
            data_schema=state["output_schema"],
            materialized_data=state["materialized_data"],
            nlp_query=state["nlp_query"],
        )
        return {"messages": [HumanMessage(content=formatted_request)]}

    async def _run_refinement(
        self,
        state: VizDesignerGraphState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """Node: invoke refinement sub-agent and build final response.

        Combines invocation + response assembly in a single node so
        ``refinement_result`` remains a local variable (never touches
        the parent state).
        """
        agent = self._create_refinement_stage()

        result = await agent.ainvoke(
            {
                "messages": state["messages"],
                "selection_result": state["selection_result"],
                "nlp_query": state["nlp_query"],
                "output_schema": state["output_schema"],
                "materialized_data": state["materialized_data"],
                "conversation_id": state.get("conversation_id"),
            },
            config=config,
        )

        # Build final AgentChartResponse from refinement output
        agent_response = self._assemble_chart_response(
            refinement_result=result["structured_response"],
            materialized_data=state.get("materialized_data", []),
            output_schema=state.get("output_schema", {}),
            conversation_id=state.get("conversation_id"),
        )
        return {"agent_response": agent_response}

    # ------------------------------------------------------------------
    # Sub-agent factories (not graph nodes)
    # ------------------------------------------------------------------

    def _create_selection_stage(self) -> CompiledStateGraph:
        """Create the chart-selection sub-agent."""
        stage = self.settings.get_stage_settings("selection")
        return create_agent(
            model=stage.model,
            tools=self._tools,
            system_prompt=stage.prompt or SELECTION_SYSTEM_PROMPT,
            state_schema=VizSelectionState,      # focused state
            response_format=VizSelectionResponseSchema,
            context_schema=AgentContext,
            debug=self.settings.debug,
            name="viz_selection",
        )

    def _create_refinement_stage(self) -> CompiledStateGraph:
        """Create the refinement sub-agent."""
        stage = self.settings.get_stage_settings("refinement")
        return create_agent(
            model=stage.model,
            tools=self._tools,
            system_prompt=stage.prompt or REFINEMENT_SYSTEM_PROMPT,
            state_schema=VizRefinementState,     # focused state
            response_format=VizRefinementResponseSchema,
            context_schema=AgentContext,
            debug=self.settings.debug,
            name="viz_refinement",
        )
```

**Key patterns demonstrated:**

1. **`input=` / `output=` schema separation** — Callers only see `VizDesignerInput` and `VizDesignerOutput`; bridge fields like `selection_result` are hidden inside `VizDesignerGraphState`.

2. **"Invoke from a node"** — Each node transforms parent state into a sub-agent's focused state, invokes the sub-agent, and maps the result back to a parent state update. Sub-agents have completely different schemas from the parent.

3. **Config propagation** — `config: RunnableConfig` is forwarded from node to sub-agent, so `AgentContext` (via `context_schema`) propagates automatically without custom wrappers.

4. **Sub-agent factories** — `_create_selection_stage()` and `_create_refinement_stage()` are regular methods (not graph nodes) that create sub-agent instances using `create_agent()` with their own focused states.

5. **Pure transform nodes** — `_selection_to_refinement` is a data-transformation node (no LLM call) that bridges the output of one sub-agent into the input format of the next.

6. **Response assembly as a private method** — `_assemble_chart_response()` is called from the refinement node, not a separate graph node. This keeps `refinement_result` as a local variable and avoids polluting the parent state.

### 5. Multi-Agent System

Coordinator agent managing multiple specialized agents:

```python
# agents/viz_coordinator.py
from typing import Literal, List
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, START, END

from .base import AgentBuilder
from .state import VisualizationState
from .visualization_designer import VisualizationDesigner
from .chart_selector import ChartSelectionAgent


class VizCoordinator(AgentBuilder):
    """
    Supervisor agent coordinating multiple visualization specialists.
    
    This agent implements the supervisor pattern:
    - Routes tasks to specialized agents
    - Aggregates results
    - Handles fallbacks
    
    Use when:
    - Multiple specialized agents needed
    - Dynamic routing based on task type
    - Need to aggregate multiple agent outputs
    """
    
    def __init__(
        self,
        supervisor_llm: ChatOpenAI,
        designer_agent: VisualizationDesigner,
        selector_agent: ChartSelectionAgent,
        verbose: bool = False,
    ):
        """
        Initialize coordinator with sub-agents.
        
        Args:
            supervisor_llm: LLM for routing decisions
            designer_agent: Full pipeline agent for complex requests
            selector_agent: Fast agent for simple chart selection
            verbose: Enable detailed logging
        """
        self.supervisor_llm = supervisor_llm
        self.designer_agent = designer_agent
        self.selector_agent = selector_agent
        self.verbose = verbose
        
        # Compile sub-agents
        self.designer_runnable = designer_agent.compile()
        self.selector_runnable = selector_agent.compile()
        
        self._graph: Optional[Runnable] = None
    
    def _build(self) -> CompiledStateGraph:
        """
        Build the supervisor graph with routing logic.
        
        Returns:
            CompiledStateGraph: Compiled graph that routes to appropriate agent
        """
        workflow = StateGraph(VisualizationState)
        
        # Add nodes
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("designer", self._call_designer)
        workflow.add_node("selector", self._call_selector)
        workflow.add_node("aggregate", self._aggregate_results)
        
        # Routing
        workflow.add_edge(START, "supervisor")
        workflow.add_conditional_edges(
            "supervisor",
            self._route_task,
            {
                "designer": "designer",
                "selector": "selector",
            }
        )
        workflow.add_edge("designer", "aggregate")
        workflow.add_edge("selector", "aggregate")
        workflow.add_edge("aggregate", END)
        
        return workflow.compile()
    
    def _supervisor_node(self, state: VisualizationState) -> dict:
        """Supervisor analyzes task and decides routing"""
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        # Determine task complexity
        complexity_prompt = f"""Analyze this visualization request:
{last_message}

Is this a simple chart selection or complex visualization design?
Respond with: SIMPLE or COMPLEX"""
        
        response = self.supervisor_llm.invoke([("human", complexity_prompt)])
        
        task_type = "simple" if "SIMPLE" in response.content.upper() else "complex"
        
        return {"metadata": {"task_type": task_type}}
    
    def _route_task(
        self,
        state: VisualizationState
    ) -> Literal["designer", "selector"]:
        """Route to appropriate agent based on task type"""
        task_type = state.get("metadata", {}).get("task_type", "simple")
        
        if task_type == "complex":
            if self.verbose:
                print("Routing to designer agent")
            return "designer"
        else:
            if self.verbose:
                print("Routing to selector agent")
            return "selector"
    
    def _call_designer(self, state: VisualizationState) -> dict:
        """Call full designer agent"""
        result = self.designer_runnable.invoke(state)
        return result
    
    def _call_selector(self, state: VisualizationState) -> dict:
        """Call simple selector agent"""
        result = self.selector_runnable.invoke(state)
        return result
    
    def _aggregate_results(self, state: VisualizationState) -> dict:
        """Aggregate and format final results"""
        # Add reasoning about routing decision
        task_type = state.get("metadata", {}).get("task_type", "unknown")
        reasoning = f"Routed to {task_type} agent based on task complexity analysis"
        
        return {"reasoning": reasoning}
```

### 6. Dependency Injection Container

Wire all agents through DI container:

```python
# containers.py
from dependency_injector import containers, providers
from agents.chart_selector import ChartSelectionAgent
from agents.visualization_designer import VisualizationDesigner
from agents.viz_coordinator import VizCoordinator
from agents.config import get_agent_settings
from langchain_openai import ChatOpenAI


class AgentsContainer(containers.DeclarativeContainer):
    """
    Container for all agent dependencies.
    
    This container manages:
    - Configuration settings
    - Shared resources (LLMs, vector stores)
    - Tool creation
    - Agent builders
    """
    
    # Configuration
    config = providers.Configuration()
    agent_settings = providers.Singleton(get_agent_settings)
    
    # Shared Resources (Singletons)
    
    # Main LLM for complex reasoning
    main_llm = providers.Singleton(
        ChatOpenAI,
        model=agent_settings.provided.main_model,
        temperature=agent_settings.provided.temperature,
        api_key=agent_settings.provided.openai_api_key,
        max_retries=3,
    )
    
    # Fast LLM for simple tasks
    fast_llm = providers.Singleton(
        ChatOpenAI,
        model=agent_settings.provided.fast_model,
        temperature=0.0,
        api_key=agent_settings.provided.openai_api_key,
        max_retries=3,
    )
    
    # Vector store for RAG
    vector_store = providers.Singleton(
        create_vector_store,
        embedding_model=agent_settings.provided.embedding_model,
    )
    
    # Tools (Factories)
    
    search_viz_rules = providers.Factory(
        create_search_viz_rules_tool,
        vector_store=vector_store,
    )
    
    analyze_schema = providers.Factory(
        create_analyze_schema_tool,
    )
    
    validate_chart = providers.Factory(
        create_validate_chart_tool,
    )
    
    # Create tool lists for different agents
    selector_tools = providers.List(
        search_viz_rules,
        analyze_schema,
    )
    
    designer_tools = providers.List(
        search_viz_rules,
        analyze_schema,
        validate_chart,
    )
    
    # Agents (Factories - new instance per request)
    
    chart_selection_agent = providers.Factory(
        ChartSelectionAgent,
        llm=fast_llm,  # Use fast LLM for simple selection
        tools=selector_tools,
        max_iterations=agent_settings.provided.max_iterations,
        verbose=agent_settings.provided.verbose,
    )
    
    visualization_designer = providers.Factory(
        VisualizationDesigner,
        llm=main_llm,  # Use main LLM for complex design
        chart_selector=chart_selection_agent,
        tools=designer_tools,
        max_refinement_iterations=3,
        verbose=agent_settings.provided.verbose,
    )
    
    viz_coordinator = providers.Factory(
        VizCoordinator,
        supervisor_llm=fast_llm,  # Fast routing decisions
        designer_agent=visualization_designer,
        selector_agent=chart_selection_agent,
        verbose=agent_settings.provided.verbose,
    )


# Alternative: Minimal container for testing
class TestAgentsContainer(containers.DeclarativeContainer):
    """Simplified container for testing with mocks"""
    
    # Mock LLM
    llm = providers.Singleton(
        lambda: MockChatOpenAI(model="gpt-4"),
    )
    
    # Mock tools
    tools = providers.List()
    
    # Agent with mocked dependencies
    chart_selection_agent = providers.Factory(
        ChartSelectionAgent,
        llm=llm,
        tools=tools,
        verbose=True,
    )
```

### 7. FastAPI Integration

Integrate agents into FastAPI with proper lifecycle management:

```python
# api/routes/visualizations.py
from fastapi import APIRouter, Depends, HTTPException
from dependency_injector.wiring import inject, Provide
from contextlib import asynccontextmanager

from containers import AgentsContainer
from agents.viz_coordinator import VizCoordinator
from agents.state import VisualizationState


router = APIRouter(prefix="/api/visualizations", tags=["visualizations"])


@router.post("/create")
@inject
async def create_visualization(
    request: VisualizationRequest,
    coordinator: VizCoordinator = Depends(Provide[AgentsContainer.viz_coordinator]),
):
    """
    Create visualization using coordinator agent.
    
    The coordinator will:
    1. Analyze request complexity
    2. Route to appropriate agent
    3. Return chart configuration
    """
    try:
        # Build the agent graph
        agent_graph = coordinator.compile()
        
        # Prepare initial state
        initial_state: VisualizationState = {
            "messages": [("human", request.query)],
            "data_schema": request.data_schema,
            "sample_data": request.sample_data,
            "reasoning": None,
        }
        
        # Invoke agent
        result = await agent_graph.ainvoke(initial_state)
        
        # Extract chart config
        chart_config = result.get("chart_config")
        
        if not chart_config:
            raise HTTPException(
                status_code=500,
                detail="Agent failed to generate chart configuration"
            )
        
        return {
            "chart_config": chart_config,
            "reasoning": result.get("reasoning"),
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Visualization creation failed: {str(e)}"
        )


@router.post("/select-chart")
@inject
async def select_chart(
    request: ChartSelectionRequest,
    selector: ChartSelectionAgent = Depends(
        Provide[AgentsContainer.chart_selection_agent]
    ),
):
    """
    Simple chart selection endpoint using selector agent directly.
    
    Bypasses coordinator for known simple tasks.
    """
    agent = selector.compile()
    
    initial_state: VisualizationState = {
        "messages": [("human", request.query)],
        "data_schema": request.data_schema,
        "sample_data": request.sample_data,
        "reasoning": None,
    }
    
    result = await agent.ainvoke(initial_state)
    
    return {
        "chart_type": result.get("chart_type"),
        "chart_config": result.get("chart_config"),
        "reasoning": result.get("reasoning"),
    }
```

### 8. Application Setup

Wire everything together in main application:

```python
# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dependency_injector.wiring import Provide

from containers import AgentsContainer


# Create container
container = AgentsContainer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    
    Handles:
    - Container wiring to modules
    - Shared resource initialization (if needed)
    """
    # Wire container to modules
    container.wire(
        modules=[
            "api.routes.visualizations",
            "api.routes.charts",
        ]
    )
    
    yield
    
    # Cleanup on shutdown
    container.unwire()


# Create FastAPI app
app = FastAPI(
    title="ActBI Visualization API",
    lifespan=lifespan,
)

# Attach container to app
app.container = container

# Include routers
from api.routes import visualizations, charts
app.include_router(visualizations.router)
app.include_router(charts.router)
```

### 9. Testing Patterns

Comprehensive testing with mocked dependencies:

```python
# tests/test_agents.py
import pytest
from unittest.mock import AsyncMock, Mock, patch
from dependency_injector import containers, providers

from agents.chart_selector import ChartSelectionAgent
from agents.state import VisualizationState


class MockChatOpenAI:
    """Mock LLM for testing"""
    
    def __init__(self, model: str):
        self.model = model
    
    def invoke(self, messages):
        """Return mock response"""
        return Mock(content='{"chart_type": "bar", "config": {}}')


class TestChartSelectionAgent:
    """Test suite for chart selection agent"""
    
    @pytest.fixture
    def mock_llm(self):
        """Provide mock LLM"""
        return MockChatOpenAI(model="gpt-4")
    
    @pytest.fixture
    def mock_tools(self):
        """Provide mock tools"""
        return []
    
    @pytest.fixture
    def agent(self, mock_llm, mock_tools):
        """Provide agent with mocked dependencies"""
        return ChartSelectionAgent(
            llm=mock_llm,
            tools=mock_tools,
            verbose=True,
        )
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent.llm is not None
        assert agent.tools == []
        assert agent.max_iterations == 10
    
    def test_compile_returns_runnable(self, agent):
        """Test compile() returns a CompiledStateGraph (Runnable)"""
        runnable = agent.compile()
        assert runnable is not None
        assert hasattr(runnable, 'invoke')
        assert hasattr(runnable, 'ainvoke')
    
    @pytest.mark.asyncio
    async def test_agent_invocation(self, agent):
        """Test agent can be invoked"""
        runnable = agent.compile()
        
        initial_state: VisualizationState = {
            "messages": [("human", "Create a bar chart")],
            "data_schema": {"x": "string", "y": "number"},
            "sample_data": [{"x": "A", "y": 10}],
        }
        
        # Mock the actual agent execution
        with patch.object(runnable, 'ainvoke', new=AsyncMock(
            return_value={"chart_type": "bar"}
        )):
            result = await runnable.ainvoke(initial_state)
            assert result["chart_type"] == "bar"


class TestVisualizationDesigner:
    """Test suite for visualization designer"""
    
    @pytest.fixture
    def designer(self):
        """Create designer with test settings"""
        settings = VisualizationDesignerSettings(
            default_model="gpt-4o-mini",
            debug=True,
        )
        return VisualizationDesigner(settings=settings)
    
    def test_compile_returns_graph(self, designer):
        """Test compile() returns a CompiledStateGraph"""
        graph = designer.compile()
        assert graph is not None
        assert hasattr(graph, 'ainvoke')
    
    def test_compile_is_cached(self, designer):
        """Test compile() returns same instance on repeated calls"""
        graph1 = designer.compile()
        graph2 = designer.compile()
        assert graph1 is graph2


class TestAgentBuilderGuard:
    """Tests for AgentBuilder compile() protection."""

    def test_subclass_cannot_override_compile(self):
        """Subclassing AgentBuilder with a compile() method raises TypeError."""
        with pytest.raises(TypeError, match="must not override compile"):
            class BadAgent(AgentBuilder):
                def _build(self) -> CompiledStateGraph:
                    ...
                def compile(self) -> CompiledStateGraph:  # type: ignore[override]
                    ...


# Integration tests
@pytest.mark.asyncio
async def test_end_to_end_visualization_flow():
    """
    Integration test for complete visualization flow.
    
    Tests:
    - Container wiring
    - Agent composition
    - State propagation
    - Error handling
    """
    # Create test container
    from tests.containers import TestApplicationContainer
    container = TestApplicationContainer()
    
    # Get coordinator
    coordinator = container.viz_coordinator()
    
    # Compile graph
    graph = coordinator.compile()
    
    # Test invocation
    result = await graph.ainvoke({
        "messages": [("human", "Create a bar chart")],
        "data_schema": {"x": "string", "y": "number"},
        "sample_data": [{"x": "A", "y": 10}],
        "reasoning": None,
    })
    
    # Verify result
    assert result.get("chart_config") is not None
    assert result.get("reasoning") is not None
```

### 10. Composite Configuration Pattern

For complex agents, use Pydantic Settings to encapsulate all configuration in a single object. This provides type safety, validation, and a clean agent interface.

**Convention**: All agent settings use the `AGENT_<AGENT_NAME>_` prefix for environment variables to maintain consistency across the codebase (e.g., `AGENT_VIZ_DESIGNER_`, `AGENT_CHART_SELECTOR_`).

#### Pattern: Agent-Specific Settings

Each agent gets its own settings class that includes all configuration:

```python
# agents/viz_designer/config.py
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional


class StageSettings(BaseModel):
    """Configuration for a single pipeline stage."""
    
    model: Optional[str] = None  # None = use pipeline default
    temperature: float = 0.0
    max_tokens: int = 4000
    max_iterations: int = 5
    
    class Config:
        # Allow arbitrary types for prompts, tools
        arbitrary_types_allowed = True


class VisualizationDesignerSettings(BaseSettings):
    """
    Complete configuration for VisualizationDesigner agent.
    
    This encapsulates ALL configuration for the agent:
    - Models and LLM parameters
    - Pipeline behavior
    - Stage-specific overrides
    - Debug settings
    
    Loaded from AGENT_VIZ_DESIGNER_* environment variables.
    
    Example .env:
        AGENT_VIZ_DESIGNER_DEFAULT_MODEL=anthropic:claude-sonnet-4-20250514
        AGENT_VIZ_DESIGNER_DEBUG=true
        AGENT_VIZ_DESIGNER_SELECTION_MODEL=gpt-4
    """
    
    model_config = SettingsConfigDict(
        env_prefix="AGENT_VIZ_DESIGNER_",
        env_file=".env",
        case_sensitive=False,
    )
    
    # Pipeline-wide defaults
    default_model: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 4000
    debug: bool = False
    
    # Stage-specific overrides
    selection_model: Optional[str] = None
    selection_temperature: Optional[float] = None
    selection_max_iterations: int = 5
    
    refinement_model: Optional[str] = None
    refinement_temperature: Optional[float] = None
    refinement_max_iterations: int = 3
    
    styling_model: Optional[str] = None
    styling_temperature: Optional[float] = None
    
    implementation_model: Optional[str] = None
    implementation_temperature: Optional[float] = None
    
    def get_stage_settings(self, stage: str) -> StageSettings:
        """
        Get settings for a specific stage, with fallback to defaults.
        
        Args:
            stage: One of "selection", "refinement", "styling", "implementation"
            
        Returns:
            StageSettings with stage-specific overrides applied
        """
        # Get stage-specific values or fall back to defaults
        model = getattr(self, f"{stage}_model") or self.default_model
        temperature = getattr(self, f"{stage}_temperature")
        if temperature is None:
            temperature = self.temperature
        
        max_iterations = getattr(self, f"{stage}_max_iterations", 5)
        
        return StageSettings(
            model=model,
            temperature=temperature,
            max_tokens=self.max_tokens,
            max_iterations=max_iterations,
        )


@lru_cache
def get_viz_designer_settings() -> VisualizationDesignerSettings:
    """
    Get cached visualization designer settings.
    
    Called once per process. All configuration loaded from environment.
    """
    return VisualizationDesignerSettings()


# Example: Another agent's settings following the same pattern
class ChartSelectorSettings(BaseSettings):
    """
    Configuration for ChartSelectionAgent.
    
    Loaded from AGENT_CHART_SELECTOR_* environment variables.
    
    Example .env:
        AGENT_CHART_SELECTOR_MODEL=gpt-4o-mini
        AGENT_CHART_SELECTOR_TEMPERATURE=0.0
        AGENT_CHART_SELECTOR_MAX_ITERATIONS=5
    """
    
    model_config = SettingsConfigDict(
        env_prefix="AGENT_CHART_SELECTOR_",
        env_file=".env",
        case_sensitive=False,
    )
    
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_iterations: int = 5
    debug: bool = False


@lru_cache
def get_chart_selector_settings() -> ChartSelectorSettings:
    """Get cached chart selector settings."""
    return ChartSelectorSettings()
```

#### Agent Implementation with Composite Config

The agent takes only the settings object as its parameter:

```python
# agents/viz_designer/agent.py
from typing import Optional
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, START, END

from agents.base import AgentBuilder
from agents.viz_designer.config import VisualizationDesignerSettings
from agents.viz_designer.state import VisualizationState


class VisualizationDesigner(AgentBuilder):
    """
    Visualization designer with multi-stage pipeline.
    
    Configuration is fully encapsulated in VisualizationDesignerSettings.
    All dependencies (LLMs, tools) are created internally based on settings.
    
    Directory structure:
        agents/viz_designer/
            __init__.py
            agent.py         # This file - the builder
            config.py        # VisualizationDesignerSettings
            state.py         # VisualizationState
            prompts.py       # Prompt templates
            nodes.py         # Node functions (optional)
    """
    
    def __init__(
        self,
        settings: VisualizationDesignerSettings,
    ):
        """
        Initialize designer with complete configuration.
        
        Args:
            settings: Complete configuration for the agent including
                     models, temperatures, stage overrides, etc.
        """
        super().__init__()
        self.settings = settings
        
        # Internal state
        self._refinement_count = 0
    
    def _build(self) -> CompiledStateGraph:
        """
        Build the multi-stage visualization pipeline using StateGraph.
        
        Returns:
            CompiledStateGraph: Compiled graph that processes VisualizationState
        """
        workflow = StateGraph(VisualizationState)
        
        # Add nodes for each stage
        workflow.add_node("select_chart", self._select_chart_node)
        workflow.add_node("refine_chart", self._refine_chart_node)
        workflow.add_node("style_chart", self._style_chart_node)
        workflow.add_node("implement_chart", self._implement_chart_node)
        
        # Add edges
        workflow.add_edge(START, "select_chart")
        workflow.add_edge("select_chart", "refine_chart")
        workflow.add_conditional_edges(
            "refine_chart",
            self._should_continue_refining,
            {
                "continue": "refine_chart",
                "style": "style_chart",
            }
        )
        workflow.add_edge("style_chart", "implement_chart")
        workflow.add_edge("implement_chart", END)
        
        # Compile with debug mode from settings
        return workflow.compile(debug=self.settings.debug)
    
    def _select_chart_node(self, state: VisualizationState) -> dict:
        """Node: Select initial chart type"""
        from langchain.chat_models import init_chat_model
        
        # Get stage-specific settings
        stage_settings = self.settings.get_stage_settings("selection")
        
        # Initialize LLM for this stage
        llm = init_chat_model(
            model=stage_settings.model,
            temperature=stage_settings.temperature,
            max_tokens=stage_settings.max_tokens,
        )
        
        # Selection logic using stage-specific LLM
        # ...
        
        return {
            "chart_type": "bar",
            "reasoning": f"Selected bar chart using {stage_settings.model}",
        }
    
    def _refine_chart_node(self, state: VisualizationState) -> dict:
        """Node: Refine chart configuration"""
        from langchain.chat_models import init_chat_model
        
        # Get stage-specific settings
        stage_settings = self.settings.get_stage_settings("refinement")
        
        # Initialize LLM for this stage
        llm = init_chat_model(
            model=stage_settings.model,
            temperature=stage_settings.temperature,
            max_tokens=stage_settings.max_tokens,
        )
        
        self._refinement_count += 1
        
        # Refinement logic
        # ...
        
        return {
            "chart_config": {"refined": True},
            "reasoning": f"Refined using {stage_settings.model}",
        }
    
    def _style_chart_node(self, state: VisualizationState) -> dict:
        """Node: Apply styling"""
        stage_settings = self.settings.get_stage_settings("styling")
        # Styling logic
        return {"chart_config": state["chart_config"]}
    
    def _implement_chart_node(self, state: VisualizationState) -> dict:
        """Node: Generate implementation code"""
        stage_settings = self.settings.get_stage_settings("implementation")
        # Implementation logic
        return {"chart_config": state["chart_config"]}
    
    def _should_continue_refining(self, state: VisualizationState) -> str:
        """Conditional: Check if more refinement needed"""
        stage_settings = self.settings.get_stage_settings("refinement")
        
        if self._refinement_count >= stage_settings.max_iterations:
            return "style"
        
        # Check if refinement needed
        if state.get("needs_refinement"):
            return "continue"
        
        return "style"
```

#### Container Integration

The container only needs to provide the settings object:

```python
# containers.py
from dependency_injector import containers, providers
from agents.viz_designer.agent import VisualizationDesigner
from agents.viz_designer.config import get_viz_designer_settings


class AgentsContainer(containers.DeclarativeContainer):
    """
    Container for agent dependencies.
    
    With composite settings pattern, the container is simplified:
    - Provide settings objects (Singletons)
    - Agents create their own internal dependencies
    """
    
    # Settings objects (Singletons - loaded once from env)
    viz_designer_settings = providers.Singleton(get_viz_designer_settings)
    
    # Agent takes only settings (Factory - new instance per request)
    viz_designer = providers.Factory(
        VisualizationDesigner,
        settings=viz_designer_settings,
    )


# Usage in FastAPI
@router.post("/visualizations/create")
@inject
async def create_visualization(
    request: VisualizationRequest,
    designer: VisualizationDesigner = Depends(
        Provide[AgentsContainer.viz_designer]
    ),
):
    """Create visualization using designer agent"""
    graph = designer.compile()
    
    result = await graph.ainvoke({
        "messages": [("human", request.query)],
        "data_schema": request.data_schema,
        "sample_data": request.sample_data,
        "reasoning": None,
    })
    
    return {
        "chart_config": result["chart_config"],
        "reasoning": result["reasoning"],
    }
```

#### Alternative: Shared Dependencies with Settings

If you need to inject external dependencies (like clients) alongside settings:

```python
# agents/viz_designer/agent.py
from xlake.client import XLakeClient


class VisualizationDesigner(AgentBuilder):
    """
    Designer with both settings and external dependencies.
    """
    
    def __init__(
        self,
        settings: VisualizationDesignerSettings,
        xlake_client: XLakeClient,  # External dependency
    ):
        """
        Initialize with settings and external dependencies.
        
        Args:
            settings: Complete agent configuration
            xlake_client: External XLake client (managed by container)
        """
        super().__init__()
        self.settings = settings
        self.xlake_client = xlake_client
    
    def _build(self) -> CompiledStateGraph:
        """Build graph using both settings and external dependencies"""
        # Agent can use self.settings for LLM config
        # and self.xlake_client for external operations
        # ...


# Container
class AgentsContainer(containers.DeclarativeContainer):
    # Shared external dependencies
    xlake_client = providers.Singleton(
        XLakeClient,
        base_url=config.xlake_base_url,
        api_key=config.xlake_api_key,
    )
    
    # Settings
    viz_designer_settings = providers.Singleton(get_viz_designer_settings)
    
    # Agent gets both
    viz_designer = providers.Factory(
        VisualizationDesigner,
        settings=viz_designer_settings,
        xlake_client=xlake_client,
    )
```

#### Benefits of Composite Settings Pattern

1. **Single Source of Truth**: All agent config in one place
2. **Type Safety**: Pydantic validates all settings
3. **Environment-Based**: Easy to override via env vars
4. **Clean Agent Interface**: Agent takes one parameter
5. **Stage Flexibility**: Per-stage overrides without complex wiring
6. **Testability**: Easy to create test settings
7. **Documentation**: Settings class documents all options

#### Directory Structure Example

```
agents/
├── viz_designer/
│   ├── __init__.py              # Exports VisualizationDesigner
│   ├── agent.py                 # VisualizationDesigner class (builder)
│   ├── config.py                # VisualizationDesignerSettings
│   ├── state.py                 # VisualizationState
│   ├── prompts.py               # Prompt templates
│   └── nodes.py                 # Optional: node functions
│
├── chart_selector/
│   ├── __init__.py              # Exports ChartSelectionAgent
│   ├── agent.py                 # ChartSelectionAgent class
│   ├── config.py                # ChartSelectorSettings
│   └── state.py                 # State definition
│
└── base.py                      # AgentBuilder protocol
```

#### Testing with Composite Settings

```python
# tests/test_viz_designer.py
import pytest
from agents.viz_designer.agent import VisualizationDesigner
from agents.viz_designer.config import VisualizationDesignerSettings


class TestVisualizationDesigner:
    """Test suite for visualization designer"""
    
    @pytest.fixture
    def test_settings(self):
        """Provide test settings without env vars"""
        return VisualizationDesignerSettings(
            default_model="gpt-4",
            selection_model="gpt-4o-mini",  # Fast model for selection
            debug=True,
            selection_max_iterations=2,
            refinement_max_iterations=1,
        )
    
    @pytest.fixture
    def designer(self, test_settings):
        """Provide designer with test settings"""
        return VisualizationDesigner(settings=test_settings)
    
    def test_designer_initialization(self, designer, test_settings):
        """Test designer initializes with settings"""
        assert designer.settings == test_settings
        assert designer.settings.debug is True
    
    def test_stage_settings(self, test_settings):
        """Test stage-specific settings resolution"""
        selection = test_settings.get_stage_settings("selection")
        assert selection.model == "gpt-4o-mini"  # Override
        assert selection.max_iterations == 2
        
        refinement = test_settings.get_stage_settings("refinement")
        assert refinement.model == "gpt-4"  # Falls back to default
        assert refinement.max_iterations == 1
    
    @pytest.mark.asyncio
    async def test_designer_invocation(self, designer):
        """Test designer can be invoked"""
        graph = designer.compile()
        
        result = await graph.ainvoke({
            "messages": [("human", "Create a bar chart")],
            "data_schema": {"x": "string", "y": "number"},
            "sample_data": [{"x": "A", "y": 10}],
            "reasoning": None,
        })
        
        assert result["chart_type"] is not None
        assert result["reasoning"] is not None
```

### 11. Tool Builder Pattern

Tools often have external dependencies (APIs, clients, vector stores) and configuration. Use the builder pattern for tools just like agents, with settings encapsulation and dependency injection.

**Convention**: All tool settings use the `TOOL_<TOOL_NAME>_` prefix for environment variables (e.g., `TOOL_VIZ_SEARCH_`, `TOOL_WEATHER_API_`).

#### Tool Builder Base Class

Define an abstract base class that all tool builders must extend (follows the
same pattern as ``AgentBuilder``):

```python
# tools/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, final

from langchain_core.tools import BaseTool


class ToolBuilder(ABC):
    """Base builder for all LangChain tools.

    Subclasses implement ``_build()`` to construct the tool.
    Consumers call ``compile()`` to get a standard ``BaseTool``.

    The ``_build`` method (single-underscore prefix) is *protected by
    convention* — it should only be called by the framework, never
    directly by application code.

    The ``compile`` method is **non-overridable**: it is decorated with
    ``@typing.final`` (for static type-checkers) and enforced at
    runtime via ``__init_subclass__``.  If a subclass attempts to
    define ``compile``, a ``TypeError`` is raised at class-definition
    time.

    Tools are stateless - dependencies (clients, APIs, caches) are
    injected via constructor and managed externally by the container.
    """

    def __init__(self) -> None:
        self._compiled: BaseTool | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "compile" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override compile(). "
                "Implement _build() instead."
            )

    @abstractmethod
    def _build(self) -> BaseTool:
        """Construct and return the tool.

        Subclasses must implement this method.  It is called once by
        ``compile()`` and the result is cached for subsequent calls.

        Returns:
            A ``BaseTool`` ready for use with agents.
        """
        ...

    @final
    def compile(self) -> BaseTool:
        """Compile and return the tool (cached).

        This is the **public entry-point** for obtaining the tool.
        It calls ``_build()`` on the first invocation and caches
        the result for subsequent calls.

        Returns:
            A ``BaseTool`` ready for use with agents.
        """
        if self._compiled is None:
            self._compiled = self._build()
        return self._compiled

    def reset(self) -> None:
        """Clear the cached compilation."""
        self._compiled = None
```

#### Pattern: Tool-Specific Settings

Each tool gets its own settings class and directory:

```python
# tools/viz_search/config.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional


class VizSearchToolSettings(BaseSettings):
    """
    Configuration for visualization design rules search tool.
    
    This tool searches a vector store for relevant visualization
    design guidelines based on data characteristics and user intent.
    
    Loaded from TOOL_VIZ_SEARCH_* environment variables.
    
    Example .env:
        TOOL_VIZ_SEARCH_COLLECTION_NAME=viz_design_rules
        TOOL_VIZ_SEARCH_TOP_K=5
        TOOL_VIZ_SEARCH_SCORE_THRESHOLD=0.7
        TOOL_VIZ_SEARCH_RERANK=true
    """
    
    model_config = SettingsConfigDict(
        env_prefix="TOOL_VIZ_SEARCH_",
        env_file=".env",
        case_sensitive=False,
    )
    
    # Vector store configuration
    collection_name: str = "viz_design_rules"
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Search behavior
    rerank: bool = True
    max_retries: int = 3
    timeout: float = 10.0
    
    # Embedding configuration
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    
    # Tool metadata
    tool_name: str = "search_viz_design_rules"
    tool_description: str = """Search visualization design rules and best practices.
    
Use this tool when you need to:
- Find the best chart type for specific data characteristics
- Learn about visualization design principles
- Understand when to use different chart types
- Get guidance on data encoding, axes, colors, etc.

Args:
    query: Natural language description of what you're looking for
    
Returns:
    Relevant design rules and guidelines with source citations
"""


@lru_cache
def get_viz_search_tool_settings() -> VizSearchToolSettings:
    """Get cached viz search tool settings."""
    return VizSearchToolSettings()
```

#### Tool Implementation with Composite Config

The tool builder takes settings and external dependencies:

```python
# tools/viz_search/tool.py
from typing import Optional
from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores import VectorStore

from tools.base import ToolBuilder
from tools.viz_search.config import VizSearchToolSettings


class VizSearchTool(ToolBuilder):
    """
    Visualization design rules search tool.
    
    Searches a vector store for relevant visualization guidelines
    based on natural language queries.
    
    Directory structure:
        tools/viz_search/
            __init__.py
            tool.py         # This file - the builder
            config.py       # VizSearchToolSettings
    """
    
    def __init__(
        self,
        settings: VizSearchToolSettings,
        vector_store: VectorStore,  # External dependency
    ):
        """
        Initialize search tool with settings and dependencies.
        
        Args:
            settings: Complete tool configuration
            vector_store: Vector store instance (managed by container)
        """
        super().__init__()  # Initialize base class (sets up caching)
        self.settings = settings
        self.vector_store = vector_store
    
    def _build(self) -> BaseTool:
        """
        Build the search tool.
        
        Implements the ToolBuilder._build() hook. Called by compile()
        and the result is cached automatically.
        
        Returns:
            BaseTool: Configured search tool ready to bind to agents
        """
        # Create the tool using @tool decorator with closure
        settings = self.settings
        vector_store = self.vector_store
        
        @tool(name=settings.tool_name)
        def search_viz_design_rules(query: str) -> str:
            # Tool description from settings
            f"""{settings.tool_description}"""
            
            try:
                # Search vector store with settings-based parameters
                results = vector_store.similarity_search_with_score(
                    query,
                    k=settings.top_k,
                )
                
                # Filter by score threshold
                filtered_results = [
                    (doc, score) 
                    for doc, score in results 
                    if score >= settings.score_threshold
                ]
                
                if not filtered_results:
                    return f"No design rules found matching '{query}' (threshold: {settings.score_threshold})"
                
                # Format results
                formatted = []
                for i, (doc, score) in enumerate(filtered_results, 1):
                    source = doc.metadata.get("source", "Unknown")
                    formatted.append(
                        f"[Result {i} - Score: {score:.2f} - Source: {source}]\n"
                        f"{doc.page_content}"
                    )
                
                return "\n\n---\n\n".join(formatted)
                
            except Exception as e:
                return f"Search failed: {str(e)}"
        
        # Override the docstring with settings value
        search_viz_design_rules.__doc__ = settings.tool_description
        
        return search_viz_design_rules


# Example: API-based tool with retry logic
class WeatherAPITool(ToolBuilder):
    """
    Weather API tool with retry and caching.
    """
    
    def __init__(
        self,
        settings: WeatherAPIToolSettings,
        http_client: httpx.AsyncClient,  # External dependency
    ):
        super().__init__()  # Initialize base class (sets up caching)
        self.settings = settings
        self.http_client = http_client
    
    def _build(self) -> BaseTool:
        """Build weather API tool.
        
        Implements the ToolBuilder._build() hook. Called by compile()
        and the result is cached automatically.
        """
        settings = self.settings
        http_client = self.http_client
            
            @tool(name=settings.tool_name)
            async def get_weather(location: str) -> str:
                """Get current weather for a location.
                
                Use this when the user asks about weather conditions,
                temperature, or forecast for a specific location.
                
                Args:
                    location: City name or "City, Country" format
                    
                Returns:
                    Current weather information
                """
                url = f"{settings.api_base_url}/current"
                params = {
                    "location": location,
                    "apikey": settings.api_key,
                    "units": settings.units,
                }
                
                for attempt in range(settings.max_retries):
                    try:
                        response = await http_client.get(
                            url,
                            params=params,
                            timeout=settings.timeout,
                        )
                        response.raise_for_status()
                        data = response.json()
                        
                        return (
                            f"Weather in {location}:\n"
                            f"Temperature: {data['temp']}°{settings.units.upper()}\n"
                            f"Conditions: {data['conditions']}\n"
                            f"Humidity: {data['humidity']}%"
                        )
                    
                    except httpx.HTTPError as e:
                        if attempt == settings.max_retries - 1:
                            return f"Weather API error: {str(e)}"
                        continue
        
        return get_weather
```

#### Container Integration for Tools

Wire tools through the container alongside agents:

```python
# containers.py
from dependency_injector import containers, providers

from tools.viz_search.tool import VizSearchTool
from tools.viz_search.config import get_viz_search_tool_settings
from tools.weather_api.tool import WeatherAPITool
from tools.weather_api.config import get_weather_api_tool_settings


class ToolsContainer(containers.DeclarativeContainer):
    """
    Container for tool dependencies.
    
    Tools are built through ToolBuilder pattern with their own
    settings and external dependencies.
    """
    
    # External dependencies (Singletons - shared resources)
    vector_store = providers.Singleton(
        create_vector_store,
        embedding_model="text-embedding-3-small",
    )
    
    http_client = providers.Singleton(
        httpx.AsyncClient,
        timeout=30.0,
        headers={"User-Agent": "ActBI/1.0"},
    )
    
    # Tool settings (Singletons - loaded once)
    viz_search_settings = providers.Singleton(get_viz_search_tool_settings)
    weather_api_settings = providers.Singleton(get_weather_api_tool_settings)
    
    # Tool builders (Factories - new instance per request if needed)
    viz_search_tool_builder = providers.Factory(
        VizSearchTool,
        settings=viz_search_settings,
        vector_store=vector_store,
    )
    
    weather_api_tool_builder = providers.Factory(
        WeatherAPITool,
        settings=weather_api_settings,
        http_client=http_client,
    )
    
    # Built tools (Singletons - tools are stateless, can be reused)
    viz_search_tool = providers.Singleton(
        lambda builder: builder.compile(),
        builder=viz_search_tool_builder,
    )
    
    weather_api_tool = providers.Singleton(
        lambda builder: builder.compile(),
        builder=weather_api_tool_builder,
    )


class AgentsContainer(containers.DeclarativeContainer):
    """
    Container for agents that use tools.
    """
    
    # Import tools container
    tools = providers.Container(ToolsContainer)
    
    # Agent settings
    viz_designer_settings = providers.Singleton(get_viz_designer_settings)
    
    # Create tool lists for agents
    viz_designer_tools = providers.List(
        tools.viz_search_tool,
        # other tools...
    )
    
    # Agent with tools
    viz_designer = providers.Factory(
        VisualizationDesigner,
        settings=viz_designer_settings,
        tools=viz_designer_tools,  # Inject list of built tools
    )
```

#### Alternative: Agent Creates Tools Internally

For simpler cases, agents can create tools using builders:

```python
# agents/viz_designer/agent.py
from tools.viz_search.tool import VizSearchTool


class VisualizationDesigner(AgentBuilder):
    """
    Designer that creates its own tools from builders.
    """
    
    def __init__(
        self,
        settings: VisualizationDesignerSettings,
        viz_search_tool_builder: VizSearchTool,  # Inject builder
    ):
        self.settings = settings
        self.viz_search_tool_builder = viz_search_tool_builder
        self._graph = None
    
    def _build(self) -> CompiledStateGraph:
        """Build graph and create tools"""
        # Compile the tool (cached after first call)
        viz_search_tool = self.viz_search_tool_builder.compile()
        
        # Use in agent
        tools = [viz_search_tool]
        
        # Create graph with tools...
        # ...


# Container
class AgentsContainer(containers.DeclarativeContainer):
    tools = providers.Container(ToolsContainer)
    
    viz_designer = providers.Factory(
        VisualizationDesigner,
        settings=viz_designer_settings,
        viz_search_tool_builder=tools.viz_search_tool_builder,  # Inject builder
    )
```

#### Testing Tools

Test tools with mocked dependencies:

```python
# tests/test_tools/test_viz_search.py
import pytest
from unittest.mock import Mock

from tools.viz_search.tool import VizSearchTool
from tools.viz_search.config import VizSearchToolSettings


class TestVizSearchTool:
    """Test suite for viz search tool"""
    
    @pytest.fixture
    def test_settings(self):
        """Provide test settings"""
        return VizSearchToolSettings(
            collection_name="test_collection",
            top_k=3,
            score_threshold=0.5,
            debug=True,
        )
    
    @pytest.fixture
    def mock_vector_store(self):
        """Provide mock vector store"""
        mock = Mock()
        mock.similarity_search_with_score.return_value = [
            (Mock(page_content="Rule 1", metadata={"source": "guide.md"}), 0.9),
            (Mock(page_content="Rule 2", metadata={"source": "guide.md"}), 0.8),
        ]
        return mock
    
    @pytest.fixture
    def tool_builder(self, test_settings, mock_vector_store):
        """Provide tool builder with mocked dependencies"""
        return VizSearchTool(
            settings=test_settings,
            vector_store=mock_vector_store,
        )
    
    def test_tool_initialization(self, tool_builder, test_settings):
        """Test tool initializes with settings"""
        assert tool_builder.settings == test_settings
        assert tool_builder.vector_store is not None
    
    def test_compile_returns_tool(self, tool_builder):
        """Test compile() returns a BaseTool"""
        tool = tool_builder.compile()
        assert tool is not None
        assert hasattr(tool, 'invoke')
        assert tool.name == "search_viz_design_rules"
    
    def test_compile_caches_result(self, tool_builder):
        """Test compile() caches the result"""
        tool1 = tool_builder.compile()
        tool2 = tool_builder.compile()
        assert tool1 is tool2
    
    def test_reset_clears_cache(self, tool_builder):
        """Test reset() clears the cached compilation"""
        tool1 = tool_builder.compile()
        tool_builder.reset()
        tool2 = tool_builder.compile()
        assert tool1 is not tool2
    
    def test_tool_invocation(self, tool_builder, mock_vector_store):
        """Test tool can be invoked"""
        tool = tool_builder.compile()
        
        result = tool.invoke({"query": "bar chart guidelines"})
        
        assert "Rule 1" in result
        assert "Rule 2" in result
        mock_vector_store.similarity_search_with_score.assert_called_once()
    
    def test_tool_score_filtering(self, tool_builder, mock_vector_store):
        """Test tool filters by score threshold"""
        # Return one result below threshold
        mock_vector_store.similarity_search_with_score.return_value = [
            (Mock(page_content="High score", metadata={}), 0.8),
            (Mock(page_content="Low score", metadata={}), 0.3),
        ]
        
        tool = tool_builder.compile()
        result = tool.invoke({"query": "test"})
        
        assert "High score" in result
        assert "Low score" not in result
```

#### Tool Settings Examples

Additional tool settings examples:

```python
# tools/weather_api/config.py
class WeatherAPIToolSettings(BaseSettings):
    """
    Configuration for weather API tool.
    
    Example .env:
        TOOL_WEATHER_API_BASE_URL=https://api.weather.com/v1
        TOOL_WEATHER_API_KEY=your-api-key
        TOOL_WEATHER_API_UNITS=metric
    """
    
    model_config = SettingsConfigDict(
        env_prefix="TOOL_WEATHER_API_",
        env_file=".env",
        case_sensitive=False,
    )
    
    api_base_url: str
    api_key: str
    units: str = "metric"  # or "imperial"
    timeout: float = 10.0
    max_retries: int = 3
    
    tool_name: str = "get_weather"
    tool_description: str = "Get current weather for a location"


# tools/database_query/config.py
class DatabaseQueryToolSettings(BaseSettings):
    """
    Configuration for database query tool.
    
    Example .env:
        TOOL_DATABASE_QUERY_MAX_ROWS=1000
        TOOL_DATABASE_QUERY_TIMEOUT=30.0
        TOOL_DATABASE_QUERY_ALLOWED_TABLES=users,products,orders
    """
    
    model_config = SettingsConfigDict(
        env_prefix="TOOL_DATABASE_QUERY_",
        env_file=".env",
        case_sensitive=False,
    )
    
    max_rows: int = Field(default=1000, ge=1, le=10000)
    timeout: float = 30.0
    allowed_tables: list[str] = Field(default_factory=list)
    read_only: bool = True
    
    tool_name: str = "query_database"
```

#### Directory Structure for Tools

```
tools/
├── __init__.py
├── base.py                      # ToolBuilder protocol
│
├── viz_search/                  # Vector search tool
│   ├── __init__.py             # Exports VizSearchTool
│   ├── tool.py                 # VizSearchTool builder
│   └── config.py               # VizSearchToolSettings
│
├── weather_api/                 # API-based tool
│   ├── __init__.py
│   ├── tool.py                 # WeatherAPITool builder
│   └── config.py               # WeatherAPIToolSettings
│
├── database_query/              # Database tool
│   ├── __init__.py
│   ├── tool.py                 # DatabaseQueryTool builder
│   └── config.py               # DatabaseQueryToolSettings
│
└── web_search/                  # Web search tool
    ├── __init__.py
    ├── tool.py                 # WebSearchTool builder
    └── config.py               # WebSearchToolSettings
```

#### Benefits of Tool Builder Pattern

1. **Consistent Architecture**: Same pattern as agents
2. **Explicit Dependencies**: All deps injected via constructor
3. **Type Safety**: Pydantic validates tool configuration
4. **Testability**: Easy to mock external dependencies
5. **Reusability**: Tools can be shared across agents
6. **Configuration**: Per-tool settings with env var override
7. **Stateless**: Tools are stateless, safe to reuse

### 12. Configuration Management (Legacy Pattern)

*Note: The composite settings pattern above is recommended for complex agents. This section shows the simpler pattern for shared configuration across multiple agents.*

Structured configuration with Pydantic Settings:

```python
# agents/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache


class AgentSettings(BaseSettings):
    """
    Shared agent configuration from AGENT_* environment variables.
    
    Use this pattern when multiple agents share the same configuration.
    For agent-specific config, use the composite pattern instead.
    """
    
    # Model configuration
    main_model: str = "gpt-4-turbo-preview"
    fast_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    
    # API keys
    openai_api_key: str
    anthropic_api_key: str = ""
    
    # Agent behavior
    temperature: float = 0.0
    max_iterations: int = 10
    max_refinement_iterations: int = 3
    
    # Performance
    request_timeout: float = 30.0
    max_retries: int = 3
    
    # Logging
    verbose: bool = False
    log_level: str = "INFO"
    
    # Memory
    enable_checkpointing: bool = False
    checkpoint_dir: str = "./checkpoints"
    
    class Config:
        env_prefix = "AGENT_"
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_agent_settings() -> AgentSettings:
    """
    Get cached agent settings.
    
    Called once per process, ensures single source of truth.
    """
    return AgentSettings()
```

### 13. Streaming Support

Enable streaming responses from agents:

```python
# agents/streaming.py
from typing import AsyncIterator
from langchain_core.messages import BaseMessage


class StreamingAgent(AgentBuilder):
    """
    Agent with streaming support for real-time responses.
    
    Useful for:
    - Long-running agent tasks
    - Progressive UI updates
    - Real-time user feedback
    """
    
    def build_streaming(self) -> Runnable:
        """
        Build agent with streaming enabled.
        
        Returns:
            Runnable: Agent that supports .astream()
        """
        # Compile agent with streaming config
        agent = self.compile()
        return agent.with_config({"recursion_limit": 100})
    
    async def stream_response(
        self,
        state: VisualizationState
    ) -> AsyncIterator[dict]:
        """
        Stream agent responses as they're generated.
        
        Yields:
            dict: Incremental state updates
        """
        agent = self.build_streaming()
        
        async for chunk in agent.astream(state):
            # Yield each intermediate state
            yield {
                "node": chunk.get("node"),
                "state": chunk.get("state"),
                "timestamp": chunk.get("timestamp"),
            }


# FastAPI endpoint with streaming
@router.post("/visualizations/stream")
@inject
async def stream_visualization(
    request: VisualizationRequest,
    agent: StreamingAgent = Depends(Provide[AgentsContainer.streaming_agent]),
):
    """Stream visualization creation progress"""
    
    from fastapi.responses import StreamingResponse
    import json
    
    async def generate():
        async for chunk in agent.stream_response(request.to_state()):
            yield f"data: {json.dumps(chunk)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )
```

## Advanced Patterns

### Pattern: Agent Composition

Compose multiple agents into pipelines:

```python
# agents/pipeline.py
from langchain_core.runnables import RunnableSequence


class AgentPipeline(AgentBuilder):
    """
    Sequential pipeline of multiple agents.
    
    Each agent processes the output of the previous one.
    """
    
    def __init__(self, agents: List[AgentBuilder]):
        """
        Initialize pipeline with ordered agents.
        
        Args:
            agents: List of agent builders to chain
        """
        self.agents = agents
        self._pipeline: Optional[Runnable] = None
    
    def _build(self) -> CompiledStateGraph:
        """Build sequential pipeline of agents.
        
        Note: For simple sequential composition, ``RunnableSequence``
        works. For complex multi-stage pipelines, prefer the
        "invoke from a node" pattern (see Section 4).
        """
        # Compile all sub-agents
        runnables = [agent.compile() for agent in self.agents]
        
        # Compose into sequence
        return RunnableSequence(*runnables)


# Usage in container
class AgentsContainer(containers.DeclarativeContainer):
    # Individual agents
    selector = providers.Factory(ChartSelectionAgent, ...)
    validator = providers.Factory(ChartValidator, ...)
    formatter = providers.Factory(ChartFormatter, ...)
    
    # Pipeline combining them
    viz_pipeline = providers.Factory(
        AgentPipeline,
        agents=providers.List(
            selector,
            validator,
            formatter,
        )
    )
```

### Pattern: Memory Integration

Add persistent memory to agents:

```python
# agents/memory.py
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore


class MemoryAgent(AgentBuilder):
    """
    Agent with persistent memory across sessions.
    
    Uses LangGraph's checkpointing and store for:
    - Conversation history
    - User preferences
    - Previous decisions
    
    Note: Checkpointer and store are injected as dependencies
    and their lifecycle is managed externally (typically at
    the application level).
    """
    
    def __init__(
        self,
        llm: ChatOpenAI,
        tools: list,
        checkpointer: PostgresSaver,
        store: PostgresStore,
    ):
        """
        Initialize agent with memory dependencies.
        
        Args:
            llm: Language model for reasoning
            tools: Agent tools
            checkpointer: Persistent checkpointer for conversation state
            store: Persistent store for long-term memory
        """
        self.llm = llm
        self.tools = tools
        self.checkpointer = checkpointer
        self.store = store
    
    def _build(self) -> CompiledStateGraph:
        """Build agent with memory"""
        # Create graph
        workflow = StateGraph(VisualizationState)
        # ... add nodes and edges ...
        
        # Compile with memory
        return workflow.compile(
            checkpointer=self.checkpointer,
            store=self.store,
        )
```

### Pattern: Dynamic Tool Loading

Load tools dynamically based on context:

```python
# agents/dynamic_tools.py
from typing import Callable, List
from langchain_core.tools import BaseTool


class DynamicToolAgent(AgentBuilder):
    """
    Agent with dynamic tool loading based on context.
    
    Tools are loaded based on:
    - User permissions
    - Data availability
    - Task requirements
    """
    
    def __init__(
        self,
        llm: ChatOpenAI,
        tool_registry: dict[str, Callable[[], BaseTool]],
        default_tools: List[str],
    ):
        self.llm = llm
        self.tool_registry = tool_registry
        self.default_tools = default_tools
    
    def get_tools_for_context(self, context: dict) -> List[BaseTool]:
        """
        Load appropriate tools based on context.
        
        Args:
            context: User context, permissions, data sources
            
        Returns:
            List of tools available for this context
        """
        available_tools = []
        
        # Always add default tools
        for tool_name in self.default_tools:
            if tool_name in self.tool_registry:
                tool = self.tool_registry[tool_name]()
                available_tools.append(tool)
        
        # Add contextual tools
        if context.get("has_database_access"):
            db_tool = self.tool_registry["query_database"]()
            available_tools.append(db_tool)
        
        if context.get("has_api_access"):
            api_tool = self.tool_registry["call_api"]()
            available_tools.append(api_tool)
        
        return available_tools
    
    def _build(self, context: dict = None) -> CompiledStateGraph:
        """Build agent with context-appropriate tools"""
        context = context or {}
        tools = self.get_tools_for_context(context)
        
        return create_agent(
            model=self.llm,
            tools=tools,
            prompt=self.prompt,
        )
```

## Best Practices

### 1. Always Call `compile()`, Never `_build()`

```python
# Good — consumers call compile()
designer = VisualizationDesigner(settings=settings)
graph = designer.compile()      # cached, full Runnable API
result = await graph.ainvoke(input, context=ctx)

# Bad — calling _build() directly (bypasses caching, breaks contract)
graph = designer._build()       # DON'T DO THIS
```

### 2. Use Factories for Per-Request Agents

```python
# Good - Factory for per-request agents
viz_designer = providers.Factory(VisualizationDesigner, ...)

# Bad - Singleton for stateful agent (state leaks between requests)
viz_designer = providers.Singleton(VisualizationDesigner, ...)
```

### 3. Use Composite Settings for Complex Agents

```python
# Good - single settings object encapsulates all config
class VisualizationDesigner(AgentBuilder):
    def __init__(self, settings: VisualizationDesignerSettings):
        super().__init__()
        self.settings = settings

# Bad - too many parameters, hard to maintain
class VisualizationDesigner(AgentBuilder):
    def __init__(
        self,
        model: str,
        temperature: float,
        max_iterations: int,
        selection_model: str,
        refinement_model: str,
        # ... 20 more parameters
    ):
        ...
```

### 4. Use Per-Stage Focused States (Not a Monolithic State)

```python
# Good — each sub-agent has its own focused state
class VizSelectionState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    nlp_query: str
    output_schema: dict[str, Any]
    materialized_data: list[dict[str, Any]]
    structured_response: VizSelectionResponseSchema | None

class VizRefinementState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    selection_result: VizSelectionResponseSchema | None
    nlp_query: str
    # ...

# Bad — one massive state shared across all sub-agents
class MonolithicState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    nlp_query: str
    selection_result: ...
    refinement_result: ...
    chart_response: ...
    # every field from every stage ...
```

### 5. Use Input/Output Schema Separation for Orchestrators

```python
# Good — callers see narrow Input/Output; bridge fields are hidden
graph = StateGraph(
    VizDesignerGraphState,      # internal (includes bridge fields)
    input=VizDesignerInput,     # what caller provides
    output=VizDesignerOutput,   # what caller receives
)

# Bad — exposing all internal state to callers
graph = StateGraph(VizDesignerGraphState)   # leaks bridge fields
```

### 6. Use the "Invoke from a Node" Pattern for Subgraphs

```python
# Good — node transforms parent state -> subgraph state, invokes, maps back
async def _run_selection(self, state: GraphState, config: RunnableConfig) -> dict:
    agent = self._create_selection_stage()
    result = await agent.ainvoke(
        {"messages": state["messages"], "nlp_query": state["nlp_query"], ...},
        config=config,   # propagates AgentContext automatically
    )
    return {"selection_result": result["structured_response"]}

# Bad — sharing the same state object between parent and subgraph
async def _run_selection(self, state: GraphState, config: RunnableConfig) -> dict:
    agent = self._create_selection_stage()
    return await agent.ainvoke(state, config=config)  # state mismatch!
```

### 7. Type All State Classes

```python
# Good - fully typed state
class VizSelectionState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    nlp_query: str
    output_schema: dict[str, Any]
    structured_response: VizSelectionResponseSchema | None

# Bad - untyped state
class VizSelectionState(TypedDict):
    messages: list
    nlp_query: Any
```

### 8. Document Agent Capabilities

```python
class VisualizationDesigner(AgentBuilder):
    """2-stage visualization pipeline: Selection -> Refinement.

    Uses the LangGraph "invoke from a node" subgraph pattern so each
    sub-agent operates on its own focused state while the parent graph
    uses a thin orchestration state with ``input=``/``output=`` schema
    separation to hide internal bridge fields.

    Capabilities:
    - Selects optimal chart type via tool-augmented reasoning
    - Classifies data series (prominence, purpose, sentiment)
    - Generates highlights linked to insights
    - Supports combo charts with sub-chart specs

    Limitations:
    - 2-stage only (no iterative refinement loop yet)
    - Does not persist results (upstream API layer responsibility)

    Dependencies:
    - VisualizationDesignerSettings (composite config)
    - XLakeClient (optional, for viz design rules search)
    """
```

### 9. Test at Multiple Levels

```python
# Unit test — test the compile() guard
def test_subclass_cannot_override_compile():
    with pytest.raises(TypeError, match="must not override compile"):
        class BadAgent(AgentBuilder):
            def _build(self): ...
            def compile(self): ...

# Unit test — test compile returns graph
def test_compile_returns_graph():
    designer = VisualizationDesigner(settings=test_settings)
    graph = designer.compile()
    assert isinstance(graph, CompiledStateGraph)

# Integration test — test node transforms
def test_selection_to_refinement():
    designer = VisualizationDesigner(settings=test_settings)
    result = designer._selection_to_refinement(mock_state)
    assert "messages" in result

# E2E test — test through API
async def test_agent_endpoint():
    response = await client.post("/api/visualizations/create", json=request)
    assert response.status_code == 200
```

## Project Structure

```
project/
├── agents/
│   ├── __init__.py
│   ├── base.py                     # AgentBuilder ABC (_build/compile)
│   ├── abstract_agent.py           # DEPRECATED: Legacy BaseAgent — use AgentBuilder instead
│   ├── lib/
│   │   └── utils.py                # create_agent_context() and other utilities
│   ├── models.py                   # AgentContext dataclass
│   │
│   ├── viz_designer/               # Complex agent with composite config
│   │   ├── __init__.py             # Exports VisualizationDesigner
│   │   ├── agent.py                # VisualizationDesigner (_build)
│   │   ├── config.py               # VisualizationDesignerSettings
│   │   ├── schemas.py              # Input/Output/GraphState + sub-agent states
│   │   ├── prompts.py              # Prompt templates
│   │   └── README.md               # Agent documentation
│   │
│   ├── chart_selector/             # Simple agent with composite config
│   │   ├── __init__.py             # Exports ChartSelectionAgent
│   │   ├── agent.py                # ChartSelectionAgent (_build)
│   │   ├── config.py               # ChartSelectorSettings
│   │   └── schemas.py              # State definition
│   │
│   ├── viz_coordinator/            # Multi-agent coordinator
│   │   ├── __init__.py
│   │   ├── agent.py                # VizCoordinator builder
│   │   ├── config.py               # CoordinatorSettings
│   │   └── state.py
│   │
│   ├── streaming.py                # Streaming support
│   ├── memory.py                   # Memory integration
│   └── pipeline.py                 # Agent composition
│
├── containers/
│   ├── __init__.py
│   ├── agents.py                   # AgentsContainer
│   ├── application.py              # ApplicationContainer
│   └── testing.py                  # TestContainer
│
├── api/
│   ├── __init__.py
│   └── routes/
│       ├── visualizations.py       # Visualization endpoints
│       └── charts.py               # Chart endpoints
│
├── tools/
│   ├── __init__.py
│   ├── search.py                   # Search tools
│   ├── validation.py               # Validation tools
│   └── formatting.py               # Formatting tools
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Test fixtures
│   ├── containers.py               # Test containers
│   ├── test_tools/
│   │   ├── test_viz_search/
│   │   │   ├── test_tool.py        # Tool tests
│   │   │   └── test_config.py      # Config tests
│   │   ├── test_weather_api.py
│   │   └── test_database_query.py
│   ├── test_agents/
│   │   ├── test_viz_designer/
│   │   │   ├── test_agent.py       # Agent tests
│   │   │   ├── test_config.py      # Config tests
│   │   │   └── test_nodes.py       # Node tests
│   │   ├── test_chart_selector.py
│   │   └── test_coordinator.py
│   └── test_integration/
│       └── test_e2e_flow.py
│
├── .env                            # Environment variables
├── main.py                         # FastAPI application
└── pyproject.toml                  # Dependencies
```

## Summary

This architecture provides:

1. **Explicit Dependencies**: All dependencies injected through constructors
2. **Builder Pattern**: Consistent interface via `AgentBuilder` ABC — `_build()` for subclasses, `compile()` for consumers
3. **Non-overridable compile()**: `@final` + `__init_subclass__` guard ensures subclasses can't break the contract
4. **Subgraph Isolation**: "Invoke from a node" pattern gives each sub-agent its own focused state
5. **Input/Output Schema Separation**: `StateGraph(State, input=Input, output=Output)` hides internal bridge fields
6. **Flexibility**: Choose `create_agent()` or `StateGraph` per use case inside `_build()`
7. **Composability**: `compile()` returns `CompiledStateGraph` (full LangChain Runnable API)
8. **Testability**: Easy mocking through DI containers; sealed `compile()` is easy to test
9. **Type Safety**: Full typing with TypedDict and Pydantic
10. **Stateless Agents & Tools**: Per-request isolation, no shared state
11. **Composite Settings**: Agent and tool-specific configuration encapsulated
12. **Production Ready**: Error handling, streaming, memory, monitoring

**Key Takeaways:**

- Define agents as classes extending `AgentBuilder` ABC
- Implement `_build()` (protected) — consumers call `compile()` (public, cached, sealed)
- Define tools as classes implementing `ToolBuilder` (tools use `build()`)
- Inject all dependencies through `__init__()`
- Use composite Pydantic Settings for complex configuration
- Standardize structure: `agent.py` + `config.py` + `schemas.py` per component
- Use `AGENT_<NAME>_` prefix for agent settings
- Use `TOOL_<NAME>_` prefix for tool settings
- For multi-stage agents, use per-stage focused states (not a monolithic shared state)
- Use `input=`/`output=` on `StateGraph` to hide bridge fields from callers
- Use the "invoke from a node" pattern for subgraph integration
- Forward `config: RunnableConfig` to sub-agents for `AgentContext` propagation
- Wire through DI container, not manual instantiation
- Agents and tools are stateless — dependencies managed externally
- Test at unit, integration, and E2E levels

**Reference implementation:** See `agents/src/agents/viz_designer/agent.py` for a complete example of all patterns above.

---

**Version**: 1.3  
**Last Updated**: 2026-02-12  
**Author**: ActBI Engineering Team
