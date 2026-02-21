# Dependency Injection Architecture Design

## Overview

This document describes the dependency injection (DI) architecture for the Acme Inc. platform. We use `dependency-injector` with Pydantic Settings to provide centralized configuration management, explicit dependency graphs, and flexible component wiring across our FastAPI backend, LangChain agents, and AcmeStore clients.

## Core Principles

### 1. Package-Level Settings Ownership

Each package owns its configuration through Pydantic Settings classes with environment variable prefixes:

```python
# acmestore/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class AcmeStoreSettings(BaseSettings):
    """AcmeStore configuration from ACMESTORE_* environment variables"""
    base_url: str
    api_key: str
    timeout: float = 30.0
    max_retries: int = 3
    
    class Config:
        env_prefix = "ACMESTORE_"
        env_file = ".env"
        case_sensitive = False

@lru_cache
def get_acmestore_settings() -> AcmeStoreSettings:
    """Cached settings instance - called once per process"""
    return AcmeStoreSettings()
```

**Benefits:**
- Clear ownership: Each package controls its configuration
- Type safety: Pydantic validates types at startup
- Environment-based: Configuration comes from env vars, not code
- Self-documenting: Settings class shows all required configuration

### 2. Constructor Injection

All dependencies are passed explicitly through `__init__()` methods - no global state:

```python
# acmestore/client.py
import httpx

class AcmeStoreClient:
    """AcmeStore API client with explicit dependencies"""
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
    
    async def get_chart(self, chart_id: str) -> dict:
        response = await self.client.get(f"/charts/{chart_id}")
        return response.json()
    
    async def close(self):
        await self.client.aclose()
```

**Benefits:**
- Testability: Easy to inject mocks in tests
- Clarity: Dependencies are explicit in function signatures
- Flexibility: Can create instances with different configurations
- No side effects: No hidden global state

### 3. Singleton vs Factory Pattern

The DI container uses two provider types:

**Singleton** - Same instance reused across requests:
- Expensive resources (HTTP clients, database connections)
- Stateful components (LLM instances, vector stores)
- Configuration objects

**Factory** - New instance created per request:
- Stateless services
- Per-request agents
- Request handlers

```python
# containers.py
class ApplicationContainer(containers.DeclarativeContainer):
    # Singleton - reuse expensive HTTP client
    acmestore_client = providers.Singleton(
        AcmeStoreClient,
        base_url=acmestore_settings.provided.base_url,
        api_key=acmestore_settings.provided.api_key,
    )
    
    # Factory - new agent per request
    chart_selection_agent = providers.Factory(
        ChartSelectionAgent,
        llm=llm,
        acmestore_client=acmestore_client,  # Injects the singleton
    )
```

## Architecture

### Component Layers

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Routes                        │
│              (@inject + Depends(Provide))                │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              ApplicationContainer                        │
│    (Defines how to create and wire all components)      │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┬──────────────┐
         ▼                       ▼              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Services     │    │     Agents      │    │    Clients      │
│ (Factory)       │    │   (Factory)     │    │  (Singleton)    │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                       │
         └──────────────────────┴───────────────────────┘
                                │
                    ┌───────────▼──────────┐
                    │  Package Settings    │
                    │  (Pydantic)          │
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  Environment Vars    │
                    │  (.env file)         │
                    └──────────────────────┘
```

### Dependency Flow Example

```
User Request → FastAPI Route
                    ↓
         @inject + Depends(Provide[Container.visualization_service])
                    ↓
         Container looks up visualization_service
                    ↓
    Needs: chart_selection_agent + acmestore_client
                    ↓
    ┌─────────────┴─────────────┐
    ↓                           ↓
Create Agent                Get AcmeStore Client (Singleton)
Needs: llm + acmestore_client       ↓
    ↓                       Get acmestore_settings
Create LLM (Singleton)          ↓
    ↓                       Create AcmeStoreClient(...)
Get agent_settings              ↓
    ↓                       Return instance
Create ChatOpenAI(...)
    ↓
Get acmestore_client (reuse singleton)
    ↓
Create ChartSelectionAgent(...)
    ↓
    └─────────────┬─────────────┘
                  ↓
    Create InteractionService(...)
                  ↓
    Return to route handler
```

## Implementation Patterns

### Pattern 1: Agent with Injected Dependencies

```python
# agents/chart_selection_agent.py
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from acmestore.client import AcmeStoreClient
from typing import List

class ChartSelectionAgent:
    """Chart selection agent with injected dependencies"""
    
    def __init__(
        self,
        llm: ChatOpenAI,
        acmestore_client: AcmeStoreClient,
        tools: List = None,
    ):
        """
        Initialize agent with dependencies.
        
        Args:
            llm: LangChain LLM instance for reasoning
            acmestore_client: AcmeStore client for chart operations
            tools: List of LangChain tools for the agent
        """
        self.llm = llm
        self.acmestore_client = acmestore_client
        self.tools = tools or []
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a data interaction expert..."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Create agent using injected LLM
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt,
        )
    
    async def select_chart(
        self,
        query: str,
        data_schema: dict,
    ) -> dict:
        """
        Select optimal chart type for query and data.
        
        Args:
            query: Natural language query from user
            data_schema: Schema of the data to visualize
            
        Returns:
            Chart configuration with type and settings
        """
        result = await self.agent.ainvoke({
            "input": query,
            "schema": data_schema,
        })
        
        return result
```

### Pattern 2: Service Orchestrating Multiple Components

```python
# services/visualization_service.py
from agents.chart_selection_agent import ChartSelectionAgent
from acmestore.client import AcmeStoreClient

class InteractionService:
    """High-level service for interaction creation"""
    
    def __init__(
        self,
        agent: ChartSelectionAgent,
        acmestore_client: AcmeStoreClient,
    ):
        """
        Initialize service with dependencies.
        
        Args:
            agent: Agent for chart selection logic
            acmestore_client: Client for AcmeStore operations
        """
        self.agent = agent
        self.acmestore_client = acmestore_client
    
    async def create_visualization(
        self,
        query: str,
        data_schema: dict,
    ) -> dict:
        """
        Create complete interaction from user query.
        
        Args:
            query: Natural language description
            data_schema: Structure of data to visualize
            
        Returns:
            Saved chart configuration
        """
        # Use injected agent for selection
        chart_config = await self.agent.select_chart(
            query=query,
            data_schema=data_schema,
        )
        
        # Use injected client to save
        saved_chart = await self.acmestore_client.create_chart(chart_config)
        
        return saved_chart
```

### Pattern 3: Container Configuration

```python
# containers.py
from dependency_injector import containers, providers
from acmestore.config import get_acmestore_settings
from acmestore.client import AcmeStoreClient
from agents.config import get_agent_settings
from agents.chart_selection_agent import ChartSelectionAgent
from services.visualization_service import InteractionService
from langchain_openai import ChatOpenAI

class ApplicationContainer(containers.DeclarativeContainer):
    """
    Main application container.
    
    Defines how to create and wire all components.
    Uses Singleton for expensive resources, Factory for per-request.
    """
    
    # ==================== Settings ====================
    acmestore_settings = providers.Singleton(get_acmestore_settings)
    agent_settings = providers.Singleton(get_agent_settings)
    
    # ==================== Infrastructure ====================
    
    # AcmeStore Client (Singleton - reuse HTTP connection)
    acmestore_client = providers.Singleton(
        AcmeStoreClient,
        base_url=acmestore_settings.provided.base_url,
        api_key=acmestore_settings.provided.api_key,
        timeout=acmestore_settings.provided.timeout,
    )
    
    # ==================== LLM & Tools ====================
    
    # LLM (Singleton - reuse for efficiency)
    llm = providers.Singleton(
        ChatOpenAI,
        model=agent_settings.provided.llm_model,
        temperature=agent_settings.provided.llm_temperature,
        api_key=agent_settings.provided.openai_api_key,
    )
    
    # Tools list for agents
    tools = providers.List()
    
    # ==================== Agents ====================
    
    # Chart Selection Agent (Factory - new per request)
    chart_selection_agent = providers.Factory(
        ChartSelectionAgent,
        llm=llm,  # Injects singleton LLM
        acmestore_client=acmestore_client,  # Injects singleton client
        tools=tools,
    )
    
    # ==================== Services ====================
    
    # Visualization Service (Factory - new per request)
    visualization_service = providers.Factory(
        InteractionService,
        agent=chart_selection_agent,  # Creates new agent
        acmestore_client=acmestore_client,  # Injects singleton client
    )
    
    # ==================== Wiring ====================
    wiring_config = containers.WiringConfiguration(
        modules=[
            "api.routes.charts",
            "api.routes.dashboards",
            "api.routes.visualizations",
        ]
    )
```

### Pattern 4: FastAPI Route with Dependency Injection

```python
# api/routes/charts.py
from fastapi import APIRouter, Depends
from dependency_injector.wiring import inject, Provide
from containers import ApplicationContainer
from services.visualization_service import InteractionService
from acmestore.client import AcmeStoreClient

router = APIRouter()

@router.post("/charts/select")
@inject
async def select_chart(
    query: str,
    data_schema: dict,
    visualization_service: InteractionService = Depends(
        Provide[ApplicationContainer.visualization_service]
    ),
):
    """
    Select and create chart interaction.
    
    The visualization_service is automatically injected by the DI container.
    It comes with all its dependencies (agent, AcmeStore client, LLM, etc.) 
    already wired together.
    """
    result = await visualization_service.create_visualization(
        query=query,
        data_schema=data_schema,
    )
    return result


@router.get("/charts/{chart_id}")
@inject
async def get_chart(
    chart_id: str,
    acmestore_client: AcmeStoreClient = Depends(
        Provide[ApplicationContainer.acmestore_client]
    ),
):
    """
    Get chart by ID.
    
    Only needs AcmeStore client, not the full service stack.
    Container provides just what we need.
    """
    chart = await acmestore_client.get_chart(chart_id)
    return chart
```

### Pattern 5: FastAPI Application Setup

```python
# main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from containers import ApplicationContainer

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles initialization and cleanup of async resources
    like HTTP clients, database connections, etc.
    """
    # Startup
    container = app.container
    
    # Initialize async resources
    acmestore_client = container.acmestore_client()
    await acmestore_client.initialize()
    
    yield
    
    # Shutdown
    await acmestore_client.close()

def create_app() -> FastAPI:
    """
    Application factory.
    
    Creates FastAPI app with dependency injection container.
    """
    app = FastAPI(
        title="Acme Inc. API",
        lifespan=lifespan,
    )
    
    # Create and attach container
    container = ApplicationContainer()
    app.container = container
    
    # Import and register routes
    from api.routes import charts, dashboards
    app.include_router(charts.router, prefix="/api")
    app.include_router(dashboards.router, prefix="/api")
    
    return app

app = create_app()
```

## Testing Patterns

### Pattern 6: Test Container with Mocks

```python
# tests/containers.py
from dependency_injector import containers, providers
from containers import ApplicationContainer
from unittest.mock import Mock
from acmestore.client import AcmeStoreClient
from langchain_openai import ChatOpenAI

class TestContainer(ApplicationContainer):
    """
    Test container with mocked external dependencies.
    
    Inherits from ApplicationContainer but overrides external
    services with mocks for isolated testing.
    """
    
    # Override AcmeStore client with mock
    acmestore_client = providers.Singleton(
        Mock,
        spec=AcmeStoreClient,
    )
    
    # Use cheaper/faster LLM for tests
    llm = providers.Singleton(
        ChatOpenAI,
        model="gpt-3.5-turbo",
        temperature=0,
    )
    
    # Keep real agents and services to test business logic
```

### Pattern 7: Test Fixtures

```python
# tests/conftest.py
import pytest
from tests.containers import TestContainer

@pytest.fixture
def container():
    """
    Provide test container for all tests.
    
    Automatically wires and unwires for each test.
    """
    container = TestContainer()
    yield container
    container.unwire()

@pytest.fixture
def acmestore_client(container):
    """Get mocked AcmeStore client from container"""
    return container.acmestore_client()

@pytest.fixture
def chart_selection_agent(container):
    """Get agent with mocked dependencies"""
    return container.chart_selection_agent()

@pytest.fixture
def visualization_service(container):
    """Get service with all dependencies wired"""
    return container.visualization_service()
```

### Pattern 8: Writing Tests

```python
# tests/test_agents.py
import pytest

def test_chart_selection(chart_selection_agent):
    """
    Test chart selection with injected agent.
    
    Agent has mocked AcmeStore client and real (test) LLM from TestContainer.
    """
    result = chart_selection_agent.select_chart(
        query="Show sales by region",
        data_schema={"columns": ["region", "sales"]}
    )
    
    assert result is not None
    assert "chart_type" in result


def test_chart_selection_custom_mock(container, mocker):
    """
    Test with custom mock overriding container dependency.
    """
    # Create custom mock
    mock_acmestore = mocker.Mock()
    mock_acmestore.create_chart.return_value = {"id": "123", "type": "bar"}
    
    # Override specific dependency for this test
    with container.acmestore_client.override(mock_acmestore):
        agent = container.chart_selection_agent()
        result = agent.select_chart("test query", {})
        
        # Verify mock was called
        mock_acmestore.create_chart.assert_called_once()


def test_visualization_service(visualization_service, mocker):
    """
    Test service with all dependencies mocked.
    """
    # Service has mocked agent and client
    result = visualization_service.create_visualization(
        query="Compare quarterly revenue",
        data_schema={"columns": ["quarter", "revenue"]}
    )
    
    assert result is not None
```

## Environment Configuration

### Environment Variable Organization

Use package prefixes to organize configuration:

```bash
# .env

# AcmeStore Configuration
ACMESTORE_BASE_URL=http://localhost:8001
ACMESTORE_API_KEY=secret_key_123
ACMESTORE_TIMEOUT=60.0
ACMESTORE_MAX_RETRIES=5

# Agent Configuration
AGENT_LLM_MODEL=gpt-4
AGENT_LLM_TEMPERATURE=0.0
AGENT_OPENAI_API_KEY=sk-...
AGENT_MAX_TOKENS=8000

# Database Configuration
DB_URL=postgresql://localhost:5432/acme
DB_POOL_SIZE=20
DB_POOL_TIMEOUT=30

# Auth Configuration
AUTH_JWT_SECRET=your-secret-key
AUTH_GOOGLE_CLIENT_ID=xxx.apps.googleusercontent.com
AUTH_GOOGLE_CLIENT_SECRET=GOCSPX-xxx

# Supabase Configuration
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
SUPABASE_SERVICE_ROLE_KEY=eyJ...
```

### Settings Class Mapping

| Package | Settings Class | Env Prefix | Example Variables |
|---------|---------------|------------|-------------------|
| acmestore | `AcmeStoreSettings` | `ACMESTORE_` | `ACMESTORE_BASE_URL`, `ACMESTORE_API_KEY` |
| agents | `AgentSettings` | `AGENT_` | `AGENT_LLM_MODEL`, `AGENT_OPENAI_API_KEY` |
| database | `DatabaseSettings` | `DB_` | `DB_URL`, `DB_POOL_SIZE` |
| auth | `AuthSettings` | `AUTH_` | `AUTH_JWT_SECRET`, `AUTH_GOOGLE_CLIENT_ID` |

## Multiple Container Configurations

### Production Container

```python
# containers.py
class ApplicationContainer(containers.DeclarativeContainer):
    """Full production container with all real services"""
    
    acmestore_settings = providers.Singleton(get_acmestore_settings)
    
    acmestore_client = providers.Singleton(
        AcmeStoreClient,
        base_url=acmestore_settings.provided.base_url,
        api_key=acmestore_settings.provided.api_key,
    )
    
    # ... all production providers
```

### Testing Container

```python
# tests/containers.py
class TestContainer(ApplicationContainer):
    """Test container with mocked external dependencies"""
    
    acmestore_client = providers.Singleton(Mock, spec=AcmeStoreClient)
    llm = providers.Singleton(ChatOpenAI, model="gpt-3.5-turbo")
```

### Minimal Container (for Scripts/CLI)

```python
# containers/minimal.py
class MinimalAcmeStoreContainer(containers.DeclarativeContainer):
    """Minimal container for AcmeStore operations only"""
    
    acmestore_settings = providers.Singleton(get_acmestore_settings)
    
    acmestore_client = providers.Singleton(
        AcmeStoreClient,
        base_url=acmestore_settings.provided.base_url,
        api_key=acmestore_settings.provided.api_key,
    )

# Usage in script
container = MinimalAcmeStoreContainer()
client = container.acmestore_client()
result = client.get_chart("chart-123")
```

### Development Container

```python
# containers/development.py
class DevelopmentContainer(ApplicationContainer):
    """Development with local overrides"""
    
    # Override to use local AcmeStore instance
    acmestore_client = providers.Singleton(
        AcmeStoreClient,
        base_url="http://localhost:8001",
        api_key="dev-key",
    )
    
    # Use local vector store instead of Supabase
    vector_store = providers.Singleton(
        ChromaDB,
        persist_directory="./dev_db"
    )
```

## File Organization

```
acme/
├── .env                              # Environment variables
│
├── containers.py                     # Main application container
├── containers/
│   ├── __init__.py
│   ├── testing.py                   # Test container
│   ├── minimal.py                   # Minimal containers
│   └── development.py               # Development overrides
│
├── acmestore/
│   ├── __init__.py
│   ├── config.py                    # AcmeStoreSettings
│   ├── client.py                    # AcmeStoreClient
│   └── models.py
│
├── agents/
│   ├── __init__.py
│   ├── config.py                    # AgentSettings
│   ├── chart_selection_agent.py    # Agents
│   └── refinement_agent.py
│
├── database/
│   ├── __init__.py
│   ├── config.py                    # DatabaseSettings
│   └── client.py
│
├── services/
│   ├── __init__.py
│   └── visualization_service.py    # Services
│
├── api/
│   ├── __init__.py
│   └── routes/
│       ├── __init__.py
│       ├── charts.py               # Routes with @inject
│       ├── dashboards.py
│       └── visualizations.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Test fixtures
│   ├── containers.py               # Test container
│   ├── test_agents.py
│   └── test_services.py
│
└── main.py                         # FastAPI app
```

## Best Practices

### 1. Always Use Type Hints

```python
# Good
def __init__(
    self,
    llm: ChatOpenAI,
    acmestore_client: AcmeStoreClient,
    tools: List[Tool] = None,
):
    ...

# Bad
def __init__(self, llm, acmestore_client, tools=None):
    ...
```

### 2. Document Dependencies in Docstrings

```python
def __init__(
    self,
    llm: ChatOpenAI,
    acmestore_client: AcmeStoreClient,
):
    """
    Initialize agent with dependencies.
    
    Args:
        llm: LangChain LLM for reasoning and generation
        acmestore_client: Client for chart operations and storage
    """
    ...
```

### 3. Use Singleton for Expensive Resources

```python
# Singleton - reuse HTTP connection pool
acmestore_client = providers.Singleton(AcmeStoreClient, ...)

# Singleton - reuse LLM instance
llm = providers.Singleton(ChatOpenAI, ...)

# Singleton - reuse database connection pool
database = providers.Singleton(Database, ...)
```

### 4. Use Factory for Stateless Components

```python
# Factory - new agent per request
chart_selection_agent = providers.Factory(ChartSelectionAgent, ...)

# Factory - new service per request
visualization_service = providers.Factory(InteractionService, ...)
```

### 5. Keep Settings Classes Simple

```python
# Good - focused, clear
class AcmeStoreSettings(BaseSettings):
    base_url: str
    api_key: str
    timeout: float = 30.0
    
    class Config:
        env_prefix = "ACMESTORE_"

# Bad - mixing concerns
class Settings(BaseSettings):
    acmestore_url: str
    db_url: str
    auth_secret: str
    # Too many responsibilities
```

### 6. Wire Containers to Specific Modules

```python
# Good - explicit module list
wiring_config = containers.WiringConfiguration(
    modules=[
        "api.routes.charts",
        "api.routes.dashboards",
    ]
)

# Bad - wiring too broadly
wiring_config = containers.WiringConfiguration(
    packages=["api"]  # Wires everything, can be slow
)
```

### 7. Use Container Overrides in Tests

```python
# Good - override specific dependencies
with container.acmestore_client.override(mock_client):
    service = container.visualization_service()
    # Test with mocked client

# Bad - creating instances manually in tests
service = InteractionService(
    agent=mock_agent,
    acmestore_client=mock_client,
)  # Bypasses container, loses DI benefits
```

### 8. Manage Async Resource Lifecycle

```python
# Good - proper initialization and cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    client = app.container.acmestore_client()
    await client.initialize()
    
    yield
    
    # Shutdown
    await client.close()

# Bad - no cleanup
app = FastAPI()
client = app.container.acmestore_client()
# Connections leak on shutdown
```

## Common Patterns

### Pattern: Creating Tools for Agents

```python
# containers.py
def create_search_tool(vector_store: VectorStore):
    """Factory function for search tool"""
    return Tool(
        name="search_rules",
        description="Search interaction design rules",
        func=vector_store.search,
    )

class ApplicationContainer(containers.DeclarativeContainer):
    vector_store = providers.Singleton(VectorStore, ...)
    
    # Use Callable provider for factory function
    search_tool = providers.Factory(
        create_search_tool,
        vector_store=vector_store,
    )
    
    # Create list of tools
    tools = providers.List(
        search_tool,
        # other tools...
    )
    
    # Agent gets tools list
    agent = providers.Factory(
        ChartSelectionAgent,
        llm=llm,
        tools=tools,
    )
```

### Pattern: Conditional Configuration

```python
# containers.py
import os

class ApplicationContainer(containers.DeclarativeContainer):
    settings = providers.Singleton(get_settings)
    
    # Use different provider based on environment
    llm = providers.Singleton(
        ChatOpenAI,
        model=settings.provided.llm_model,
        api_key=settings.provided.openai_api_key,
        # Add debug logging in development
        verbose=os.getenv("ENVIRONMENT") == "development",
    )
```

### Pattern: Nested Dependencies

```python
class ApplicationContainer(containers.DeclarativeContainer):
    # Level 1: Settings
    db_settings = providers.Singleton(get_database_settings)
    
    # Level 2: Database (depends on settings)
    database = providers.Singleton(
        Database,
        url=db_settings.provided.url,
    )
    
    # Level 3: Repository (depends on database)
    chart_repository = providers.Singleton(
        ChartRepository,
        database=database,
    )
    
    # Level 4: Service (depends on repository)
    chart_service = providers.Factory(
        ChartService,
        repository=chart_repository,
    )
```

## Required Dependencies

```bash
# Install with uv
uv add dependency-injector[yaml]
uv add pydantic-settings
uv add fastapi
uv add langchain langchain-openai
uv add httpx

# Development dependencies
uv add --dev pytest
uv add --dev pytest-asyncio
uv add --dev pytest-mock
```

## Key Benefits

1. **Testability**: Easy to inject mocks and test components in isolation
2. **Flexibility**: Multiple container configurations for different contexts
3. **Type Safety**: Pydantic validates configuration at startup
4. **Clarity**: Dependencies are explicit in function signatures
5. **Maintainability**: Package-level settings ownership
6. **No Global State**: All dependencies injected, no hidden coupling
7. **Lifecycle Management**: Proper async resource initialization/cleanup
8. **Configuration Management**: Environment-based, type-safe settings

## Resources

- **dependency-injector**: https://python-dependency-injector.ets-labs.org/
- **Pydantic Settings**: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
- **FastAPI Dependency Injection**: https://fastapi.tiangolo.com/tutorial/dependencies/

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-04  
**Author**: Acme Inc. Engineering Team
