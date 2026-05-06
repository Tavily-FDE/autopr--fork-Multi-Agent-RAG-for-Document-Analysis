# RAG Multi-Agent System

An intelligent multi-agent system combining Retrieval-Augmented Generation (RAG) with agentic workflows. This system orchestrates PDF research, web search, report generation, and iterative refinement using LangGraph and Ollama for a complete document intelligence pipeline.

**Key Features:**
- 🤖 Multi-agent workflow powered by LangGraph
- 📚 RAG with Chroma/FAISS vector databases
- 🌐 Web search integration via DuckDuckGo or Tavily
- 🔄 Iterative report refinement with critic evaluation
- 🏗️ Modular, production-ready architecture
- ⚡ Local LLM support with Ollama

## 📁 Project Structure

```
project-folder/
├── config.json                      # Configuration file (setup here)
├── main.py                         # Main orchestrator (entry point)
├── models.py                       # Pydantic models & TypedDict
├── rag_setup.py                    # Vector DB setup (Chroma/FAISS)
├── nodes.py                        # Workflow nodes
├── utils.py                        # Helper functions
└── papers/                         # Your PDF files here
```

## � Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Ollama (For local LLM models) - https://ollama.ai
- Git (optional)

### Step 1: Clone the Project
```bash
git clone <your-repo-url>
cd multi_agent_rag
```

### Step 2: Create Virtual Environment

#### Option A: Using venv
```bash
# Linux/MacOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

#### Option B: Using conda
```bash
# Create conda environment
conda create -n rag-agent python=3.10

# Activate conda environment
conda activate rag-agent
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Setup Ollama
```bash
# Download and install Ollama from https://ollama.ai
# Then run your desired model, for example:
ollama pull mistral
ollama run mistral
```

### Step 5: Configure Settings
Edit `config.json` according to your needs:
- LLM model name
- Embedding model
- Vector database selection (Chroma or FAISS)
- Documents folder path

### Step 6: Add PDF Documents
Add PDF files to the `papers/` folder:
```bash
mkdir -p papers/
# Place PDF files in papers/ folder
```

## 🚀 Quick Start

### Option 1: Using main.py (Recommended)
```bash
python main.py
```
## 📦 Module Overview

### `models.py`
Contains all Pydantic and TypedDict models:
- `GraderOutput`: PDF relevance evaluation
- `CriticOutput`: Report quality evaluation with quality_score
- `AgentState`: Workflow state definition

### `rag_setup.py`
Vector database initialization:
- `RAGSetup` class orchestrates setup
- `_setup_chroma()`: ChromaDB setup
- `_setup_faiss()`: FAISS setup
- Automatic PDF loading and persistence

**Key features:**
- Auto-loads existing DB from disk
- Creates new index if not found
- Supports both Chroma and FAISS

### `nodes.py`
Workflow node implementations:
- `WorkflowNodes` class contains all nodes
- `pdf_researcher_node()`: Topic check + DB search
- `web_searcher_node()`: Web search
- `writer_node()`: Report generation
- `critic_node()`: Report evaluation

**Clean separation of concerns** - each node is a method

### `utils.py`
Helper functions:
- `check_topic_relevance()`: Heuristic-based topic classification
- `should_search_web()`: Routing decision logic

**Reusable utilities** - can be imported elsewhere

### `main.py`
Main orchestrator (`RAGMultiAgent` class):
- Initializes all components
- Sets up LangGraph workflow
- Routing logic handlers
- Interactive mode

**Single entry point** - all initialization happens here

## 🔄 Data Flow

```
User Input
    ↓
pdf_researcher_node
    ├─→ check_topic_relevance()
    └─→ Search database (if on-topic)
        ↓
    decide_if_web_needed()
        ├─→ end_workflow (if off-topic)
        ├─→ go_to_writer (if in DB)
        └─→ go_to_web (if on-topic but not in DB)
            ↓
        web_searcher_node
            ↓
        writer_node
            ├─→ Uses RAG context
            └─→ Uses web context
                ↓
            critic_node
                ├─→ end (if approved)
                └─→ rewrite (if iterations < max)
                    ↓ (loops back to writer_node)
                    Final Report
```

## 🌐 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `TAVILY_API_KEY` | Only when `web_search.provider` is `"tavily"` in `config.json` | API key for Tavily web search. Get one at https://tavily.com. Not needed when using the default DuckDuckGo provider. |

## 🎛️ Configuration

Edit `config.json` to:
- Switch web search provider: `"web_search": {"provider": "duckduckgo"}` (default) or `"tavily"` (requires `TAVILY_API_KEY`)
- Switch between **Chroma** and **FAISS**: `"vector_db": "chroma"` or `"faiss"`
- Adjust LLM settings (model, temperature)
- Set embedding model
- Configure context limits
- Change critic approval threshold

Edit `config.json` to customize:

```json
{
  "critic": {
    "max_iterations": 3,           // Max revision attempts
    "approval_threshold": 0.7      // Quality score to pass (0.0-1.0)
  },
  "prompts": {
    "grader": "..."                // Relevance checking prompt
    "critic": "..."                // Report evaluation prompt
    "writer_*": "..."              // Report generation prompts
  }
}
```

## 🔧 Adding New Components

### Add a new workflow node:
1. Add method to `WorkflowNodes` class in `nodes.py`
2. Add node to workflow in `main.py`'s `_setup_workflow()`

### Add a new vector database:
1. Create new method in `rag_setup.py` (e.g., `_setup_weaviate()`)
2. Update routing logic in `setup()` method

### Add new helpers:
1. Add function to `utils.py`
2. Import where needed

## 🎯 Next Improvements

- [ ] Add tests for each module
- [ ] Create API wrapper
- [ ] Add async support
- [ ] Support for more LLM providers
- [ ] Caching layer for web searches
- [ ] Monitoring/logging improvements
