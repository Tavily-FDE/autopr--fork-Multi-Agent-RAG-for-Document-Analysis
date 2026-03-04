"""
Main RAG Multi-Agent system orchestrator.
"""
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END

from models import AgentState, GraderOutput, CriticOutput
from rag_setup import RAGSetup
from nodes import WorkflowNodes
from utils import should_search_web


class RAGMultiAgent:
    """Advanced Multi-Agent RAG System with LangGraph workflow"""
    
    def __init__(self, config_path: str = "./config.json"):
        print("🔧 System Initializing...\n")
        
        # Load configuration
        print("⚙️  Loading configuration...")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        print("1️⃣  Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config["embedding"]["model_name"]
        )
        
        print("2️⃣  Connecting to Ollama LLM...")
        self.llm = ChatOllama(
            model=self.config["llm"]["model"],
            temperature=self.config["llm"]["temperature"],
            base_url=self.config["llm"]["base_url"]
        )
        
        # Structured output LLM chains
        self.grader_llm = self.llm.with_structured_output(GraderOutput)
        self.critic_llm = self.llm.with_structured_output(CriticOutput)
        
        print("3️⃣  Setting up Vector Database (Persistence)...")
        rag = RAGSetup(self.config, self.embeddings)
        self.retriever = rag.setup()
        
        print("4️⃣  Setting up Web Search Tool (DuckDuckGo)...")
        self.web_search_tool = DuckDuckGoSearchRun()
        
        # Initialize workflow nodes
        self.nodes = WorkflowNodes(
            self.config,
            self.llm,
            self.grader_llm,
            self.critic_llm,
            self.retriever,
            self.web_search_tool
        )
        
        # Setup LangGraph workflow
        self._setup_workflow()
        
        print("\n✅ System Ready! You can now ask questions.\n")

    def _setup_workflow(self):
        """Setup LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("pdf_researcher", self.nodes.pdf_researcher_node)
        workflow.add_node("web_searcher", self.nodes.web_searcher_node)
        workflow.add_node("writer", self.nodes.writer_node)
        workflow.add_node("critic", self.nodes.critic_node)
        
        workflow.add_edge(START, "pdf_researcher")
        workflow.add_conditional_edges(
            "pdf_researcher",
            self._decide_if_web_needed,
            {"go_to_web": "web_searcher", "go_to_writer": "writer", "end_workflow": END}
        )
        workflow.add_edge("web_searcher", "writer")
        workflow.add_edge("writer", "critic")
        workflow.add_conditional_edges(
            "critic",
            self._should_continue,
            {"end": END, "rewrite": "writer"}
        )
        
        self.app = workflow.compile()

    def _decide_if_web_needed(self, state: AgentState) -> str:
        """Routing logic: Decide if web search is needed"""
        is_topic_related = state.get("is_topic_related", True)
        found_in_db = state.get("found_in_db", False)
        rag_context = state.get("rag_context", "")
        
        return should_search_web(is_topic_related, found_in_db, rag_context)

    def _should_continue(self, state: AgentState) -> str:
        """Iteration control: Decide if more revisions are needed"""
        if state["is_satisfactory"]:
            return "end"
        max_iterations = self.config["critic"]["max_iterations"]
        if state["iterations"] >= max_iterations:
            print(f"⚠️ Maximum revision limit ({max_iterations}) reached. Final report accepted.")
            return "end"
        return "rewrite"

    def process_query(self, user_question: str) -> str:
        """Process a user question and return the final report"""
        inputs = {
            "question": user_question,
            "iterations": 0,
            "rag_context": "",
            "web_context": "",
            "feedback": "",
            "report": "",
            "is_satisfactory": False,
            "is_relevant": False,
            "is_topic_related": True,
            "found_in_db": False
        }
        config = {"configurable": {"thread_id": "1"}}
        final_state = self.app.invoke(inputs, config)
        return final_state["report"]
    
    def run_interactive(self):
        """Run interactive Q&A mode"""
        print("=" * 60)
        print("🤖 Interactive Mode")
        print("=" * 60)
        print("Type 'quit', 'exit', or 'q' to exit\n")
        
        while True:
            try:
                user_input = input("\n❓ Enter your question: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\n👋 System shutting down. Goodbye!")
                    break
                if not user_input:
                    continue
                
                print("\n⏳ Processing...\n" + "-" * 60)
                report = self.process_query(user_input)
                print("\n" + "=" * 60)
                print("📊 FINAL REPORT")
                print("=" * 60)
                print(report)
                print("=" * 60)
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted. Exiting...")
                break
            except Exception as e:
                print(f"\n❌ Unexpected Error: {str(e)}")
                continue


if __name__ == "__main__":
    agent = RAGMultiAgent()
    agent.run_interactive()
