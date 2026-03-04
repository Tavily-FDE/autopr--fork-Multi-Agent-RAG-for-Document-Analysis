"""
Vector database setup and management (Chroma or FAISS).
"""
import os
from typing import Optional
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.retrievers import BaseRetriever


class RAGSetup:
    """Handles vector database initialization and setup"""

    def __init__(self, config: dict, embeddings: HuggingFaceEmbeddings):
        self.config = config
        self.embeddings = embeddings
        self.retriever: Optional[BaseRetriever] = None

    def setup(self) -> Optional[BaseRetriever]:
        """Initialize RAG system with chosen vector database"""
        vector_db_type = self.config["rag"].get("vector_db", "chroma").lower()
        papers_dir = self.config["rag"]["papers_dir"]
        
        if not os.path.exists(papers_dir):
            os.makedirs(papers_dir)
        
        # Route to appropriate vector DB setup
        if vector_db_type == "faiss":
            self._setup_faiss(papers_dir)
        else:  # Default to Chroma
            self._setup_chroma(papers_dir)
        
        return self.retriever

    def _setup_chroma(self, papers_dir: str):
        """Setup Chroma vector database"""
        persist_dir = self.config["rag"]["chroma"]["persist_dir"]
        retriever_k = self.config["rag"]["retriever_k"]
        chunk_size = self.config["rag"]["chunk_size"]
        chunk_overlap = self.config["rag"]["chunk_overlap"]
        
        print("   🔵 Using ChromaDB as vector store...")
        
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)
        
        # Load existing vector database from disk if available
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            print("   📚 Loading existing ChromaDB from disk...")
            try:
                vectorstore = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embeddings
                )
                self.retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})
                print("   ✅ ChromaDB loaded successfully.")
                return
            except Exception as e:
                print(f"   ⚠️  Error loading ChromaDB: {e}. Will create new one...")
        
        # If DB doesn't exist, read PDFs, split, and save
        print("   ⚙️  Reading PDFs and creating new ChromaDB...")
        self._create_and_save_chroma(
            papers_dir, persist_dir, chunk_size, chunk_overlap, retriever_k
        )

    def _create_and_save_chroma(
        self, papers_dir: str, persist_dir: str, 
        chunk_size: int, chunk_overlap: int, retriever_k: int
    ):
        """Create new Chroma database from PDFs"""
        loader = PyPDFDirectoryLoader(papers_dir)
        documents = loader.load()
        
        if not documents:
            print("   ⚠️  Warning: No PDF files found in 'papers' folder. Web search will be used only!")
            self.retriever = None
            return
        
        splits = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        ).split_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})
        print(f"   ✅ {len(documents)} PDF files loaded and saved to ChromaDB ('{persist_dir}').")

    def _setup_faiss(self, papers_dir: str):
        """Setup FAISS vector database"""
        persist_dir = self.config["rag"]["faiss"]["persist_dir"]
        index_name = self.config["rag"]["faiss"]["index_name"]
        retriever_k = self.config["rag"]["retriever_k"]
        chunk_size = self.config["rag"]["chunk_size"]
        chunk_overlap = self.config["rag"]["chunk_overlap"]
        
        faiss_index_path = os.path.join(persist_dir, index_name)
        
        print("   🟦 Using FAISS as vector store...")
        
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)
        
        # Load existing FAISS index if available
        if os.path.exists(faiss_index_path):
            print("   📚 Loading existing FAISS index from disk...")
            try:
                vectorstore = FAISS.load_local(
                    folder_path=persist_dir,
                    embeddings=self.embeddings,
                    index_name=index_name,
                    allow_dangerous_deserialization=True
                )
                self.retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})
                print("   ✅ FAISS index loaded successfully.")
                return
            except Exception as e:
                print(f"   ⚠️  Error loading FAISS index: {e}. Will create new one...")
        
        # If index doesn't exist, read PDFs, split, and save
        print("   ⚙️  Reading PDFs and creating new FAISS index...")
        self._create_and_save_faiss(
            papers_dir, persist_dir, index_name, chunk_size, chunk_overlap, retriever_k
        )

    def _create_and_save_faiss(
        self, papers_dir: str, persist_dir: str, index_name: str,
        chunk_size: int, chunk_overlap: int, retriever_k: int
    ):
        """Create new FAISS index from PDFs"""
        loader = PyPDFDirectoryLoader(papers_dir)
        documents = loader.load()
        
        if not documents:
            print("   ⚠️  Warning: No PDF files found in 'papers' folder. Web search will be used only!")
            self.retriever = None
            return
        
        splits = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        ).split_documents(documents)
        
        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        
        # Save FAISS index to disk
        vectorstore.save_local(
            folder_path=persist_dir,
            index_name=index_name
        )
        
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})
        print(f"   ✅ {len(documents)} PDF files loaded and saved to FAISS ('{os.path.join(persist_dir, index_name)}').")
