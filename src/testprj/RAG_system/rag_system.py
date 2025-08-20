"""
RAG (Retrieval Augmented Generation) System
===========================================

This system combines document retrieval with language generation to answer questions
based on your own documents. It works by:

1. Reading documents from a folder
2. Breaking them into chunks
3. Converting chunks to embeddings (numerical representations)
4. Storing embeddings in a vector database
5. When asked a question, finding relevant chunks
6. Using those chunks to generate informed answers

Example workflow:
User asks: "What's our company's vacation policy?"
→ System searches through all company documents
→ Finds relevant sections from employee handbook
→ Uses Llama 3.2 to generate answer based on found content
→ Cites the source document
"""

import asyncio
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any
import hashlib

# Core libraries for vector database and document processing
import chromadb  # Local vector database - stores document embeddings
from chromadb.config import Settings
import PyPDF2  # For reading PDF files
import docx  # For reading Word documents
from sentence_transformers import SentenceTransformer  # For generating embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Smart text chunking

# Agent framework for creating AI agents
from agents import (
    Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, 
    TResponseInputItem, function_tool, set_tracing_disabled, RawResponsesStreamEvent
)
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class RAGSystem:
    """
    Main RAG System Class
    ====================
    
    This class handles all RAG functionality:
    - Document processing and storage
    - Vector search
    - AI agent creation
    
    Example usage:
        rag = RAGSystem(books_directory="./my_docs")
        rag.scan_and_process_books()  # Process all documents
        agent = rag.create_rag_agent()  # Create AI agent
    """
    
    def __init__(self, books_directory: str = None, db_path: str = "./chroma_db"):
        # Auto-detect books directory location
        if books_directory is None:
            # Try different possible locations
            possible_locations = [
                "./books",  # Project root
                "./src/testprj/RAG_system/books",  # Inside RAG_system
                "books",  # Current directory
                Path(__file__).parent / "books"  # Same directory as this script
            ]
            
            books_directory = "./books"  # Default
            for location in possible_locations:
                if Path(location).exists():
                    books_directory = location
                    print(f"Found books directory at: {location}")
                    break
            else:
                print("No books directory found. Creating default at ./books")
                books_directory = "./books"
        """
        Initialize the RAG system
        
        Args:
            books_directory: Where to look for documents (PDFs, DOCX, etc.)
            db_path: Where to store the vector database
            
        Example:
            # Default setup
            rag = RAGSystem()
            
            # Custom paths
            rag = RAGSystem(
                books_directory="./company_docs", 
                db_path="./my_vectordb"
            )
        """
        self.books_directory = Path(books_directory)
        self.db_path = Path(db_path)
        
        # Create directories if they don't exist
        # Example: If "./books" doesn't exist, create it
        self.books_directory.mkdir(exist_ok=True)
        self.db_path.mkdir(exist_ok=True)
        
        # Initialize ChromaDB - our local vector database
        # Think of this as a smart filing cabinet that can find similar documents
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)  # Disable usage tracking
        )
        
        # Create or get existing collection (like a table in a database)
        # Example: All our document chunks will be stored in this collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity for search
        )
        
        # Initialize embedding model - converts text to numbers that represent meaning
        # Example: "dog" and "puppy" will have similar embeddings
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and good quality
        
        # Initialize text splitter - breaks long documents into smaller chunks
        # Why? LLMs work better with smaller, focused pieces of text
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,        # Each chunk ~1000 characters (about 2-3 paragraphs)
            chunk_overlap=200,      # 200 character overlap between chunks for context
            length_function=len,    # Use character count for length
        )
        # Example: A 5000 character document becomes ~5 chunks with overlap
        
        # Set up Ollama/Llama model for generating responses
        self.setup_llm()
        
        # Track which files we've already processed to avoid duplicate work
        self.processed_files = self.get_processed_files()
        
    def setup_llm(self):
        """
        Setup Llama 3.2 via Ollama for text generation
        
        This connects to your local Ollama server running Llama 3.2
        Example: When user asks a question, this model generates the answer
        """
        # Connect to local Ollama server
        external_provider = AsyncOpenAI(
            api_key="ollama",  # Dummy value (Ollama doesn't need real API key)
            base_url="http://localhost:11434/v1"  # Local Ollama endpoint
        )
        
        # Create model interface
        self.model = OpenAIChatCompletionsModel(
            model="llama3.2:latest",  # Use your local Llama 3.2 model
            openai_client=external_provider
        )
        
        set_tracing_disabled(True)  # Disable internal logging for performance
    
    def get_processed_files(self) -> set:
        """
        Get list of files we've already processed from the database
        
        Why? If you run the system again, we don't want to reprocess 
        the same files - that would be slow and wasteful.
        
        Returns:
            Set of file paths that are already in the database
            
        Example:
            If database contains chunks from "manual.pdf" and "guide.docx",
            this returns {"./books/manual.pdf", "./books/guide.docx"}
        """
        try:
            # Get all metadata from stored chunks
            results = self.collection.get(include=["metadatas"])
            files = set()
            
            # Extract unique file paths from metadata
            for metadata in results["metadatas"]:
                if "file_path" in metadata:
                    files.add(metadata["file_path"])
            return files
        except:
            # If database is empty or has issues, return empty set
            return set()
    
    def extract_text_from_pdf(self, file_path: Path) -> str:
        """
        Extract text content from PDF files
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
            
        Example:
            pdf_text = rag.extract_text_from_pdf(Path("./books/manual.pdf"))
            # pdf_text now contains all text from the PDF
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Go through each page and extract text
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                    
                return text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: Path) -> str:
        """
        Extract text content from Word documents (.docx)
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Extracted text as a string
            
        Example:
            docx_text = rag.extract_text_from_docx(Path("./books/policy.docx"))
            # docx_text contains all paragraphs from the Word document
        """
        try:
            doc = docx.Document(file_path)
            text = ""
            
            # Extract text from each paragraph
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
                
            return text
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path: Path) -> str:
        """
        Extract text from plain text files (.txt, .md)
        
        Args:
            file_path: Path to the text file
            
        Returns:
            File content as a string
            
        Example:
            txt_content = rag.extract_text_from_txt(Path("./books/notes.txt"))
            # txt_content contains the entire file content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT {file_path}: {e}")
            return ""
    
    def extract_text(self, file_path: Path) -> str:
        """
        Universal text extractor - automatically handles different file types
        
        Args:
            file_path: Path to any supported file
            
        Returns:
            Extracted text regardless of file type
            
        Example:
            # Works with any supported file type
            text1 = rag.extract_text(Path("./books/manual.pdf"))
            text2 = rag.extract_text(Path("./books/policy.docx"))
            text3 = rag.extract_text(Path("./books/notes.txt"))
        """
        extension = file_path.suffix.lower()
        
        # Route to appropriate extractor based on file extension
        if extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif extension == '.docx':
            return self.extract_text_from_docx(file_path)
        elif extension in ['.txt', '.md']:
            return self.extract_text_from_txt(file_path)
        else:
            print(f"Unsupported file type: {extension}")
            return ""
    
    def process_document(self, file_path: Path):
        """
        Process a single document and add it to the vector database
        
        This is the core processing pipeline:
        1. Extract text from document
        2. Split into chunks
        3. Generate embeddings
        4. Store in database
        
        Args:
            file_path: Path to document to process
            
        Example:
            rag.process_document(Path("./books/company_manual.pdf"))
            # Now the manual is searchable in the system
        """
        # Skip if already processed
        if str(file_path) in self.processed_files:
            print(f"Skipping already processed file: {file_path.name}")
            return
        
        print(f"Processing: {file_path.name}")
        
        # Step 1: Extract text from the document
        text = self.extract_text(file_path)
        if not text.strip():
            print(f"No text extracted from {file_path.name}")
            return
        
        # Step 2: Split text into manageable chunks
        # Example: A 10,000 character document becomes ~10 chunks
        chunks = self.text_splitter.split_text(text)
        print(f"Split into {len(chunks)} chunks")
        
        # Step 3: Generate embeddings (vector representations) for each chunk
        # Example: "The company policy states..." becomes [0.1, -0.3, 0.7, ...]
        embeddings = self.embedding_model.encode(chunks).tolist()
        
        # Step 4: Prepare data for storage
        ids = []        # Unique identifiers for each chunk
        metadatas = []  # Information about each chunk
        
        for i, chunk in enumerate(chunks):
            # Create unique ID: filename_chunkindex_hash
            # Example: "manual_0_a1b2c3d4", "manual_1_e5f6g7h8"
            chunk_id = f"{file_path.stem}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
            ids.append(chunk_id)
            
            # Store metadata about this chunk
            metadata = {
                "file_path": str(file_path),        # Where it came from
                "file_name": file_path.name,        # Just the filename
                "chunk_index": i,                   # Which chunk number
                "total_chunks": len(chunks)         # How many total chunks
            }
            metadatas.append(metadata)
        
        # Step 5: Add everything to ChromaDB
        self.collection.add(
            embeddings=embeddings,  # Vector representations
            documents=chunks,       # Original text chunks
            metadatas=metadatas,   # Information about each chunk
            ids=ids                # Unique identifiers
        )
        
        # Update our tracking
        self.processed_files.add(str(file_path))
        print(f"Successfully processed {file_path.name}")
    
    def scan_and_process_books(self):
        """
        Scan the books directory and process all supported documents
        
        This is typically the first thing you run - it finds all your documents
        and processes them into the searchable database.
        
        Example:
            # Put files in ./books/ folder:
            # - manual.pdf
            # - policies.docx  
            # - notes.txt
            
            rag.scan_and_process_books()
            # All three files are now processed and searchable
        """
        print("Scanning books directory...")
        
        # Define which file types we can handle
        supported_extensions = {'.pdf', '.docx', '.txt', '.md'}
        
        # Find all files in the books directory (including subdirectories)
        files_found = list(self.books_directory.glob("**/*"))
        
        # Filter to only supported document types
        document_files = [
            f for f in files_found 
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
        
        if not document_files:
            print("No supported document files found in books directory.")
            print("Supported formats: PDF, DOCX, TXT, MD")
            return
        
        print(f"Found {len(document_files)} document files")
        
        # Process each document
        for file_path in document_files:
            self.process_document(file_path)
        
        print("Processing complete!")
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for document chunks relevant to a query
        
        This is where the magic happens - convert the user's question into
        an embedding and find the most similar document chunks.
        
        Args:
            query: User's question or search term
            n_results: How many relevant chunks to return
            
        Returns:
            List of relevant document chunks with metadata
            
        Example:
            results = rag.search_documents("vacation policy", n_results=3)
            # Returns 3 most relevant chunks about vacation policies
            
            for result in results:
                print(f"From {result['metadata']['file_name']}: {result['content']}")
        """
        # Convert the user's query into an embedding
        # Example: "vacation policy" → [0.2, -0.1, 0.8, ...]
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search ChromaDB for similar embeddings
        # It uses cosine similarity to find chunks with similar meaning
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results into a nice structure
        search_results = []
        for i in range(len(results["documents"][0])):
            search_results.append({
                "content": results["documents"][0][i],      # The actual text chunk
                "metadata": results["metadatas"][0][i],     # File info, chunk info
                "distance": results["distances"][0][i]      # How similar (lower = more similar)
            })
        
        return search_results
    
    def create_rag_agent(self):
        """
        Create an AI agent with RAG (Retrieval Augmented Generation) capabilities
        
        This agent can search your documents and generate informed responses.
        It's like having a smart assistant who has read all your documents.
        
        Returns:
            Agent that can answer questions about your documents
            
        Example:
            agent = rag.create_rag_agent()
            # Now you can ask the agent questions about your documents
        """
        
        @function_tool
        def search_knowledge_base(query: str) -> str:
            """
            Tool for the agent to search through documents
            
            This is a "function tool" - the AI agent can call this function
            when it needs to find information from your documents.
            
            Args:
                query: What to search for
                
            Returns:
                Formatted text with relevant information
                
            Example:
                When user asks "What's our vacation policy?", the agent will:
                1. Call search_knowledge_base("vacation policy")
                2. Get relevant document chunks
                3. Use those chunks to generate a comprehensive answer
            """
            # Search for relevant documents
            results = self.search_documents(query, n_results=3)
            
            if not results:
                return "No relevant information found in the knowledge base."
            
            # Format the results for the agent to use
            context = "Relevant information from knowledge base:\n\n"
            for i, result in enumerate(results, 1):
                context += f"Document {i} (from {result['metadata']['file_name']}):\n"
                context += f"{result['content']}\n\n"
            
            return context
        
        # Create the AI agent with RAG capabilities
        agent = Agent(
            name="rag_assistant",
            instructions="""You are a helpful AI assistant with access to a knowledge base of documents.
            
            When users ask questions:
            1. First, search the knowledge base using the search_knowledge_base tool
            2. Use the retrieved information to provide accurate, detailed answers
            3. Always cite which document your information comes from
            4. If the knowledge base doesn't contain relevant information, clearly state that
            5. Provide helpful and contextual responses based on the available information
            
            Be conversational and helpful while being accurate to the source material.
            
            Example interaction:
            User: "What's our company's remote work policy?"
            
            Your process:
            1. Call search_knowledge_base("remote work policy")
            2. Get relevant chunks from employee handbook
            3. Generate response like:
               "According to the employee handbook, our remote work policy allows..."
               [Source: employee_handbook.pdf]
            """,
            model=self.model,           # Use Llama 3.2 for responses
            tools=[search_knowledge_base]  # Give agent access to search function
        )
        
        return agent

async def main_async():
    """
    Async main function - sets up and runs the RAG system
    
    This demonstrates the complete workflow:
    1. Initialize the RAG system
    2. Process documents
    3. Create an AI agent
    4. Interactive chat loop
    
    Example session:
        $ python rag_system.py
        Initializing RAG System...
        Processing: company_manual.pdf
        Split into 15 chunks
        Successfully processed company_manual.pdf
        
        RAG System Ready!
        
        Ask a question: What's our vacation policy?
        Response: According to the company manual, employees are entitled to...
    """
    # Step 1: Initialize the RAG system
    print("Initializing RAG System...")
    
    # Try to find the books directory in different locations
    books_path = None
    possible_paths = [
        "./books",
        "./src/testprj/RAG_system/books", 
        "books",
        "RAG_system/books"
    ]
    
    for path in possible_paths:
        if Path(path).exists() and any(Path(path).iterdir()):
            books_path = path
            break
    
    if books_path:
        print(f"Using books directory: {books_path}")
        rag = RAGSystem(books_directory=books_path)
    else:
        print("Using default books directory: ./books")
        rag = RAGSystem()
    
    # Step 2: Process any new documents in the books folder
    # This only processes new files, skips already processed ones
    rag.scan_and_process_books()
    
    # Step 3: Create an AI agent with RAG capabilities
    agent = rag.create_rag_agent()
    
    # Step 4: Ready for questions!
    print("\n" + "="*50)
    print("RAG System Ready!")
    print("You can now ask questions about your documents.")
    print("Type 'quit' to exit, 'refresh' to reprocess documents")
    print("="*50 + "\n")
    
    # Step 5: Interactive chat loop
    while True:
        user_input = input("Ask a question: ").strip()
        
        # Handle special commands
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'refresh':
            print("Refreshing document database...")
            rag.scan_and_process_books()
            continue
        elif not user_input:
            continue
        
        # Process the user's question
        inputs = [{"content": user_input, "role": "user"}]
        
        # Run the agent to generate a response
        result = Runner.run_streamed(agent, input=inputs)
        
        # Stream the response as it's generated
        print("\nResponse: ", end="")
        async for event in result.stream_events():
            # Handle different types of streaming events for Ollama
            if isinstance(event, RawResponsesStreamEvent):
                if hasattr(event.data, 'delta') and event.data.delta:
                    print(event.data.delta, end="", flush=True)
                elif hasattr(event.data, 'content') and event.data.content:
                    print(event.data.content, end="", flush=True)
        
        print("\n" + "-"*50 + "\n")

def main():
    """
    Synchronous entry point for uv/pip script execution
    
    This function is called when you run:
    - uv run rag
    - python -m testprj.RAG_system.rag_system
    """
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nRAG System stopped by user.")
    except Exception as e:
        print(f"Error running RAG system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    """
    Entry point when script is run directly
    
    Usage:
        python rag_system.py
    """
    main()