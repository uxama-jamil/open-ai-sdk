# TestPrj - Multi-Agent RAG System

A powerful multi-agent system with Retrieval Augmented Generation (RAG) capabilities, featuring language routing agents and intelligent document search using Llama 3.2 and ChromaDB.

## 🚀 Features

### Multi-Agent Language Support
- **Triage Agent**: Routes conversations to language-specific agents
- **Language Agents**: French, Spanish, and English specialized agents
- **Language Switching**: Easy switching between languages during conversation

### RAG (Retrieval Augmented Generation)
- **Local Vector Database**: ChromaDB for fast, persistent storage
- **Free Embeddings**: all-MiniLM-L6-v2 model for high-quality semantic search
- **Multi-format Support**: PDF, DOCX, TXT, MD files
- **Intelligent Chunking**: Optimal text splitting with overlap
- **Source Attribution**: Always cites document sources
- **Incremental Processing**: Only processes new/changed files

### Privacy & Security
- **100% Local**: All processing happens on your machine
- **No API Calls**: Embeddings generated locally
- **Confidential Documents**: Perfect for sensitive information

## 📋 Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai/) with Llama 3.2 model

## 🛠️ Installation

### 1. Install Ollama and Llama 3.2

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Llama 3.2 model
ollama pull llama3.2:latest

# Start Ollama server
ollama serve
```

### 2. Clone and Setup Project

```bash
# Clone your project
git clone <your-repo-url>
cd testprj

# Install dependencies with uv
uv sync

# Alternative: Install with specific groups
uv sync --group dev  # Include development dependencies
```

### 3. Create Required Directories

```bash
mkdir -p books chroma_db
```

Your project structure should look like:
```
testprj/
├── src/testprj/
│   ├── rag_system.py
│   ├── simpleAgent.py
│   └── toolCalling.py
├── books/                 # Put your documents here
├── chroma_db/            # Vector database (auto-created)
├── pyproject.toml
├── README.md
└── .env                  # Optional environment variables
```

## 📚 RAG System Usage

### Adding Documents

Place your documents in the `books/` directory:

```bash
books/
├── company_manual.pdf
├── technical_specs.docx
├── research_notes.txt
├── policies.md
└── confidential_docs.pdf
```

**Supported formats:**
- PDF (`.pdf`)
- Word Documents (`.docx`)
- Text files (`.txt`)
- Markdown (`.md`)

### Running the RAG System

```bash
# Using the script shortcut
uv run rag

# Or directly
uv run python src/testprj/rag_system.py
```

### First Run - Document Processing

On first run, the system will:
1. 🔍 Scan the `books/` directory
2. 📄 Extract text from all supported files
3. ✂️ Split text into optimal chunks (1000 chars with 200 overlap)
4. 🧠 Generate embeddings using sentence-transformer
5. 💾 Store everything in ChromaDB

### Asking Questions

Once processing is complete:

```
Ask a question: What are the main security policies mentioned in our documents?

Response: Based on the company manual and policy documents, the main security policies include:

1. **Password Requirements**: Minimum 12 characters with complexity requirements
2. **Two-Factor Authentication**: Mandatory for all system access
3. **Data Classification**: Documents must be classified as Public, Internal, or Confidential
...

[Sources: company_manual.pdf, policies.md]
```

### Available Commands

- **Normal questions**: Just type your question
- **`refresh`**: Reprocess all documents (after adding new files)
- **`quit`**: Exit the system

## 🌐 Multi-Language Agents

### Running Language Routing System

```bash
# Using other agent scripts
uv run llm    # Simple agent
uv run tool   # Tool calling examples
```

### Language Switching Example

```
User: "Hello, I need help"
→ Routes to English Agent

User: "switch language"
→ Returns to Triage Agent

User: "Bonjour"
→ Routes to French Agent

User: "cambiar idioma"
→ Returns to Triage Agent
```

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Optional: Customize Ollama settings
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.2:latest

# Optional: RAG settings
BOOKS_DIRECTORY=./books
CHROMA_DB_PATH=./chroma_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Customizing Chunk Size

Edit `rag_system.py`:

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,      # Larger chunks for more context
    chunk_overlap=300,    # More overlap for better continuity
    length_function=len,
)
```

### Different Embedding Models

```python
# Higher quality (slower)
self.embedding_model = SentenceTransformer('all-mpnet-base-v2')

# Faster processing (lower quality)
self.embedding_model = SentenceTransformer('all-MiniLM-L12-v2')

# Multilingual support
self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

## 🔧 Development

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black src/
uv run flake8 src/
uv run mypy src/
```

### Adding New Dependencies

```bash
# Add regular dependency
uv add package-name

# Add development dependency
uv add --group dev package-name
```

## 📖 Example Use Cases

### 1. Technical Documentation
```
Q: "How do I configure the authentication system?"
A: Based on the technical specifications, authentication configuration involves...
[Source: technical_specs.docx]
```

### 2. Company Policies
```
Q: "What's our remote work policy?"
A: According to the employee handbook, remote work is permitted with...
[Source: company_manual.pdf]
```

### 3. Research Analysis
```
Q: "What methodologies were used in the climate studies?"
A: The research documents mention several methodologies including...
[Source: research_notes.txt]
```

### 4. Legal Documents
```
Q: "What are the termination clauses in our contracts?"
A: The legal documents specify termination conditions as follows...
[Source: legal_contracts.pdf]
```

## 🚨 Troubleshooting

### PDF Reading Issues
If PyPDF2 can't read certain PDFs:
```bash
uv add pypdf pdfplumber
```

### Memory Issues with Large Documents
- Reduce chunk size: `chunk_size=500`
- Use smaller embedding model
- Process documents in batches

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve

# Check available models
ollama list
```

### ChromaDB Issues
```bash
# Clear and rebuild database
rm -rf chroma_db/
# Run system again to rebuild
```

## 🔒 Security & Privacy

- ✅ **All Local Processing**: No data sent to external APIs
- ✅ **Local Embeddings**: Sentence-transformers runs locally
- ✅ **Local LLM**: Ollama runs Llama 3.2 locally
- ✅ **Local Storage**: ChromaDB stores everything on your machine
- ✅ **No Internet Required**: After initial model downloads

Perfect for:
- Confidential company documents
- Personal research notes
- Legal documents
- Medical records
- Financial information

## 📄 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📞 Support

[Add support information here]