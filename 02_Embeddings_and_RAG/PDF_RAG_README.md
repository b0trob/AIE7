# PDF RAG System for RFC 1918

This is a simplified PDF RAG (Retrieval Augmented Generation) system designed to work with the RFC 1918 document (Address Allocation for Private Internets).

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key
- Required packages: `openai`, `PyPDF2`, `numpy`, `python-dotenv`

### Setup
1. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   Or create a `.env` file with:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

2. Make sure the RFC 1918 PDF is in the `data/` directory:
   ```
   data/RFC 1918_ Address Allocation for Private Internets.pdf
   ```

### Usage

#### 1. Test Basic PDF Processing (No API calls)
```bash
python3 test_pdf_basic.py
```
This tests PDF loading, text extraction, and chunking without making any API calls.

#### 2. Run Full RAG System (Interactive)
```bash
python3 interactive_pdf_rag.py
```
This starts an interactive Q&A session where you can ask questions about RFC 1918.

#### 3. Run Automated Tests
```bash
python3 pdf_rag_demo.py
```
This runs predefined test questions to demonstrate the system's capabilities.

## 📁 Files

- `pdf_rag_demo.py` - Complete RAG system implementation
- `interactive_pdf_rag.py` - Interactive Q&A interface
- `test_pdf_basic.py` - Basic PDF processing tests (no API calls)
- `PDF_RAG_README.md` - This file

## 🏗️ System Architecture

The system consists of several key components:

### 1. SimplePDFLoader
- Extracts text from PDF using PyPDF2
- Extracts metadata (pages, file size, document info)
- Avoids the import issues in the original codebase

### 2. SimpleTextSplitter
- Splits text into manageable chunks
- Configurable chunk size and overlap
- Optimized for RAG processing

### 3. SimpleEmbeddingModel
- Uses OpenAI's text-embedding-3-small model
- Handles API calls and error management
- Supports both sync and async operations

### 4. SimpleVectorDatabase
- Stores text chunks with their embeddings
- Implements cosine similarity search
- Supports metadata filtering

### 5. SimpleChatModel
- Uses OpenAI's GPT-4o-mini for responses
- Handles conversation formatting
- Error handling for API calls

### 6. PDFRAGSystem
- Orchestrates all components
- Provides high-level interface for Q&A
- Handles context retrieval and response generation

## 🎯 Example Questions

The system can answer questions like:
- "What are the private IP address ranges defined in RFC 1918?"
- "What is the purpose of private address allocation?"
- "How many private address blocks are defined?"
- "What are the specific address ranges for each private block?"
- "What is the difference between private and public IP addresses?"

## 🔧 Customization

### Adding New Distance Metrics
You can add new distance metrics to `SimpleVectorDatabase`:

```python
def euclidean_distance(self, vec_a: List[float], vec_b: List[float]) -> float:
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    return np.linalg.norm(vec_a - vec_b)
```

### Adding Metadata Support
The system already supports metadata. You can extend it by adding more metadata fields in the chunking process.

### Using Different PDFs
Simply change the PDF path in the initialization:
```python
rag_system = PDFRAGSystem("path/to/your/document.pdf")
```

## 🐛 Troubleshooting

### Import Errors
If you get import errors with the original `aimakerspace` module, this simplified version avoids those issues by using only standard libraries and PyPDF2.

### API Key Issues
Make sure your OpenAI API key is set correctly:
```bash
echo $OPENAI_API_KEY
```

### PDF Loading Issues
If the PDF won't load, check:
1. File exists in the correct location
2. File is not corrupted
3. PyPDF2 is installed: `pip install PyPDF2`

## 📊 Performance

- **PDF Processing**: ~1-2 seconds for RFC 1918 (9 pages)
- **Embedding Creation**: ~30-60 seconds for 34 chunks
- **Query Response**: ~5-10 seconds per question

## 🔄 Integration with Original Codebase

This simplified version can be easily integrated with the original `aimakerspace` codebase once the import issues are resolved. The interfaces are designed to be compatible.

## 📝 License

This is part of the AI Boot Camp curriculum and follows the same licensing terms.
