"""
Main FastAPI Application for AI Policy Query System

This is the central API server that handles file uploads, document processing,
and natural language queries about insurance policy documents.
"""

import os
import tempfile
import uuid
import shutil
import gc
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
import re

# Import our custom modules
from core.document_parser import DocumentParser
from core.embedder import EmbeddingManager
from core.retriever import PolicyRetriever
from core.llm_reasoner import PolicyReasoner


def check_memory_usage():
    """Check if system has enough memory available."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.available < 500 * 1024 * 1024:  # 500MB
            gc.collect()  # Force garbage collection
            if memory.available < 500 * 1024 * 1024:
                raise RuntimeError("Insufficient memory available")
        return True
    except ImportError:
        # psutil not available, skip memory check
        return True


def check_disk_space(path: str, required_mb: int = 100):
    """Check if there's enough disk space."""
    try:
        total, used, free = shutil.disk_usage(path)
        if free < required_mb * 1024 * 1024:
            raise RuntimeError(f"Insufficient disk space. Need {required_mb}MB")
        return True
    except Exception as e:
        raise RuntimeError(f"Error checking disk space: {str(e)}")


# Initialize FastAPI app
app = FastAPI(
    title="AI Policy Query System",
    description="An AI-powered system for analyzing insurance policy documents and answering natural language questions",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add TrustedHostMiddleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] # Allow all hosts for development
)

# Global storage for active sessions
# In production, use a proper database or session management
active_sessions = {}

# Data directory for storing uploaded documents
DATA_DIR = Path("../data")
DATA_DIR.mkdir(exist_ok=True)

# Document metadata storage
DOCUMENTS_FILE = DATA_DIR / "documents.json"

def load_documents():
    """Load document metadata from file."""
    if DOCUMENTS_FILE.exists():
        try:
            with open(DOCUMENTS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_documents(documents):
    """Save document metadata to file."""
    try:
        with open(DOCUMENTS_FILE, 'w') as f:
            json.dump(documents, f, indent=2)
    except Exception as e:
        print(f"Error saving documents: {e}")

# Load existing documents
uploaded_documents = load_documents()


def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        text: Raw input text
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', text)
    
    # Limit length
    if len(sanitized) > 1000:
        sanitized = sanitized[:1000]
    
    return sanitized.strip()


class PolicyQuerySystem:
    """
    Main system class that orchestrates document processing and query answering.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_openai_embeddings: bool = False,
        use_perplexity_llm: bool = True,
        llm_model: str = "sonar-pro"
    ):
        """
        Initialize the policy query system.
        
        Args:
            embedding_model: Name of the embedding model
            use_openai_embeddings: Whether to use OpenAI for embeddings
            use_perplexity_llm: Whether to use Perplexity for LLM reasoning
            llm_model: Name of the Perplexity model to use
        """
        self.embedding_model = embedding_model
        self.use_openai_embeddings = use_openai_embeddings
        self.use_perplexity_llm = use_perplexity_llm
        self.llm_model = llm_model
        
        # Initialize components
        self.document_parser = DocumentParser()
        self.retriever = None
        self.reasoner = PolicyReasoner(use_perplexity=use_perplexity_llm, model_name=llm_model)
        
        # Document metadata
        self.document_metadata = {}
    
    def process_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Process a policy document and create embeddings.
        
        Args:
            file_path: Path to the uploaded file
            filename: Original filename
            
        Returns:
            Processing results and metadata
        """
        try:
            # Parse the document into chunks
            print(f"Parsing document: {filename}")
            text_chunks = self.document_parser.parse_document(file_path)
            
            if not text_chunks:
                raise ValueError("No text content found in the document")
            
            # Create embeddings and retriever
            print(f"Creating embeddings for {len(text_chunks)} chunks...")
            embedding_manager = EmbeddingManager(
                model_name=self.embedding_model,
                use_openai=self.use_openai_embeddings
            )
            embedding_manager.create_index(text_chunks)
            
            # Initialize retriever
            self.retriever = PolicyRetriever(embedding_manager)
            
            # Store document metadata
            self.document_metadata = {
                'filename': filename,
                'total_chunks': len(text_chunks),
                'embedding_model': self.embedding_model,
                'processing_status': 'completed'
            }
            
            return {
                'status': 'success',
                'message': f'Document processed successfully',
                'metadata': self.document_metadata
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error processing document: {str(e)}',
                'metadata': {}
            }
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Answer a query about the processed document.
        
        Args:
            query: Natural language question
            
        Returns:
            Structured response with decision, justification, and reference
        """
        if not self.retriever:
            raise ValueError("No document has been processed. Please upload a document first.")
        
        try:
            # Retrieve relevant context
            print(f"Retrieving context for query: {query}")
            context = self.retriever.get_context_for_query(query, max_context_length=2000, top_k=3)
            
            # Generate response using LLM
            print("Generating response using LLM...")
            response = self.reasoner.analyze_query(
                query=query,
                context=context,
                document_name=self.document_metadata.get('filename', 'Policy Document')
            )
            
            return response.to_dict()
            
        except Exception as e:
            return {
                'decision': 'Error processing query',
                'justification': f'An error occurred while processing your question: {str(e)}',
                'reference': 'System Error'
            }


# Initialize the global system instance (use Perplexity for LLM)
policy_system = PolicyQuerySystem(
    use_openai_embeddings=False,
    use_perplexity_llm=bool(os.getenv("API_KEY"))
)


@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "message": "AI Policy Query System API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "upload_and_ask": "POST /ask",
            "health": "GET /health",
            "stats": "GET /stats"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with system resource information."""
    try:
        # Get system resource information
        memory_info = {}
        disk_info = {}
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_info = {
                "total": f"{memory.total / 1024 / 1024 / 1024:.1f}GB",
                "available": f"{memory.available / 1024 / 1024:.1f}MB",
                "percent_used": f"{memory.percent:.1f}%"
            }
        except ImportError:
            memory_info = {"status": "psutil not available"}
        
        try:
            total, used, free = shutil.disk_usage(".")
            disk_info = {
                "total": f"{total / 1024 / 1024 / 1024:.1f}GB",
                "free": f"{free / 1024 / 1024 / 1024:.1f}GB",
                "percent_free": f"{(free / total) * 100:.1f}%"
            }
        except Exception as e:
            disk_info = {"error": str(e)}
        
        return {
            "status": "healthy",
            "system_ready": True,
            "components": {
                "document_parser": "active",
                "embedding_system": "active",
                "retriever": "active" if policy_system.retriever else "inactive",
                "llm_reasoner": "active"
            },
            "system_resources": {
                "memory": memory_info,
                "disk": disk_info
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "system_ready": False
        }


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    stats = {
        "embedding_model": policy_system.embedding_model,
        "llm_model": policy_system.llm_model,
        "document_metadata": policy_system.document_metadata,
        "active_sessions": len(active_sessions),
        "uploaded_documents": len(uploaded_documents)
    }
    
    if policy_system.retriever:
        retrieval_stats = policy_system.retriever.get_retrieval_stats()
        stats["retrieval_stats"] = retrieval_stats
    
    return stats


@app.get("/documents")
async def get_documents():
    """Get list of uploaded documents."""
    return {
        "documents": list(uploaded_documents.values()),
        "total": len(uploaded_documents)
    }


@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get specific document metadata."""
    if document_id not in uploaded_documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    return uploaded_documents[document_id]


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document."""
    if document_id not in uploaded_documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    try:
        # Remove file
        doc_info = uploaded_documents[document_id]
        file_path = DATA_DIR / doc_info['filename']
        if file_path.exists():
            file_path.unlink()
        
        # Remove from metadata
        del uploaded_documents[document_id]
        save_documents(uploaded_documents)
        
        return {"message": "Document deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )


@app.post("/upload")
async def upload_document(document: UploadFile = File(...)):
    """
    Upload a document for later use.
    
    This endpoint saves the document to the data folder and returns metadata
    without processing it immediately.
    """
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    file_extension = Path(document.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_extension}. Supported types: {', '.join(allowed_extensions)}"
        )
    
    # Check file size (100MB limit)
    content = await document.read()
    if len(content) > 100 * 1024 * 1024:  # 100MB
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large (max 100MB)"
        )
    
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{document.filename}"
        file_path = DATA_DIR / safe_filename
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Create document metadata
        document_id = str(uuid.uuid4())
        doc_info = {
            "id": document_id,
            "original_filename": document.filename,
            "filename": safe_filename,
            "file_type": file_extension,
            "file_size": len(content),
            "upload_date": datetime.now().isoformat(),
            "status": "uploaded"
        }
        
        # Save metadata
        uploaded_documents[document_id] = doc_info
        save_documents(uploaded_documents)
        
        return {
            "message": "Document uploaded successfully",
            "document_id": document_id,
            "document_info": doc_info
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )


@app.post("/process/{document_id}")
async def process_document(document_id: str):
    """
    Process a previously uploaded document.
    """
    if document_id not in uploaded_documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    doc_info = uploaded_documents[document_id]
    file_path = DATA_DIR / doc_info['filename']
    
    # Check if document is already processed
    if doc_info.get('status') == 'processed':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document is already processed. You can now ask questions about this document."
        )
    
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document file not found"
        )
    
    try:
        # Process the document
        processing_result = policy_system.process_document(str(file_path), doc_info['original_filename'])
        
        if processing_result['status'] != 'success':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=processing_result['message']
            )
        
        # Update document status
        doc_info['status'] = 'processed'
        doc_info['processed_date'] = datetime.now().isoformat()
        doc_info['chunks'] = processing_result['metadata'].get('total_chunks', 0)
        uploaded_documents[document_id] = doc_info
        save_documents(uploaded_documents)
        
        return {
            "message": "Document processed successfully",
            "document_id": document_id,
            "document_info": doc_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )


@app.post("/ask")
async def upload_and_ask(
    document: UploadFile = File(...),
    query: str = Form(...)
):
    """
    Main endpoint that accepts a document upload and a query, then returns a structured answer.
    
    This endpoint:
    1. Accepts a policy document (PDF or DOCX)
    2. Processes the document and creates embeddings
    3. Answers the provided query based on the document content
    4. Returns a structured JSON response
    """
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    file_extension = Path(document.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_extension}. Supported types: {', '.join(allowed_extensions)}"
        )
    
    # Validate query
    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    
    # Sanitize query input
    query = sanitize_input(query)
    
    # Validate query length
    if len(query) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query too long (max 1000 characters)"
        )
    
    # Create temporary file for processing
    temp_file = None
    temp_file_path = None
    try:
        # Check system resources before processing
        check_memory_usage()
        check_disk_space(".", 100)  # Need 100MB for processing
        
        # Read file content
        content = await document.read()
        
        # Save uploaded file temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Also save to data folder for persistence
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{document.filename}"
        file_path = DATA_DIR / safe_filename
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Create document metadata
        document_id = str(uuid.uuid4())
        doc_info = {
            "id": document_id,
            "original_filename": document.filename,
            "filename": safe_filename,
            "file_type": file_extension,
            "file_size": len(content),
            "upload_date": datetime.now().isoformat(),
            "status": "uploaded"
        }
        
        # Save metadata
        uploaded_documents[document_id] = doc_info
        save_documents(uploaded_documents)
        
        # Process the document
        processing_result = policy_system.process_document(temp_file_path, document.filename)
        
        if processing_result['status'] != 'success':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=processing_result['message']
            )
        
        # Answer the query
        answer = policy_system.answer_query(query)
        
        # Update document status to processed
        doc_info['status'] = 'processed'
        doc_info['processed_date'] = datetime.now().isoformat()
        doc_info['chunks'] = processing_result['metadata'].get('total_chunks', 0)
        uploaded_documents[document_id] = doc_info
        save_documents(uploaded_documents)
        
        # Add metadata to the response
        response = {
            **answer,
            "document_id": document_id,
            "document_info": {
                "filename": document.filename,
                "file_type": file_extension,
                "processed_chunks": processing_result['metadata'].get('total_chunks', 0),
                "document_id": document_id
            },
            "query": query
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass  # Ignore cleanup errors


@app.post("/ask-existing")
async def ask_existing_document(
    query: str = Form(...),
    document_id: str = Form(...)
):
    """
    Ask a question about a previously uploaded document.
    
    This endpoint allows querying without re-uploading the document,
    useful for asking multiple questions about the same policy.
    """
    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    
    if not document_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document ID is required"
        )
    
    # Check if document exists
    if document_id not in uploaded_documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    doc_info = uploaded_documents[document_id]
    
    # Check if document is processed
    if doc_info.get('status') != 'processed':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document is not processed. Please process the document first using the /process endpoint."
        )
    
    file_path = DATA_DIR / doc_info['filename']
    
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document file not found"
        )
    
    try:
        # Load the document into the policy system
        processing_result = policy_system.process_document(str(file_path), doc_info['original_filename'])
        
        if processing_result['status'] != 'success':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error loading document: {processing_result['message']}"
            )
        
        # Answer the query
        answer = policy_system.answer_query(query)
        
        response = {
            **answer,
            "document_info": {
                "filename": doc_info['original_filename'],
                "file_type": doc_info['file_type'],
                "processed_chunks": doc_info.get('chunks', 0),
                "document_id": document_id
            },
            "query": query
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "message": "The requested endpoint does not exist"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "An unexpected error occurred"}
    )


# Development server runner
if __name__ == "__main__":
    # Configuration for development
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,
        "log_level": "info"
    }
    
    print("🚀 Starting AI Policy Query System API...")
    print(f"📍 Server will be available at: http://localhost:{config['port']}")
    print(f"📋 API Documentation: http://localhost:{config['port']}/docs")
    print(f"🔍 Interactive API: http://localhost:{config['port']}/redoc")
    
    # Check for required environment variables
    if not os.getenv("API_KEY"):
        print("⚠️  Warning: API_KEY not found in environment variables")
        print("   Set it if you plan to use Perplexity models for LLM reasoning")
    
    # Start the server
    uvicorn.run("main:app", **config) 