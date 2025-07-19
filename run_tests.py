#!/usr/bin/env python3
"""
Comprehensive Test Suite for AI Policy Query System

This script runs all tests including:
- API functionality and endpoints
- Document processing (PDF, DOCX, TXT)
- Perplexity API integration
- Edge cases and error handling
- System integration tests
"""

import os
import sys
import json
import time
import requests
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"🧪 {title}")
    print("=" * 60)

def print_success(message):
    """Print a success message."""
    print(f"✅ {message}")

def print_error(message):
    """Print an error message."""
    print(f"❌ {message}")

def print_warning(message):
    """Print a warning message."""
    print(f"⚠️ {message}")

def print_info(message):
    """Print an info message."""
    print(f"ℹ️ {message}")

def test_environment_setup():
    """Test environment and dependencies."""
    print_header("Environment Setup Test")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print_success(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print_error(f"Python version {python_version.major}.{python_version.minor} is too old. Need 3.8+")
        return False
    
    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_success("Virtual environment is active")
    else:
        print_warning("Virtual environment not detected")
    
    # Check API key
    api_key = os.getenv("PERPLEXITY_API_KEY") or os.getenv("API_KEY")
    if api_key and api_key != "your_perplexity_api_key_here":
        print_success("Perplexity API key found")
    else:
        print_error("Perplexity API key not configured")
        return False
    
    # Check required files
    required_files = [
        "backend/main.py",
        "backend/core/llm_reasoner.py",
        "backend/core/document_parser.py",
        "backend/core/embedder.py",
        "backend/core/retriever.py",
        "config/settings.py",
        "requirements.txt"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print_success(f"Found {file_path}")
        else:
            print_error(f"Missing {file_path}")
            return False
    
    return True

def test_perplexity_api():
    """Test Perplexity API integration."""
    print_header("Perplexity API Test")
    
    try:
        from backend.core.llm_reasoner import PolicyReasoner
        
        # Initialize reasoner
        reasoner = PolicyReasoner(
            use_perplexity=True,
            model_name="sonar-pro"
        )
        
        print_success("PolicyReasoner initialized successfully")
        
        # Test with simple query
        test_query = "What is the coverage limit for this policy?"
        test_context = "This policy provides coverage up to $100,000 for medical expenses."
        test_document = "Test Policy"
        
        print_info("Testing Perplexity API call...")
        
        response = reasoner.analyze_query(
            query=test_query,
            context=test_context,
            document_name=test_document
        )
        
        print_success("Perplexity API call successful!")
        print(f"   Decision: {response.decision}")
        print(f"   Justification: {response.justification}")
        print(f"   Reference: {response.reference}")
        
        return True
        
    except Exception as e:
        print_error(f"Perplexity API test failed: {str(e)}")
        return False

def test_document_parser():
    """Test document parsing functionality."""
    print_header("Document Parser Test")
    
    try:
        from backend.core.document_parser import DocumentParser
        
        parser = DocumentParser()
        print_success("DocumentParser initialized successfully")
        
        # Test with sample text file
        sample_content = """
        INSURANCE POLICY DOCUMENT
        
        Coverage Details:
        - Medical expenses: Up to $100,000
        - Deductible: $1,000
        - Co-pay: 20%
        
        Exclusions:
        - Pre-existing conditions
        - Cosmetic procedures
        
        Terms and Conditions:
        This policy is valid for one year from the effective date.
        """
        
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_content)
            temp_file = f.name
        
        try:
            chunks = parser.parse_document(temp_file)
            print_success(f"Document parsed successfully: {len(chunks)} chunks created")
            
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                print(f"   Chunk {i+1}: {chunk[:100]}...")
            
            return True
            
        finally:
            # Clean up
            os.unlink(temp_file)
        
    except Exception as e:
        print_error(f"Document parser test failed: {str(e)}")
        return False

def test_embedding_system():
    """Test embedding generation and storage."""
    print_header("Embedding System Test")
    
    try:
        from backend.core.embedder import EmbeddingManager
        
        # Initialize embedding manager
        embedding_manager = EmbeddingManager(
            model_name="all-MiniLM-L6-v2",
            use_openai=False
        )
        
        print_success("EmbeddingManager initialized successfully")
        
        # Test with sample text chunks
        test_chunks = [
            "This policy provides coverage up to $100,000 for medical expenses.",
            "The deductible is $1,000 and co-pay is 20%.",
            "Pre-existing conditions are excluded from coverage."
        ]
        
        print_info("Creating embeddings for test chunks...")
        embedding_manager.create_index(test_chunks)
        
        print_success(f"Embeddings created successfully for {len(test_chunks)} chunks")
        
        # Test similarity search
        query = "What is the coverage limit?"
        results = embedding_manager.search(query, top_k=2)
        
        print_success(f"Similarity search successful: {len(results)} results found")
        
        return True
        
    except Exception as e:
        print_error(f"Embedding system test failed: {str(e)}")
        return False

def test_retriever_system():
    """Test document retrieval functionality."""
    print_header("Retriever System Test")
    
    try:
        from backend.core.retriever import PolicyRetriever
        from backend.core.embedder import EmbeddingManager
        
        # Create embedding manager and index
        embedding_manager = EmbeddingManager(
            model_name="all-MiniLM-L6-v2",
            use_openai=False
        )
        
        test_chunks = [
            "This policy provides coverage up to $100,000 for medical expenses.",
            "The deductible is $1,000 and co-pay is 20%.",
            "Pre-existing conditions are excluded from coverage.",
            "The policy is valid for one year from the effective date."
        ]
        
        embedding_manager.create_index(test_chunks)
        
        # Initialize retriever
        retriever = PolicyRetriever(embedding_manager)
        print_success("PolicyRetriever initialized successfully")
        
        # Test context retrieval
        query = "What is the coverage limit and deductible?"
        context = retriever.get_context_for_query(query, max_context_length=500, top_k=3)
        
        print_success(f"Context retrieval successful: {len(context)} characters")
        print(f"   Retrieved context: {context[:200]}...")
        
        return True
        
    except Exception as e:
        print_error(f"Retriever system test failed: {str(e)}")
        return False

def test_system_integration():
    """Test full system integration."""
    print_header("System Integration Test")
    
    try:
        from backend.main import PolicyQuerySystem
        
        # Initialize system
        system = PolicyQuerySystem(
            embedding_model="all-MiniLM-L6-v2",
            use_openai_embeddings=False,
            use_perplexity_llm=True,
            llm_model="sonar-pro"
        )
        
        print_success("PolicyQuerySystem initialized successfully")
        
        # Create test document
        test_content = """
        INSURANCE POLICY DOCUMENT
        
        Coverage Details:
        - Medical expenses: Up to $100,000
        - Deductible: $1,000
        - Co-pay: 20%
        
        Exclusions:
        - Pre-existing conditions
        - Cosmetic procedures
        
        Terms and Conditions:
        This policy is valid for one year from the effective date.
        """
        
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_file = f.name
        
        try:
            # Test document processing
            print_info("Testing document processing...")
            result = system.process_document(temp_file, "test_policy.txt")
            
            if result['status'] == 'success':
                print_success("Document processing successful")
            else:
                print_error(f"Document processing failed: {result['message']}")
                return False
            
            # Test query answering
            print_info("Testing query answering...")
            response = system.answer_query("What is the coverage limit for medical expenses?")
            
            if 'decision' in response and 'justification' in response:
                print_success("Query answering successful")
                print(f"   Decision: {response['decision']}")
                print(f"   Justification: {response['justification'][:100]}...")
            else:
                print_error("Query answering failed")
                return False
            
            return True
            
        finally:
            # Clean up
            os.unlink(temp_file)
        
    except Exception as e:
        print_error(f"System integration test failed: {str(e)}")
        return False

def test_edge_cases():
    """Test edge cases and error handling."""
    print_header("Edge Cases Test")
    
    try:
        from backend.core.llm_reasoner import PolicyReasoner
        
        reasoner = PolicyReasoner(use_perplexity=True, model_name="sonar-pro")
        
        # Test 1: Empty context
        print_info("Testing empty context...")
        try:
            response = reasoner.analyze_query(
                query="What is covered?",
                context="",
                document_name="Empty Document"
            )
            print_success("Empty context handled gracefully")
        except Exception as e:
            print_warning(f"Empty context test: {str(e)}")
        
        # Test 2: Very long query
        print_info("Testing very long query...")
        long_query = "What is the coverage limit for medical expenses including hospital stays, doctor visits, prescription drugs, and any other medical services that might be required under this policy?" * 10
        
        try:
            response = reasoner.analyze_query(
                query=long_query,
                context="This policy covers medical expenses up to $100,000.",
                document_name="Test Document"
            )
            print_success("Long query handled successfully")
        except Exception as e:
            print_warning(f"Long query test: {str(e)}")
        
        # Test 3: Special characters in query
        print_info("Testing special characters...")
        special_query = "What's the coverage for pre-existing conditions? (urgent!)"
        
        try:
            response = reasoner.analyze_query(
                query=special_query,
                context="Pre-existing conditions are excluded from coverage.",
                document_name="Test Document"
            )
            print_success("Special characters handled successfully")
        except Exception as e:
            print_warning(f"Special characters test: {str(e)}")
        
        return True
        
    except Exception as e:
        print_error(f"Edge cases test failed: {str(e)}")
        return False

def test_api_endpoints():
    """Test API endpoints if server is running."""
    print_header("API Endpoints Test")
    
    base_url = "http://localhost:8000"
    
    # Check if server is running
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print_success("Server is running")
            
            # Test health endpoint
            health_data = response.json()
            print_success(f"Health check passed: {health_data.get('status', 'unknown')}")
            
            # Test root endpoint
            root_response = requests.get(f"{base_url}/", timeout=5)
            if root_response.status_code == 200:
                print_success("Root endpoint accessible")
            
            # Test stats endpoint
            stats_response = requests.get(f"{base_url}/stats", timeout=5)
            if stats_response.status_code == 200:
                print_success("Stats endpoint accessible")
            
            return True
        else:
            print_warning(f"Server responded with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_warning("Server is not running. Start with: ./start_simple.sh start")
        return False
    except Exception as e:
        print_error(f"API test failed: {str(e)}")
        return False

def run_all_tests():
    """Run all tests and provide summary."""
    print_header("AI Policy Query System - Comprehensive Test Suite")
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Perplexity API", test_perplexity_api),
        ("Document Parser", test_document_parser),
        ("Embedding System", test_embedding_system),
        ("Retriever System", test_retriever_system),
        ("System Integration", test_system_integration),
        ("Edge Cases", test_edge_cases),
        ("API Endpoints", test_api_endpoints)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {str(e)}")
            results.append((test_name, False))
    
    # Print summary
    print_header("Test Results Summary")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        if result:
            print_success(f"{test_name}: PASSED")
            passed += 1
        else:
            print_error(f"{test_name}: FAILED")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print_success("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print_warning(f"⚠️ {total - passed} test(s) failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 