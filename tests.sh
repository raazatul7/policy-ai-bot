#!/bin/bash

# 🧪 AI Policy Query System - Test Runner Script
# This script runs the comprehensive test suite with proper environment setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_header() {
    echo -e "${CYAN}================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}================================${NC}"
}

# Function to check if virtual environment is active
check_venv() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        if [ -d "venv" ]; then
            print_status "Activating virtual environment..."
            source venv/bin/activate
        else
            print_error "Virtual environment not found. Please run setup first."
            print_status "Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
            exit 1
        fi
    else
        print_success "Virtual environment is active"
    fi
}

# Function to check dependencies
check_dependencies() {
    print_header "Checking Dependencies"
    
    # Check if requirements are installed
    if ! python3 -c "import fastapi, uvicorn, pdfplumber, sentence_transformers, faiss" 2>/dev/null; then
        print_warning "Some dependencies missing. Installing requirements..."
        pip install -r requirements.txt
    else
        print_success "All dependencies are installed"
    fi
}

# Function to check API key
check_api_key() {
    print_header "Checking API Configuration"
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating template..."
        echo "PERPLEXITY_API_KEY=your_perplexity_api_key_here" > .env
        echo "API_KEY=your_perplexity_api_key_here" >> .env
        echo "DEBUG=True" >> .env
        print_status "Created .env template. Please add your API key."
        return 1
    fi
    
    # Check if API key is configured
    if ! grep -q "PERPLEXITY_API_KEY=" .env || grep -q "your_perplexity_api_key_here" .env; then
        print_warning "API key not configured. Please add your Perplexity API key to .env file"
        return 1
    else
        print_success "API key configured"
        return 0
    fi
}

# Function to run tests
run_tests() {
    print_header "Running Comprehensive Test Suite"
    
    # Check if test file exists
    if [ ! -f "run_tests.py" ]; then
        print_error "Test file run_tests.py not found"
        exit 1
    fi
    
    print_status "Starting test suite..."
    echo ""
    
    # Run the test suite
    if python3 run_tests.py; then
        print_success "🎉 All tests completed successfully!"
        return 0
    else
        print_error "❌ Some tests failed. Check the output above for details."
        return 1
    fi
}

# Function to run specific test categories
run_specific_test() {
    local test_type=$1
    
    print_header "Running $test_type Test"
    
    case $test_type in
        "api")
            print_status "Testing Perplexity API integration..."
            python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()
from backend.core.llm_reasoner import PolicyReasoner
reasoner = PolicyReasoner(use_perplexity=True, model_name='sonar-pro')
response = reasoner.analyze_query('What is covered?', 'This policy covers medical expenses up to $100,000.', 'Test')
print('✅ API test successful:', response.decision)
"
            ;;
        "parser")
            print_status "Testing document parser..."
            python3 -c "
from backend.core.document_parser import DocumentParser
import tempfile
parser = DocumentParser()
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write('Test policy document content.')
    temp_file = f.name
try:
    chunks = parser.parse_document(temp_file)
    print('✅ Parser test successful:', len(chunks), 'chunks created')
finally:
    import os; os.unlink(temp_file)
"
            ;;
        "embedding")
            print_status "Testing embedding system..."
            python3 -c "
from backend.core.embedder import EmbeddingManager
embedding_manager = EmbeddingManager(model_name='all-MiniLM-L6-v2', use_openai=False)
test_chunks = ['Test chunk 1', 'Test chunk 2']
embedding_manager.create_index(test_chunks)
results = embedding_manager.search('test', top_k=1)
print('✅ Embedding test successful:', len(results), 'results found')
"
            ;;
        "integration")
            print_status "Testing system integration..."
            python3 -c "
from backend.main import PolicyQuerySystem
import tempfile
system = PolicyQuerySystem(use_perplexity_llm=True, llm_model='sonar-pro')
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write('Test policy content.')
    temp_file = f.name
try:
    result = system.process_document(temp_file, 'test.txt')
    print('✅ Integration test successful:', result['status'])
finally:
    import os; os.unlink(temp_file)
"
            ;;
        *)
            print_error "Unknown test type: $test_type"
            print_status "Available test types: api, parser, embedding, integration"
            return 1
            ;;
    esac
}

# Function to show help
show_help() {
    echo -e "${CYAN}AI Policy Query System - Test Runner${NC}"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  all              Run all tests (default)"
    echo "  api              Test Perplexity API integration"
    echo "  parser           Test document parser"
    echo "  embedding        Test embedding system"
    echo "  integration      Test system integration"
    echo "  quick            Run quick health check"
    echo "  help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0               # Run all tests"
    echo "  $0 api           # Test API only"
    echo "  $0 quick         # Quick health check"
    echo ""
    echo "Environment Setup:"
    echo "  The script will automatically:"
    echo "  - Activate virtual environment"
    echo "  - Check dependencies"
    echo "  - Verify API configuration"
    echo "  - Run comprehensive tests"
}

# Function to run quick health check
run_quick_check() {
    print_header "Quick Health Check"
    
    # Check Python
    if command -v python3 >/dev/null 2>&1; then
        print_success "Python 3 found: $(python3 --version)"
    else
        print_error "Python 3 not found"
        return 1
    fi
    
    # Check virtual environment
    check_venv
    
    # Check key files
    key_files=("run_tests.py" "backend/main.py" "requirements.txt")
    for file in "${key_files[@]}"; do
        if [ -f "$file" ]; then
            print_success "Found $file"
        else
            print_error "Missing $file"
            return 1
        fi
    done
    
    # Check API key
    if check_api_key; then
        print_success "API key configured"
    else
        print_warning "API key not configured"
    fi
    
    # Quick import test
    if python3 -c "from backend.core.llm_reasoner import PolicyReasoner; print('✅ Core modules import successfully')" 2>/dev/null; then
        print_success "Core modules import successfully"
    else
        print_error "Core modules import failed"
        return 1
    fi
    
    print_success "Quick health check passed!"
    return 0
}

# Main script logic
main() {
    case "${1:-all}" in
        "all")
            print_header "🚀 AI Policy Query System - Comprehensive Test Suite"
            check_venv
            check_dependencies
            check_api_key
            run_tests
            ;;
        "api"|"parser"|"embedding"|"integration")
            check_venv
            check_dependencies
            check_api_key
            run_specific_test "$1"
            ;;
        "quick")
            run_quick_check
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 