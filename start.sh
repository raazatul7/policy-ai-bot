#!/bin/bash

# 🚀 AI Policy Query System - Simple Startup Script
# This script starts both frontend and backend servers

set -e  # Exit on any error

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

print_header() {
    echo -e "${CYAN}================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}================================${NC}"
}

# Function to check if port is in use
port_in_use() {
    lsof -ti:$1 >/dev/null 2>&1
}

# Function to kill processes on a port
kill_port() {
    if port_in_use $1; then
        print_warning "Port $1 is in use. Killing existing processes..."
        lsof -ti:$1 | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

# Function to check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Python
    if ! command -v python3 >/dev/null 2>&1; then
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    print_status "Python 3 found: $(python3 --version)"
    
    # Check virtual environment
    if [ ! -d "venv" ]; then
        print_error "Virtual environment not found. Please run setup first."
        print_status "Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
    print_status "Virtual environment found"
    
    # Check .env file
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating template..."
        echo "PERPLEXITY_API_KEY=your_perplexity_api_key_here" > .env
        echo "API_KEY=your_perplexity_api_key_here" >> .env
        echo "DEBUG=True" >> .env
        print_status "Created .env template. Please add your API key."
    else
        print_status ".env file found"
    fi
    
    # Check API key
    if ! grep -q "PERPLEXITY_API_KEY=" .env || grep -q "your_perplexity_api_key_here" .env; then
        print_warning "API key not configured. Please add your Perplexity API key to .env file"
    else
        print_status "API key configured"
    fi
}

# Function to start backend server
start_backend() {
    print_header "Starting Backend Server"
    
    # Kill any existing processes on port 8000
    kill_port 8000
    
    # Check if virtual environment is activated
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_status "Activating virtual environment..."
        source venv/bin/activate
    fi
    
    # Start backend server
    print_status "Starting backend server on http://localhost:8000"
    print_status "API Documentation: http://localhost:8000/docs"
    
    # Start in background - FIXED: Run the correct backend file
    cd backend && python3 main.py > ../logs/backend.log 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > ../.backend_pid
    cd ..
    # Ensure we're back in the project root
    cd /Users/atulgupta/Desktop/policy-ai-bot
    
    # Wait for backend to start
    print_status "Waiting for backend to start..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            print_status "✅ Backend server started successfully!"
            return 0
        fi
        sleep 1
    done
    
    print_error "Backend server failed to start"
    return 1
}

# Function to start frontend server
start_frontend() {
    print_header "Starting Frontend Server"
    
    # Kill any existing processes on port 3000
    kill_port 3000
    
    # Check if frontend files exist
    # Navigate to the project root directory
    cd /Users/atulgupta/Desktop/policy-ai-bot
    if [ ! -f "frontend/index.html" ]; then
        print_error "Frontend files not found"
        return 1
    fi
    
    # Start frontend server
    print_status "Starting frontend server on http://localhost:3000"
    
    # Start in background using simple HTTP server
    cd frontend && python3 -m http.server 3000 > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../.frontend_pid
    cd ..
    
    # Wait for frontend to start
    print_status "Waiting for frontend to start..."
    for i in {1..60}; do
        if curl -s http://localhost:3000 >/dev/null 2>&1; then
            print_status "✅ Frontend server started successfully!"
            return 0
        fi
        sleep 1
    done
    
    # Check if process is still running even if port check failed
    if [ -f ".frontend_pid" ]; then
        FRONTEND_PID=$(cat .frontend_pid)
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            print_status "✅ Frontend server started successfully! (process running)"
            return 0
        fi
    fi
    
    print_error "Frontend server failed to start"
    return 1
}

# Function to show status
show_status() {
    print_header "System Status"
    
    echo -e "${BLUE}Backend Server:${NC}"
    if port_in_use 8000; then
        echo -e "  ${GREEN}✅ Running on http://localhost:8000${NC}"
        echo -e "  📚 API Docs: http://localhost:8000/docs"
        echo -e "  💚 Health: http://localhost:8000/health"
    else
        echo -e "  ${RED}❌ Not running${NC}"
    fi
    
    echo -e "\n${BLUE}Frontend Server:${NC}"
    if port_in_use 3000; then
        echo -e "  ${GREEN}✅ Running on http://localhost:3000${NC}"
        echo -e "  🌐 Frontend: http://localhost:3000"
    else
        echo -e "  ${RED}❌ Not running${NC}"
    fi
    
    echo -e "\n${BLUE}Quick Access:${NC}"
    echo -e "  🌐 Main Frontend: http://localhost:3000"
    echo -e "  📚 API Documentation: http://localhost:8000/docs"
    echo -e "  💚 Health Check: http://localhost:8000/health"
    echo -e "  📊 System Stats: http://localhost:8000/stats"
}

# Function to stop all services
stop_all() {
    print_header "Stopping All Services"
    
    # Stop backend
    if [ -f ".backend_pid" ]; then
        BACKEND_PID=$(cat .backend_pid)
        if kill -0 $BACKEND_PID 2>/dev/null; then
            print_status "Stopping backend server (PID: $BACKEND_PID)"
            kill $BACKEND_PID
        fi
        rm -f .backend_pid
    fi
    
    # Stop frontend
    if [ -f ".frontend_pid" ]; then
        FRONTEND_PID=$(cat .frontend_pid)
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            print_status "Stopping frontend server (PID: $FRONTEND_PID)"
            kill $FRONTEND_PID
        fi
        rm -f .frontend_pid
    fi
    
    # Kill any remaining processes on ports
    kill_port 8000
    kill_port 3000
    
    print_status "All services stopped"
}

# Function to show help
show_help() {
    echo -e "${CYAN}AI Policy Query System - Simple Startup Script${NC}"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  start     Start both frontend and backend servers"
    echo "  stop      Stop all running services"
    echo "  status    Show current system status"
    echo "  restart   Restart all services"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start    # Start all services"
    echo "  $0 stop     # Stop all services"
    echo "  $0 status   # Check system status"
    echo ""
    echo "After starting, access:"
    echo "  🌐 Frontend: http://localhost:3000"
    echo "  📚 API Docs: http://localhost:8000/docs"
    echo "  💚 Health: http://localhost:8000/health"
}

# Main script logic
main() {
    # Create logs directory
    mkdir -p logs
    
    case "${1:-start}" in
        "start")
            print_header "🚀 Starting AI Policy Query System"
            check_prerequisites
            start_backend
            start_frontend
            show_status
            print_status "🎉 All services started successfully!"
            print_status "Access your application at:"
            print_status "  🌐 Frontend: http://localhost:3000"
            print_status "  📚 API Docs: http://localhost:8000/docs"
            print_status ""
            print_status "To stop all services, run: $0 stop"
            print_status "To check status, run: $0 status"
            ;;
        "stop")
            stop_all
            ;;
        "status")
            show_status
            ;;
        "restart")
            stop_all
            sleep 2
            $0 start
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

# No trap - let services run independently

# Run main function
main "$@" 