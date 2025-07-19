#!/bin/bash

# 📊 AI Policy Query System - Log Viewer
# This script provides easy access to live server logs

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

print_header() {
    echo -e "${CYAN}================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}================================${NC}"
}

# Function to check if log files exist
check_logs() {
    if [ ! -f "logs/backend.log" ]; then
        print_warning "Backend log file not found. Server may not be running."
        return 1
    fi
    
    if [ ! -f "logs/frontend.log" ]; then
        print_warning "Frontend log file not found. Server may not be running."
        return 1
    fi
    
    return 0
}

# Function to show log file sizes
show_log_info() {
    print_header "Log File Information"
    
    if [ -f "logs/backend.log" ]; then
        BACKEND_SIZE=$(du -h logs/backend.log | cut -f1)
        BACKEND_LINES=$(wc -l < logs/backend.log)
        echo -e "${BLUE}Backend Log:${NC}"
        echo -e "  📁 File: logs/backend.log"
        echo -e "  📏 Size: $BACKEND_SIZE"
        echo -e "  📊 Lines: $BACKEND_LINES"
    else
        echo -e "${RED}Backend log not found${NC}"
    fi
    
    echo ""
    
    if [ -f "logs/frontend.log" ]; then
        FRONTEND_SIZE=$(du -h logs/frontend.log | cut -f1)
        FRONTEND_LINES=$(wc -l < logs/frontend.log)
        echo -e "${BLUE}Frontend Log:${NC}"
        echo -e "  📁 File: logs/frontend.log"
        echo -e "  📏 Size: $FRONTEND_SIZE"
        echo -e "  📊 Lines: $FRONTEND_LINES"
    else
        echo -e "${RED}Frontend log not found${NC}"
    fi
}

# Function to view backend logs
view_backend() {
    print_header "Backend Server Logs"
    
    if [ ! -f "logs/backend.log" ]; then
        print_error "Backend log file not found"
        return 1
    fi
    
    print_status "Showing live backend logs (Ctrl+C to exit)"
    print_status "Log file: logs/backend.log"
    echo ""
    
    tail -f logs/backend.log
}

# Function to view frontend logs
view_frontend() {
    print_header "Frontend Server Logs"
    
    if [ ! -f "logs/frontend.log" ]; then
        print_error "Frontend log file not found"
        return 1
    fi
    
    print_status "Showing live frontend logs (Ctrl+C to exit)"
    print_status "Log file: logs/frontend.log"
    echo ""
    
    tail -f logs/frontend.log
}

# Function to view both logs
view_both() {
    print_header "Both Server Logs"
    
    if [ ! -f "logs/backend.log" ] || [ ! -f "logs/frontend.log" ]; then
        print_error "One or both log files not found"
        return 1
    fi
    
    print_status "Showing live logs from both servers (Ctrl+C to exit)"
    echo ""
    
    # Use multitail if available, otherwise use a simple approach
    if command -v multitail >/dev/null 2>&1; then
        multitail -e "Backend" logs/backend.log -e "Frontend" logs/frontend.log
    else
        # Simple approach with different colors
        (tail -f logs/backend.log | sed 's/^/[BACKEND] /' & tail -f logs/frontend.log | sed 's/^/[FRONTEND] /' & wait)
    fi
}

# Function to view errors only
view_errors() {
    print_header "Error Logs Only"
    
    print_status "Showing only error messages from both logs (Ctrl+C to exit)"
    echo ""
    
    if [ -f "logs/backend.log" ] && [ -f "logs/frontend.log" ]; then
        tail -f logs/backend.log logs/frontend.log | grep -i error
    elif [ -f "logs/backend.log" ]; then
        tail -f logs/backend.log | grep -i error
    elif [ -f "logs/frontend.log" ]; then
        tail -f logs/frontend.log | grep -i error
    else
        print_error "No log files found"
    fi
}

# Function to clear logs
clear_logs() {
    print_header "Clear Log Files"
    
    read -p "Are you sure you want to clear all log files? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -f "logs/backend.log" ]; then
            > logs/backend.log
            print_status "Cleared backend log"
        fi
        
        if [ -f "logs/frontend.log" ]; then
            > logs/frontend.log
            print_status "Cleared frontend log"
        fi
        
        print_status "All logs cleared"
    else
        print_status "Log clearing cancelled"
    fi
}

# Function to show help
show_help() {
    echo -e "${CYAN}AI Policy Query System - Log Viewer${NC}"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  backend    View live backend server logs"
    echo "  frontend   View live frontend server logs"
    echo "  both       View both backend and frontend logs"
    echo "  errors     View only error messages"
    echo "  info       Show log file information"
    echo "  clear      Clear all log files"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 backend    # View backend logs"
    echo "  $0 both       # View both logs"
    echo "  $0 errors     # View only errors"
    echo ""
    echo "Quick Commands:"
    echo "  tail -f logs/backend.log     # Direct backend log viewing"
    echo "  tail -f logs/frontend.log    # Direct frontend log viewing"
    echo "  tail -f logs/*.log           # View all logs"
}

# Main script logic
main() {
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    case "${1:-help}" in
        "backend")
            view_backend
            ;;
        "frontend")
            view_frontend
            ;;
        "both")
            view_both
            ;;
        "errors")
            view_errors
            ;;
        "info")
            show_log_info
            ;;
        "clear")
            clear_logs
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