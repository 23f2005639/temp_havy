#!/bin/bash

# Quick Start Script for HMPI Calculator
# This script provides easy commands to set up and run the application

set -e  # Exit on any error

echo "🌊 HMPI Calculator - Quick Start"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Python is installed
check_python() {
    # Check for virtual environment first
    if [ -d ".venv" ]; then
        if [ -f ".venv/bin/python" ]; then
            # Check if venv has pip
            if .venv/bin/python -c "import pip" 2>/dev/null; then
                PYTHON_CMD=".venv/bin/python"
                print_info "Using virtual environment Python"
            else
                print_warning "Virtual environment found but pip is missing, using system Python"
                PYTHON_CMD="python3"
            fi
        else
            print_warning "Invalid virtual environment found, using system Python"
            PYTHON_CMD="python3"
        fi
    else
        # Use system Python
        if command -v python3 &> /dev/null; then
            PYTHON_CMD="python3"
        elif command -v python &> /dev/null; then
            PYTHON_CMD="python"
        else
            print_error "Python is not installed. Please install Python 3.8 or higher."
            exit 1
        fi
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    print_status "Found Python $PYTHON_VERSION"
}

# Create virtual environment if needed
create_venv() {
    if [ ! -d ".venv" ] || [ ! -f ".venv/bin/python" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv .venv
        print_status "Virtual environment created"
        PYTHON_CMD=".venv/bin/python"
    fi
}

# Install dependencies
install_deps() {
    print_info "Installing dependencies..."
    
    # If using system Python and no venv exists, create one
    if [[ "$PYTHON_CMD" == "python3" || "$PYTHON_CMD" == "python" ]]; then
        create_venv
    fi
    
    $PYTHON_CMD -m pip install --upgrade pip
    $PYTHON_CMD -m pip install -r requirements.txt
    print_status "Dependencies installed successfully"
}

# Run the application
run_app() {
    print_info "Starting HMPI Calculator..."
    print_info "The application will open at http://localhost:8501"
    print_warning "Press Ctrl+C to stop the application"
    $PYTHON_CMD -m streamlit run app.py
}

# Show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup     Install dependencies only"
    echo "  run       Run the application (install deps if needed)"
    echo "  clean     Clean up temporary files"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup     # Install dependencies"
    echo "  $0 run       # Start the application"
    echo "  $0           # Interactive mode"
}

# Clean temporary files
clean_up() {
    print_info "Cleaning up temporary files..."
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name ".DS_Store" -delete 2>/dev/null || true
    print_status "Cleanup completed"
}

# Main execution
main() {
    case "${1:-interactive}" in
        "setup")
            check_python
            install_deps
            print_status "Setup completed! Run '$0 run' to start the application."
            ;;
        "run")
            check_python
            if [ ! -d "__pycache__" ] && [ ! -f ".deps_installed" ]; then
                print_info "First time setup detected. Installing dependencies..."
                install_deps
                touch .deps_installed
            fi
            run_app
            ;;
        "clean")
            clean_up
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        "interactive")
            check_python
            echo ""
            echo "Select an option:"
            echo "1) Install dependencies and run application"
            echo "2) Install dependencies only"
            echo "3) Run application (assumes dependencies are installed)"
            echo "4) Clean temporary files"
            echo "5) Show help"
            echo ""
            read -p "Enter your choice (1-5): " choice
            
            case $choice in
                1)
                    install_deps
                    run_app
                    ;;
                2)
                    install_deps
                    print_status "Dependencies installed. Run '$0 run' to start the app."
                    ;;
                3)
                    run_app
                    ;;
                4)
                    clean_up
                    ;;
                5)
                    show_usage
                    ;;
                *)
                    print_error "Invalid choice. Please select 1-5."
                    exit 1
                    ;;
            esac
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Check if we're in the right directory
if [ ! -f "app.py" ] || [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the HMPI application directory"
    print_error "Make sure app.py and requirements.txt are present"
    exit 1
fi

# Run main function
main "$@"