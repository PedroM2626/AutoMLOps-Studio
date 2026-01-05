#!/bin/bash

# Free MLOps - Docker Build and Run Script
# This script builds and runs the Free MLOps platform using Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p data artifacts mlruns models
touch data/.gitkeep artifacts/.gitkeep mlruns/.gitkeep models/.gitkeep

# Copy .env file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file from template..."
    cp .env.example .env
    print_warning "Please review and update the .env file with your configuration."
fi

# Function to build Docker image
build_image() {
    print_status "Building Docker image..."
    docker build -t free-mlops:latest .
    print_success "Docker image built successfully!"
}

# Function to run with Docker Compose
run_compose() {
    print_status "Starting services with Docker Compose..."
    docker-compose up -d
    print_success "Services started successfully!"
    
    echo ""
    print_status "Services are available at:"
    echo "  🌐 Streamlit App: http://localhost:8501"
    echo "  📊 MLflow UI:   http://localhost:5000"
    echo "  🔌 API Docs:    http://localhost:8000/docs"
    echo ""
    print_status "To view logs: docker-compose logs -f"
    print_status "To stop:     docker-compose down"
}

# Function to run only Streamlit
run_streamlit_only() {
    print_status "Starting Streamlit app only..."
    docker run -d \
        --name free-mlops-streamlit \
        -p 8501:8501 \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/artifacts:/app/artifacts \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/free_mlops:/app/free_mlops \
        -e PYTHONPATH=/app \
        free-mlops:latest \
        streamlit run free_mlops/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
    
    print_success "Streamlit app started at http://localhost:8501"
}

# Function to run only MLflow
run_mlflow_only() {
    print_status "Starting MLflow UI only..."
    docker run -d \
        --name free-mlops-mlflow \
        -p 5000:5000 \
        -v $(pwd)/mlruns:/app/mlruns \
        -e PYTHONPATH=/app \
        free-mlops:latest \
        python -m mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
    
    print_success "MLflow UI started at http://localhost:5000"
}

# Function to run only API
run_api_only() {
    print_status "Starting API only..."
    docker run -d \
        --name free-mlops-api \
        -p 8000:8000 \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/artifacts:/app/artifacts \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/free_mlops:/app/free_mlops \
        -e PYTHONPATH=/app \
        -e API_HOST=0.0.0.0 \
        -e API_PORT=8000 \
        free-mlops:latest \
        python -m free_mlops.api
    
    print_success "API started at http://localhost:8000/docs"
}

# Function to stop all services
stop_services() {
    print_status "Stopping all services..."
    docker-compose down
    docker stop free-mlops-streamlit free-mlops-mlflow free-mlops-api 2>/dev/null || true
    docker rm free-mlops-streamlit free-mlops-mlflow free-mlops-api 2>/dev/null || true
    print_success "All services stopped!"
}

# Function to show logs
show_logs() {
    if [ -n "$1" ]; then
        docker-compose logs -f "$1"
    else
        docker-compose logs -f
    fi
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down -v --rmi all
    docker system prune -f
    print_success "Cleanup completed!"
}

# Main menu
case "$1" in
    "build")
        build_image
        ;;
    "run"|"start")
        build_image
        run_compose
        ;;
    "streamlit")
        build_image
        run_streamlit_only
        ;;
    "mlflow")
        build_image
        run_mlflow_only
        ;;
    "api")
        build_image
        run_api_only
        ;;
    "stop")
        stop_services
        ;;
    "logs")
        show_logs "$2"
        ;;
    "cleanup")
        cleanup
        ;;
    "restart")
        stop_services
        build_image
        run_compose
        ;;
    *)
        echo "Free MLOps - Docker Management Script"
        echo ""
        echo "Usage: $0 {build|run|streamlit|mlflow|api|stop|logs|cleanup|restart}"
        echo ""
        echo "Commands:"
        echo "  build     - Build Docker image"
        echo "  run/start - Build and run all services with Docker Compose"
        echo "  streamlit - Run only Streamlit app"
        echo "  mlflow    - Run only MLflow UI"
        echo "  api       - Run only FastAPI"
        echo "  stop      - Stop all services"
        echo "  logs      - Show logs (optional: specify service name)"
        echo "  cleanup   - Clean up all Docker resources"
        echo "  restart   - Stop and restart all services"
        echo ""
        echo "Examples:"
        echo "  $0 run           # Start all services"
        echo "  $0 streamlit     # Start only Streamlit"
        echo "  $0 logs app      # Show app logs"
        echo "  $0 stop          # Stop all services"
        exit 1
        ;;
esac
