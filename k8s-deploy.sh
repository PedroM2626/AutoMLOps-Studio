#!/bin/bash

# Free MLOps - Kubernetes Deployment Script
# This script deploys the Free MLOps platform to Kubernetes

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

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
    exit 1
fi

# Configuration
NAMESPACE="free-mlops"
DOCKER_IMAGE="free-mlops:latest"

# Function to build and push Docker image
build_and_push() {
    print_status "Building Docker image..."
    docker build -t $DOCKER_IMAGE .
    
    # For local clusters (minikube, kind, etc.)
    if command -v minikube &> /dev/null && minikube status &> /dev/null; then
        print_status "Loading image into minikube..."
        minikube image load $DOCKER_IMAGE
    elif command -v kind &> /dev/null; then
        print_status "Loading image into kind cluster..."
        kind load docker-image $DOCKER_IMAGE
    else
        print_warning "Please push the image to your registry:"
        echo "docker tag $DOCKER_IMAGE your-registry/$DOCKER_IMAGE"
        echo "docker push your-registry/$DOCKER_IMAGE"
        echo "Then update the image in k8s/*.yaml files"
    fi
    
    print_success "Docker image prepared!"
}

# Function to deploy to Kubernetes
deploy_k8s() {
    print_status "Deploying to Kubernetes..."
    
    # Apply namespace
    kubectl apply -f k8s/namespace.yaml
    
    # Apply persistent volumes
    kubectl apply -f k8s/persistent-volumes.yaml
    
    # Apply configmap
    kubectl apply -f k8s/configmap.yaml
    
    # Deploy applications
    kubectl apply -f k8s/app-deployment.yaml
    kubectl apply -f k8s/mlflow-deployment.yaml
    kubectl apply -f k8s/api-deployment.yaml
    
    # Apply ingress (optional)
    if [ "$1" = "--ingress" ]; then
        kubectl apply -f k8s/ingress.yaml
    fi
    
    print_success "Deployment completed!"
}

# Function to check deployment status
check_status() {
    print_status "Checking deployment status..."
    
    echo ""
    kubectl get pods -n $NAMESPACE
    echo ""
    
    print_status "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod -l app=free-mlops -n $NAMESPACE --timeout=300s
    
    echo ""
    print_status "Services:"
    kubectl get services -n $NAMESPACE
    
    if [ "$1" = "--ingress" ]; then
        echo ""
        print_status "Ingress:"
        kubectl get ingress -n $NAMESPACE
    fi
}

# Function to get access URLs
get_urls() {
    print_status "Getting access URLs..."
    
    # For minikube
    if command -v minikube &> /dev/null && minikube status &> /dev/null; then
        echo ""
        print_status "Minikube URLs:"
        echo "  🌐 Streamlit: $(minikube service free-mlops-app-service -n $NAMESPACE --url)"
        echo "  📊 MLflow:    $(minikube service free-mlops-mlflow-service -n $NAMESPACE --url)"
        echo "  🔌 API:       $(minikube service free-mlops-api-service -n $NAMESPACE --url)"
    
    # For LoadBalancer services (cloud clusters)
    elif kubectl get svc free-mlops-app-service -n $NAMESPACE -o jsonpath='{.spec.type}' | grep -q LoadBalancer; then
        echo ""
        print_status "LoadBalancer URLs (may take a few minutes to provision):"
        
        APP_IP=$(kubectl get svc free-mlops-app-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
        MLFLOW_IP=$(kubectl get svc free-mlops-mlflow-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
        API_IP=$(kubectl get svc free-mlops-api-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
        
        echo "  🌐 Streamlit: http://$APP_IP:8501"
        echo "  📊 MLflow:    http://$MLFLOW_IP:5000"
        echo "  🔌 API:       http://$API_IP:8000"
    
    # For port forwarding (local development)
    else
        echo ""
        print_status "Use port forwarding:"
        echo "  kubectl port-forward svc/free-mlops-app-service 8501:8501 -n $NAMESPACE"
        echo "  kubectl port-forward svc/free-mlops-mlflow-service 5000:5000 -n $NAMESPACE"
        echo "  kubectl port-forward svc/free-mlops-api-service 8000:8000 -n $NAMESPACE"
        echo ""
        echo "Then access:"
        echo "  🌐 Streamlit: http://localhost:8501"
        echo "  📊 MLflow:    http://localhost:5000"
        echo "  🔌 API:       http://localhost:8000"
    fi
}

# Function to show logs
show_logs() {
    if [ -n "$1" ]; then
        kubectl logs -f deployment/$1 -n $NAMESPACE
    else
        echo "Available deployments:"
        kubectl get deployments -n $NAMESPACE
        echo ""
        echo "Usage: $0 logs <deployment-name>"
        echo "Example: $0 logs free-mlops-app"
    fi
}

# Function to scale deployments
scale() {
    if [ -n "$1" ] && [ -n "$2" ]; then
        print_status "Scaling $1 to $2 replicas..."
        kubectl scale deployment $1 --replicas=$2 -n $NAMESPACE
        print_success "Scaling completed!"
    else
        echo "Usage: $0 scale <deployment-name> <replicas>"
        echo "Example: $0 scale free-mlops-app 3"
    fi
}

# Function to delete deployment
delete() {
    print_status "Deleting Free MLOps deployment..."
    kubectl delete namespace $NAMESPACE --ignore-not-found=true
    print_success "Deployment deleted!"
}

# Function to update deployment
update() {
    print_status "Updating deployment..."
    build_and_push
    kubectl rollout restart deployment/free-mlops-app -n $NAMESPACE
    kubectl rollout restart deployment/free-mlops-mlflow -n $NAMESPACE
    kubectl rollout restart deployment/free-mlops-api -n $NAMESPACE
    print_success "Deployment updated!"
}

# Main menu
case "$1" in
    "deploy")
        build_and_push
        deploy_k8s "$2"
        check_status "$2"
        get_urls
        ;;
    "build")
        build_and_push
        ;;
    "status")
        check_status "$2"
        get_urls
        ;;
    "logs")
        show_logs "$2"
        ;;
    "scale")
        scale "$2" "$3"
        ;;
    "delete")
        delete
        ;;
    "update")
        update
        ;;
    *)
        echo "Free MLOps - Kubernetes Deployment Script"
        echo ""
        echo "Usage: $0 {deploy|build|status|logs|scale|delete|update}"
        echo ""
        echo "Commands:"
        echo "  deploy        - Build and deploy all services"
        echo "  build         - Build Docker image only"
        echo "  status        - Check deployment status and URLs"
        echo "  logs          - Show logs (specify deployment name)"
        echo "  scale         - Scale deployment (name replicas)"
        echo "  delete        - Delete entire deployment"
        echo "  update        - Update deployment with new image"
        echo ""
        echo "Options:"
        echo "  --ingress     - Deploy with ingress (for deploy command)"
        echo ""
        echo "Examples:"
        echo "  $0 deploy              # Deploy all services"
        echo "  $0 deploy --ingress    # Deploy with ingress"
        echo "  $0 logs free-mlops-app # Show app logs"
        echo "  $0 scale free-mlops-app 3 # Scale app to 3 replicas"
        echo "  $0 status              # Check status"
        exit 1
        ;;
esac
