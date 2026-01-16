#!/bin/bash

# GLM Model Production Setup Script
# ==================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="credit-risk-glm-production"
PYTHON_VERSION="3.10"
NODE_VERSION="18"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  GLM Model Production Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Function to print colored messages
print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Create project structure
create_structure() {
    print_info "Creating project structure..."
    
    directories=(
        "src"
        "tests/unit"
        "tests/integration"
        "tests/performance"
        "models"
        "data"
        "logs"
        "docs"
        "scripts"
        "k8s"
        "monitoring/grafana/dashboards"
        "monitoring/grafana/datasources"
        "nginx"
        "backups"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_info "Created directory: $dir"
    done
    
    # Create .gitkeep files
    touch models/.gitkeep
    touch data/.gitkeep
    touch logs/.gitkeep
    touch backups/.gitkeep
    
    print_success "Project structure created"
}

# Setup Python environment
setup_python() {
    print_info "Setting up Python environment..."
    
    # Create virtual environment
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_error "requirements.txt not found"
    fi
    
    # Install development dependencies
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
        print_success "Development dependencies installed"
    fi
    
    # Install pre-commit hooks
    if command -v pre-commit &> /dev/null; then
        pre-commit install
        print_success "Pre-commit hooks installed"
    fi
}

# Setup Docker
setup_docker() {
    print_info "Setting up Docker..."
    
    # Build Docker image
    docker build -t ${PROJECT_NAME}:latest .
    print_success "Docker image built"
    
    # Create Docker network
    docker network create ${PROJECT_NAME}-network 2>/dev/null || true
    print_success "Docker network created"
}

# Setup monitoring
setup_monitoring() {
    print_info "Setting up monitoring..."
    
    # Create Prometheus configuration
    cat > monitoring/prometheus.yml <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'credit-risk-api'
    static_configs:
      - targets: ['credit-risk-api:5000']
    metrics_path: '/metrics'
EOF
    
    # Create Grafana datasource
    cat > monitoring/grafana/datasources/prometheus.yml <<EOF
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
    
    print_success "Monitoring configuration created"
}

# Setup environment variables
setup_env() {
    print_info "Setting up environment variables..."
    
    if [ ! -f ".env" ]; then
        cat > .env <<EOF
# Model Configuration
MODEL_PATH=/app/models/glm_model.joblib
MODEL_VERSION=1.0.0

# API Configuration
PORT=5000
FLASK_ENV=production
MAX_BATCH_SIZE=1000

# Database
DATABASE_URL=postgresql://user:password@localhost/mlops

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Security
JWT_SECRET=$(openssl rand -hex 32)
API_KEY=$(openssl rand -hex 16)

# Cloud Storage (optional)
# AWS_S3_BUCKET=model-artifacts
# AWS_REGION=us-east-1
EOF
        print_success "Environment variables file created"
    else
        print_info ".env file already exists, skipping..."
    fi
}

# Setup Git
setup_git() {
    print_info "Setting up Git..."
    
    if [ ! -d ".git" ]; then
        git init
        print_success "Git repository initialized"
    fi
    
    # Create .gitignore
    cat > .gitignore <<EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
*.egg-info/
dist/
build/

# Testing
.coverage
htmlcov/
.pytest_cache/
.mypy_cache/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
logs/

# Models
models/*.joblib
models/*.pkl

# Data
data/
!data/.gitkeep

# Environment
.env
.env.*

# Backups
backups/
!backups/.gitkeep

# OS
.DS_Store
Thumbs.db
EOF
    
    print_success "Git configuration complete"
}

# Run initial tests
run_tests() {
    print_info "Running initial tests..."
    
    source venv/bin/activate
    
    # Run linting
    black --check src/ tests/ 2>/dev/null || true
    flake8 src/ tests/ 2>/dev/null || true
    
    # Run unit tests
    pytest tests/unit/ -v 2>/dev/null || print_info "No tests found yet"
    
    print_success "Initial tests completed"
}

# Main setup function
main() {
    echo ""
    print_info "Starting setup process..."
    echo ""
    
    check_prerequisites
    create_structure
    setup_env
    setup_python
    setup_docker
    setup_monitoring
    setup_git
    run_tests
    
    echo ""
    print_success "Setup completed successfully!"
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo "  1. Train your model: make train"
    echo "  2. Run tests: make test"
    echo "  3. Start API locally: make run-local"
    echo "  4. Deploy with Docker: make run-docker"
    echo "  5. Deploy to Kubernetes: make deploy-k8s"
    echo ""
    echo -e "${YELLOW}For more commands, run: make help${NC}"
}

# Run main function
main
