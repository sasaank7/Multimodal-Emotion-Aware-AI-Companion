#!/bin/bash

# Production Deployment Script for Emotion-Aware AI Companion
# Usage: ./scripts/deploy.sh [environment]

set -e  # Exit on error

# Colors for output
RED='\033[0:31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
PROJECT_NAME="emotion-ai-companion"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Emotion-Aware AI Companion Deployment${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "Environment: ${YELLOW}$ENVIRONMENT${NC}"
echo -e "Image Tag: ${YELLOW}$IMAGE_TAG${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command_exists docker; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

if ! command_exists docker-compose; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites satisfied${NC}"
echo ""

# Load environment variables
if [ -f ".env.$ENVIRONMENT" ]; then
    echo -e "${YELLOW}Loading environment variables from .env.$ENVIRONMENT${NC}"
    export $(cat ".env.$ENVIRONMENT" | grep -v '#' | xargs)
else
    echo -e "${RED}Warning: .env.$ENVIRONMENT not found${NC}"
fi

# Function to build images
build_images() {
    echo -e "${YELLOW}Building Docker images...${NC}"

    if [ "$ENVIRONMENT" = "production" ]; then
        docker build -f Dockerfile.prod -t "$PROJECT_NAME:$IMAGE_TAG" .
    else
        docker build -t "$PROJECT_NAME:$IMAGE_TAG" .
    fi

    echo -e "${GREEN}✓ Images built successfully${NC}"
    echo ""
}

# Function to run database migrations
run_migrations() {
    echo -e "${YELLOW}Running database migrations...${NC}"

    # Run migrations in container
    docker-compose -f docker-compose.${ENVIRONMENT}.yml run --rm api \
        python -m alembic upgrade head

    echo -e "${GREEN}✓ Migrations completed${NC}"
    echo ""
}

# Function to deploy services
deploy_services() {
    echo -e "${YELLOW}Deploying services...${NC}"

    if [ "$ENVIRONMENT" = "production" ]; then
        docker-compose -f docker-compose.prod.yml up -d --remove-orphans
    else
        docker-compose up -d --remove-orphans
    fi

    echo -e "${GREEN}✓ Services deployed${NC}"
    echo ""
}

# Function to wait for services
wait_for_services() {
    echo -e "${YELLOW}Waiting for services to be ready...${NC}"

    # Wait for API health check
    max_attempts=30
    attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            echo -e "${GREEN}✓ API is healthy${NC}"
            break
        fi

        attempt=$((attempt + 1))
        echo -e "Waiting for API... ($attempt/$max_attempts)"
        sleep 2
    done

    if [ $attempt -eq $max_attempts ]; then
        echo -e "${RED}Error: API failed to start${NC}"
        exit 1
    fi

    echo ""
}

# Function to run smoke tests
run_smoke_tests() {
    echo -e "${YELLOW}Running smoke tests...${NC}"

    # Test health endpoint
    if ! curl -f http://localhost:8000/health; then
        echo -e "${RED}✗ Health check failed${NC}"
        return 1
    fi

    # Test API endpoint
    if ! curl -f http://localhost:8000/api/v1/status; then
        echo -e "${RED}✗ API status check failed${NC}"
        return 1
    fi

    echo -e "${GREEN}✓ Smoke tests passed${NC}"
    echo ""
}

# Function to show deployment info
show_deployment_info() {
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""
    echo -e "Services:"
    echo -e "  - API:        http://localhost:8000"
    echo -e "  - UI:         http://localhost:8501"
    echo -e "  - API Docs:   http://localhost:8000/api/docs"
    echo -e "  - Grafana:    http://localhost:3000"
    echo -e "  - Prometheus: http://localhost:9090"
    echo ""
    echo -e "Useful commands:"
    echo -e "  - View logs:    docker-compose -f docker-compose.$ENVIRONMENT.yml logs -f"
    echo -e "  - Stop services: docker-compose -f docker-compose.$ENVIRONMENT.yml down"
    echo -e "  - Restart:      docker-compose -f docker-compose.$ENVIRONMENT.yml restart"
    echo ""
}

# Function to backup data
backup_data() {
    echo -e "${YELLOW}Creating backup...${NC}"

    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"

    # Backup database
    docker-compose -f docker-compose.${ENVIRONMENT}.yml exec -T postgres \
        pg_dump -U emotion_ai_user emotion_ai > "$BACKUP_DIR/database.sql"

    # Backup user data
    cp -r data/user_data "$BACKUP_DIR/"

    echo -e "${GREEN}✓ Backup created at $BACKUP_DIR${NC}"
    echo ""
}

# Function to rollback deployment
rollback() {
    echo -e "${RED}Rolling back deployment...${NC}"

    docker-compose -f docker-compose.${ENVIRONMENT}.yml down
    docker-compose -f docker-compose.${ENVIRONMENT}.yml up -d

    echo -e "${GREEN}✓ Rollback complete${NC}"
}

# Main deployment flow
main() {
    echo -e "${YELLOW}Starting deployment...${NC}"
    echo ""

    # Create backup before deployment
    if [ "$ENVIRONMENT" = "production" ]; then
        backup_data
    fi

    # Build images
    build_images

    # Deploy services
    deploy_services

    # Wait for services to be ready
    wait_for_services

    # Run migrations
    if [ "$ENVIRONMENT" = "production" ]; then
        run_migrations
    fi

    # Run smoke tests
    if ! run_smoke_tests; then
        echo -e "${RED}Deployment verification failed!${NC}"
        read -p "Rollback? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rollback
        fi
        exit 1
    fi

    # Show deployment info
    show_deployment_info
}

# Handle script arguments
case "${2:-deploy}" in
    deploy)
        main
        ;;
    backup)
        backup_data
        ;;
    rollback)
        rollback
        ;;
    *)
        echo "Usage: $0 [environment] [deploy|backup|rollback]"
        exit 1
        ;;
esac
