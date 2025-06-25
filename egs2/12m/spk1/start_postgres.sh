#!/bin/bash

echo "Starting PostgreSQL with pgvector for embedding storage..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Please install Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Start PostgreSQL
echo "Starting PostgreSQL container..."
docker-compose up -d

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
sleep 10

# Test connection
echo "Testing connection..."
for i in {1..30}; do
    if docker exec embedding_postgres pg_isready -U postgres -d local_db &> /dev/null; then
        echo "PostgreSQL is ready!"
        break
    fi
    echo "   Still waiting... (attempt $i/30)"
    sleep 2
done

# Show status
echo ""
echo "Database Status:"
docker ps | grep embedding_postgres

echo ""
echo "Connection Details:"
echo "  Host: localhost"
echo "  Port: 5433"
echo "  Database: local_db"
echo "  User: postgres"
echo "  Password: postgres"

echo ""
echo " To stop PostgreSQL later, run:"
echo "   docker-compose down"
echo ""
echo " To stop and remove all data, run:"
echo "   docker-compose down -v" 