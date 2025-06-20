#!/bin/bash

echo "Setting up PostgreSQL with conda..."

# Create or activate conda environment
if conda env list | grep -q "embedding_env"; then
    echo "📦 Activating existing embedding_env..."
    conda activate embedding_env
else
    echo "📦 Creating new conda environment: embedding_env..."
    conda create -n embedding_env python=3.9 -y
    conda activate embedding_env
fi

# Install PostgreSQL and dependencies
echo "⬇️  Installing PostgreSQL and dependencies..."
conda install -c conda-forge postgresql psycopg2 -y

# Install pgvector (this might require building from source)
echo "⬇️  Installing pgvector..."
pip install pgvector

# Initialize database
echo "🗄️  Initializing PostgreSQL database..."
export PGDATA="$HOME/postgres_data"
mkdir -p $PGDATA

# Initialize if not already done
if [ ! -f "$PGDATA/PG_VERSION" ]; then
    initdb -D $PGDATA --auth-local=trust --auth-host=trust
fi

# Start PostgreSQL
echo "🚀 Starting PostgreSQL server..."
pg_ctl -D $PGDATA -l $PGDATA/logfile start

# Wait a moment for startup
sleep 5

# Create database and user
echo "🔧 Setting up database..."
createdb local_db 2>/dev/null || echo "Database local_db already exists"
psql local_db -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || echo "Extension may need manual installation"

echo ""
echo "✅ PostgreSQL setup complete!"
echo ""
echo "🔗 Connection Details:"
echo "  Host: localhost"
echo "  Port: 5432"
echo "  Database: local_db"
echo "  User: $USER"
echo ""
echo "🛠️  To stop PostgreSQL later, run:"
echo "   pg_ctl -D $PGDATA stop"
echo ""
echo "⚠️  Note: pgvector extension may need manual compilation if conda version doesn't work" 