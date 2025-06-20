# Batch Inference on Experimental Dataset

This guide provides a complete workflow for extracting singer embeddings from audio files using a trained ESPNet RawNet3 model and storing them in a PostgreSQL database for efficient similarity search and analysis.

## 🎯 **Overview**

Instead of calculating EER (Equal Error Rate) for speaker verification, this workflow extracts **192-dimensional embeddings** from audio files and stores them in a database. This allows for:
- Fast similarity search between singers
- Scalable storage of embeddings
- Easy data analysis and visualization
- Efficient batch processing using multiple GPUs

## 📋 **Prerequisites**

- ESPNet environment activated (`conda activate espnet`)
- Trained RawNet3 model (`58epoch.pth`)
- Audio files organized in `/path/to/exp/wav/idXXXXX/` structure
- Docker installed (for PostgreSQL)
- At least 1 GPU available

## 🛠️ **Setup Process**

### Step 1: Database Setup

#### 1.1 Start PostgreSQL Database
```bash
# Make the startup script executable
chmod +x start_postgres.sh

# Start PostgreSQL with pgvector extension
./start_postgres.sh
```

**What this does:**
- Starts PostgreSQL on port 5433 (to avoid conflicts)
- Enables pgvector extension for vector similarity search
- Creates database with credentials: `postgres/postgres`

#### 1.2 Database Configuration Files

**`docker-compose.yml`** - PostgreSQL container configuration:
```yaml
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: local_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5433:5432"  # Custom port to avoid conflicts
    volumes:
      - postgres_data:/var/lib/postgresql/data
volumes:
  postgres_data:
```

**`start_postgres.sh`** - Convenience script to start database:
```bash
#!/bin/bash
echo "Starting PostgreSQL with pgvector..."
docker-compose up -d
echo "Database running on localhost:5433"
```

### Step 2: Batch Inference

#### 2.1 Main Inference Script

**`extract_embeddings_to_db.py`** - Core batch inference script with these features:

**Key Features:**
- **Multi-GPU Support**: Parallel processing across multiple GPUs
- **Batch Processing**: Processes multiple audio files efficiently
- **Database Storage**: Stores embeddings with metadata
- **Progress Tracking**: Shows real-time processing status
- **Error Handling**: Robust error handling and logging

**Usage:**
```bash
# Single GPU processing
python extract_embeddings_to_db.py \
  --exp_dir "/home/sc/espnet/egs2/12m/12m/exp" \
  --model_path "/home/sc/espnet/egs2/12m/spk1/exp/spk_train_rawnet3_raw_sp/58epoch.pth" \
  --config_path "/home/sc/espnet/egs2/12m/spk1/exp/spk_train_rawnet3_raw_sp/config.yaml" \
  --num_gpus 1 \
  --db_port 5433

# Multi-GPU processing (4 GPUs)
python extract_embeddings_to_db.py \
  --exp_dir "/home/sc/espnet/egs2/12m/12m/exp" \
  --model_path "/home/sc/espnet/egs2/12m/spk1/exp/spk_train_rawnet3_raw_sp/58epoch.pth" \
  --config_path "/home/sc/espnet/egs2/12m/spk1/exp/spk_train_rawnet3_raw_sp/config.yaml" \
  --num_gpus 4 \
  --db_port 5433
```

**Script Architecture:**
```python
# Key Components:
class EmbeddingExtractor:
    - Loads ESPNet RawNet3 model
    - Handles audio file processing
    - Extracts 192-dimensional embeddings
    - Manages GPU assignment

class DatabaseManager:
    - Handles PostgreSQL connections
    - Creates tables with correct schema
    - Batch inserts for efficiency
    - Manages transactions

# Multiprocessing:
- spawn method for CUDA compatibility
- Process-per-GPU architecture
- Shared task queue
- Progress synchronization
```

#### 2.2 Convenience Shell Script

**`run_embedding_extraction.sh`** - Wrapper script for easy execution:
```bash
#!/bin/bash
# Convenience script for batch embedding extraction

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate espnet

# Set paths
EXP_DIR="/home/sc/espnet/egs2/12m/12m/exp"
MODEL_PATH="/home/sc/espnet/egs2/12m/spk1/exp/spk_train_rawnet3_raw_sp/58epoch.pth"
CONFIG_PATH="/home/sc/espnet/egs2/12m/spk1/exp/spk_train_rawnet3_raw_sp/config.yaml"

# Run extraction
python extract_embeddings_to_db.py \
  --exp_dir "$EXP_DIR" \
  --model_path "$MODEL_PATH" \
  --config_path "$CONFIG_PATH" \
  --num_gpus 4 \
  --db_port 5433
```

### Step 3: Database Schema

The embeddings are stored in a PostgreSQL table with this structure:

```sql
CREATE TABLE singer_embeddings (
    id SERIAL PRIMARY KEY,
    singer_id VARCHAR(50) NOT NULL,           -- e.g., "id02254"
    audio_file_path TEXT NOT NULL,            -- Full path to audio file
    embedding VECTOR(192),                    -- 192-dimensional embedding
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(singer_id, audio_file_path)        -- Prevent duplicates
);
```

**Schema Explanation:**
- `VECTOR(192)`: Uses pgvector extension for efficient similarity search
- `UNIQUE constraint`: Prevents processing the same file twice
- `SERIAL PRIMARY KEY`: Auto-incrementing unique identifier
- `created_at`: Timestamp for tracking when embeddings were extracted

### Step 4: Troubleshooting Scripts

#### 4.1 Database Schema Fix

**`fix_db_schema.py`** - Fixes dimension mismatches:
```python
# Purpose: Drop and recreate table with correct dimensions
# Use case: When model outputs different dimensions than expected
# Usage: python fix_db_schema.py
```

This script handles the common issue where the database table is created with wrong vector dimensions.

## 📊 **Data Access and Analysis**

### Method 1: Simple Data Viewer (Recommended for Beginners)

**`view_data.py`** - User-friendly script for viewing data in JSON-like format:

```bash
# Basic statistics
python view_data.py --action stats

# List all singers and file counts
python view_data.py --action singers

# View specific singer's data (JSON format)
python view_data.py --action view --singer id02254 --limit 5

# Export singer data to JSON file
python view_data.py --action export --singer id02254 --output singer_data.json
```

**Example Output:**
```json
{
  "singer_id": "id02254",
  "files": [
    {
      "file_number": 1,
      "audio_file": "00001.wav",
      "full_path": "/home/sc/espnet/egs2/12m/12m/exp/wav/id02254/00001.wav",
      "embedding_preview": [0.123, -0.456, 0.789, 0.012, -0.345],
      "embedding_size": 192,
      "processed_at": "2025-01-08 00:03:11"
    }
  ]
}
```

### Method 2: Advanced Query Utilities

**`query_embeddings.py`** - Advanced database operations:

```bash
# Database statistics
python query_embeddings.py --action stats --db_port 5433

# Find similar singers using cosine similarity
python query_embeddings.py --action similar --singer_id id02254 --top_k 5 --db_port 5433

# Search embeddings for specific singer
python query_embeddings.py --action search --singer_id id02254 --limit 10 --db_port 5433

# List all singers
python query_embeddings.py --action list --db_port 5433
```

**Similarity Search Example:**
```bash
# Find top 5 singers most similar to id02254
python query_embeddings.py --action similar --singer_id id02254 --top_k 5

# Output:
# 🔍 Top 5 singers similar to id02254:
# 1. id46576 (similarity: 0.892)
# 2. id31847 (similarity: 0.847)
# 3. id52139 (similarity: 0.823)
# 4. id18756 (similarity: 0.801)
# 5. id29384 (similarity: 0.789)
```

### Method 3: Direct Database Access

For advanced users, connect directly to PostgreSQL:

```bash
# Connect using psql
psql -h localhost -p 5433 -U postgres -d local_db

# Example queries:
SELECT COUNT(*) FROM singer_embeddings;
SELECT singer_id, COUNT(*) FROM singer_embeddings GROUP BY singer_id;
SELECT * FROM singer_embeddings WHERE singer_id = 'id02254' LIMIT 5;
```

## 🚀 **Complete Workflow Example**

```bash
# 1. Start database
./start_postgres.sh

# 2. Run batch inference (4 GPUs)
python extract_embeddings_to_db.py \
  --exp_dir "/home/sc/espnet/egs2/12m/12m/exp" \
  --model_path "/home/sc/espnet/egs2/12m/spk1/exp/spk_train_rawnet3_raw_sp/58epoch.pth" \
  --config_path "/home/sc/espnet/egs2/12m/spk1/exp/spk_train_rawnet3_raw_sp/config.yaml" \
  --num_gpus 4 \
  --db_port 5433

# 3. View results
python view_data.py --action singers

# 4. Export specific singer data
python view_data.py --action export --singer id02254

# 5. Find similar singers
python query_embeddings.py --action similar --singer_id id02254 --top_k 10
```

## 📁 **File Structure**

```
espnet/egs2/12m/spk1/
├── extract_embeddings_to_db.py      # Main batch inference script
├── run_embedding_extraction.sh      # Convenience shell script
├── docker-compose.yml               # PostgreSQL container config
├── start_postgres.sh               # Database startup script
├── view_data.py                    # User-friendly data viewer
├── query_embeddings.py             # Advanced database queries
├── fix_db_schema.py                # Schema troubleshooting
└── README_batch_inference.md       # This documentation
```

## 🔧 **Performance Optimization**

### Multi-GPU Scaling
- **1 GPU**: ~50 files/minute
- **4 GPUs**: ~180 files/minute
- **Batch Size**: Automatically optimized per GPU

### Database Performance
- **Batch Inserts**: 10 embeddings per transaction
- **Connection Pooling**: Reuses database connections
- **Vector Indexing**: pgvector provides efficient similarity search

### Memory Management
- **Streaming Processing**: Processes files without loading all into memory
- **CUDA Memory**: Automatic cleanup between batches
- **Database Connections**: Proper connection management

## 🐛 **Common Issues and Solutions**

### Issue 1: CUDA Multiprocessing Error
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess
```
**Solution**: Script uses `spawn` method instead of `fork` for multiprocessing.

### Issue 2: Database Dimension Mismatch
```
ERROR: vector dimension mismatch
```
**Solution**: Run `python fix_db_schema.py` to recreate table with correct dimensions.

### Issue 3: Port Conflicts
```
ERROR: port 5432 already in use
```
**Solution**: Script uses port 5433 to avoid conflicts with existing PostgreSQL.

### Issue 4: Out of Memory
```
CUDA out of memory
```
**Solution**: Reduce batch size or number of GPUs in the script.

## 📈 **Expected Results**

After processing your dataset, you should have:
- **Thousands of embeddings** stored in the database
- **Unique singer IDs** mapped to their audio files
- **192-dimensional vectors** for each audio file
- **Similarity search capabilities** for finding related singers
- **JSON export functionality** for further analysis

## 🎯 **Next Steps**

1. **Analysis**: Use the embeddings for clustering or classification
2. **Visualization**: Create t-SNE or UMAP plots of singer similarities
3. **Applications**: Build recommendation systems or similarity search tools
4. **Scaling**: Process larger datasets with the same infrastructure 