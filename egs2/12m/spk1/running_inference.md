# Running Inference and Evaluation Guide

This guide provides comprehensive instructions for setting up the database, extracting speaker embeddings, and running evaluations using the ESPNet speaker verification system.

## Overview

The evaluation pipeline consists of the following components:

1. **Database Setup**: PostgreSQL with pgvector extension for storing embeddings
2. **Model Configuration**: RawNet3 model trained for speaker verification
3. **Embedding Extraction**: Multi-GPU extraction pipeline for processing audio files
4. **Evaluation Tools**: Query and analysis utilities for examining results
5. **Visualization**: Jupyter notebook for data exploration

## Prerequisites

- NVIDIA GPUs with CUDA support
- Docker and Docker Compose (for containerized database)
- Python 3.9+ with ESPNet2 installed
- Audio files organized in the expected directory structure

## Directory Structure

Your audio files should be organized as follows:
```
/path/to/your/data/
├── wav/
│   ├── id00001/
│   │   ├── audio1.wav
│   │   ├── audio2.wav
│   │   └── ...
│   ├── id00002/
│   │   └── ...
│   └── ...
```

## Step 1: Database Setup

### Option A: Docker-based Setup (Recommended)

The easiest way to set up PostgreSQL with pgvector support:

1. **Start PostgreSQL container:**
   ```bash
   ./start_postgres.sh
   ```
   
   This script will:
   - Start a PostgreSQL container with pgvector extension
   - Create database `local_db` on port 5433
   - Set up proper volumes for data persistence

2. **Verify database is running:**
   ```bash
   docker ps | grep embedding_postgres
   ```

### Option B: Conda-based Setup

If you prefer a local installation:

1. **Run the setup script:**
   ```bash
   ./setup_conda_postgres.sh
   ```
   
   This script will:
   - Create conda environment `embedding_env`
   - Install PostgreSQL and dependencies
   - Initialize database with pgvector extension

### Database Schema Initialization

After setting up PostgreSQL, initialize the schema:

```bash
python3 fix_db_schema.py
```

This creates the `singer_embeddings` table with:
- `id`: Primary key
- `singer_id`: Speaker identifier (e.g., 'id00001')  
- `audio_file_path`: Full path to audio file
- `embedding`: 192-dimensional vector (VECTOR type)
- `source`: Data source ('ai' or 'human')
- `created_at`: Timestamp

## Step 2: Model Configuration

The system uses a pre-trained RawNet3 model. Key configuration files:

- **Model**: `exp/spk_train_rawnet3_raw_sp/58epoch.pth`
- **Config**: `exp/spk_train_rawnet3_raw_sp/config.yaml`
- **Training Config**: `conf/train_rawnet3.yaml`

### Model Specifications

From `conf/train_rawnet3.yaml`:
- **Frontend**: ASteroid frontend with SINC filters
- **Encoder**: RawNet3 with 1536 hidden dimensions
- **Projector**: Outputs 192-dimensional embeddings
- **Pooling**: Channel attention statistics pooling
- **Loss**: AAM-Softmax with margin=0.3, scale=30

## Step 3: Embedding Extraction

### Main Extraction Script

The primary extraction is handled by `run_embedding_extraction.sh`:

```bash
./run_embedding_extraction.sh
```

### Script Configuration

Edit the following variables in `run_embedding_extraction.sh`:

```bash
# Data directory containing wav/idXXXXX folders
EXP_DIR="/path/to/your/data"

# Model files (should be pre-trained)
MODEL_PATH="/home/sc/espnet/egs2/12m/spk1/exp/spk_train_rawnet3_raw_sp/58epoch.pth"
CONFIG_PATH="/home/sc/espnet/egs2/12m/spk1/exp/spk_train_rawnet3_raw_sp/config.yaml"

# Database connection
DB_HOST="localhost"
DB_PORT=5433
DB_NAME="local_db"
DB_USER="postgres"
DB_PASSWORD="postgres"

# Table and source configuration
TABLE_NAME="singfake_embeddings"  # Customize table name
SOURCE_TYPE="ai"  # or "human" depending on your data

# GPU configuration
NUM_GPUS=2  # Adjust based on available GPUs
```

### Manual Extraction

For more control, run the extraction directly:

```bash
python3 extract_embeddings_to_db.py \
    --exp_dir "/path/to/your/data" \
    --model_path "exp/spk_train_rawnet3_raw_sp/58epoch.pth" \
    --config_path "exp/spk_train_rawnet3_raw_sp/config.yaml" \
    --num_gpus 2 \
    --db_host localhost \
    --db_port 5433 \
    --table_name "singer_embeddings" \
    --source "ai"
```

### Extraction Process Details

The `extract_embeddings_to_db.py` script:

1. **Multi-GPU Processing**: Distributes singers across available GPUs
2. **Batch Processing**: Processes audio files in batches of 10 for memory efficiency
3. **Error Handling**: Continues processing even if individual files fail
4. **Database Integration**: Stores embeddings directly in PostgreSQL with conflict resolution
5. **Progress Logging**: Provides detailed progress information

### Performance Optimization

- **Memory Management**: Uses batch insertion to avoid memory overflow
- **GPU Distribution**: Automatically distributes workload across specified GPUs
- **Multiprocessing**: Uses spawn method for CUDA compatibility
- **Error Recovery**: Skips problematic files and continues processing

## Step 4: Database Management and Debugging

### Debug Embeddings

Check what's stored in the database:

```bash
python3 debug_embeddings.py
```

This script:
- Connects to the database
- Retrieves sample embeddings
- Verifies data types and dimensions
- Reports any storage issues

### Clean and Restart

If you need to restart extraction:

```bash
python3 clean_and_restart.py
```

This script:
- Removes all existing embeddings
- Preserves table structure
- Provides cleanup statistics

### Database Statistics

Check the database status:

```bash
python3 query_embeddings.py --action stats
```

## Step 5: Querying and Analysis

### Query Embeddings Script

The `query_embeddings.py` provides several analysis options:

#### Basic Statistics
```bash
python3 query_embeddings.py --action stats --db_port 5433
```

#### List All Singers
```bash
python3 query_embeddings.py --action list --db_port 5433
```

#### Find Similar Singers
```bash
python3 query_embeddings.py --action similar --singer_id id00001 --top_k 10 --db_port 5433
```

#### Search by Embedding
```bash
python3 query_embeddings.py --action search --top_k 10 --db_port 5433
```

### Query Capabilities

The query system supports:

1. **Cosine Similarity Search**: Uses pgvector's optimized similarity operators
2. **Average Embeddings**: Computes per-singer average embeddings
3. **Batch Queries**: Efficient processing of multiple queries
4. **Flexible Filtering**: Search by singer, source type, or similarity threshold

## Step 6: Database Connection Details

### Docker Setup (Port 5433)
```bash
Host: localhost
Port: 5433
Database: local_db
User: postgres
Password: postgres
```

### Conda Setup (Port 5432)
```bash
Host: localhost  
Port: 5432
Database: local_db
User: $USER
```

## Step 7: Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in extraction script
   - Decrease number of parallel processes
   - Use fewer GPUs

2. **Database Connection Failed**
   - Verify PostgreSQL is running: `docker ps`
   - Check port availability: `netstat -an | grep 5433`
   - Restart database: `docker-compose restart`

3. **Model Loading Errors**
   - Verify model file exists and is readable
   - Check CUDA availability: `nvidia-smi`
   - Ensure ESPNet2 is properly installed

4. **Embedding Dimension Mismatch**
   - Run `fix_db_schema.py` to recreate table with correct dimensions
   - Verify model outputs 192-dimensional embeddings

### Performance Monitoring

Monitor extraction progress:
- Check log output for processing rates
- Monitor GPU utilization: `nvidia-smi`
- Check database size: `du -sh postgres_data/`

## Step 8: Evaluation and Analysis

### Data Exploration

For detailed evaluation and visualization, use the provided Jupyter notebook:

```bash
jupyter notebook see_data.ipynb
```

This notebook provides comprehensive analysis tools for exploring the extracted embeddings and running evaluations.

## Summary

The complete workflow:

1. **Setup**: `./start_postgres.sh` + `python3 fix_db_schema.py`
2. **Extract**: `./run_embedding_extraction.sh` 
3. **Query**: `python3 query_embeddings.py --action stats`
4. **Analyze**: `jupyter notebook see_data.ipynb`

This pipeline provides a complete solution for speaker embedding extraction, storage, and analysis using ESPNet's RawNet3 model with PostgreSQL backend storage. 