# Speaker Embedding Extraction Guide

This document describes the workflow for extracting speaker embeddings from a RawNet3 model checkpoint and storing them in a PostgreSQL database.

## Prerequisites

- ESPNet environment activated (`conda activate espnet`)
- Docker and Docker Compose
- NVIDIA GPUs with CUDA support
- Audio files organized in `/path/to/data/wav/idXXXXX/` structure
- Trained RawNet3 model (`58epoch.pth`)
current best version: MODEL_PATH="/home/sc/espnet/egs2/12m/spk1/exp/spk_train_rawnet3_raw_sp/58epoch.pth"

### Expected Directory Structure

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

## **Setup**

### Step 1: Start Database

A `docker-compose.yml` file is provided in this directory. Start the PostgreSQL database with pgvector extension:

```bash
docker-compose up -d
```

This creates a PostgreSQL database with:
- Database: `local_db`
- User: `postgres` / Password: `postgres`
- Port: `5433` (to avoid conflicts with existing PostgreSQL)
- pgvector extension enabled

The database schema will be created automatically when you run the extraction script.

## Model Configuration

The system uses a pre-trained RawNet3 model that outputs 192-dimensional embeddings.

Required files:
- Model: `exp/spk_train_rawnet3_raw_sp/58epoch.pth`
- Config: `exp/spk_train_rawnet3_raw_sp/config.yaml`

## Embedding Extraction

### Option: Automated Script

Edit `run_embedding_extraction.sh` to set your paths:

```bash
#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate espnet

# Edit these paths
EXP_DIR="/path/to/your/data"
MODEL_PATH="/home/sc/espnet/egs2/12m/spk1/exp/spk_train_rawnet3_raw_sp/58epoch.pth"
CONFIG_PATH="/home/sc/espnet/egs2/12m/spk1/exp/spk_train_rawnet3_raw_sp/config.yaml"

# Database settings
DB_HOST="localhost"
DB_PORT=5433
DB_NAME="local_db"
DB_USER="postgres"
DB_PASSWORD="postgres"

# Processing settings
TABLE_NAME="singer_embeddings"
SOURCE_TYPE="ai"
NUM_GPUS=2

python3 extract_embeddings_to_db.py \
    --exp_dir "$EXP_DIR" \
    --model_path "$MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    --num_gpus $NUM_GPUS \
    --db_host $DB_HOST \
    --db_port $DB_PORT \
    --db_name $DB_NAME \
    --db_user $DB_USER \
    --db_password $DB_PASSWORD \
    --table_name $TABLE_NAME \
    --source $SOURCE_TYPE
```

Run with:
```bash
chmod +x run_embedding_extraction.sh
./run_embedding_extraction.sh
```


### Basic Statistics
```bash
python query_embeddings.py --action stats --db_port 5433
```

### View Singer Data
```bash
python view_data.py --action singers
python view_data.py --action view --singer id02254 --limit 5
```

### Find Similar Singers
```bash
python query_embeddings.py --action similar --singer_id id02254 --top_k 10 --db_port 5433
```

### Direct Database Access
```bash
psql -h localhost -p 5433 -U postgres -d local_db

# Example queries:
SELECT COUNT(*) FROM singer_embeddings;
SELECT singer_id, COUNT(*) FROM singer_embeddings GROUP BY singer_id LIMIT 10;
```

## Troubleshooting

### Database Connection Issues
```bash
# Check if database is running
docker ps | grep embedding_postgres

# Restart database
docker-compose restart
```

### CUDA Memory Issues
- Reduce `--num_gpus` 
- Monitor with `nvidia-smi`


## Complete Workflow

```bash
# 1. Start database
docker-compose up -d

# 2. Run extraction (schema will be created automatically)
./run_embedding_extraction.sh

# 3. View results
python query_embeddings.py --action stats --db_port 5433
```

## Performance

- 1 GPU: ~50 files/minute
- 4 GPUs: ~180 files/minute
- Uses batch processing for memory efficiency
- Automatic error handling and recovery
- Database schema created automatically

## File Structure

```
espnet/egs2/12m/spk1/
├── docker-compose.yml              # PostgreSQL setup
├── extract_embeddings_to_db.py     # Main extraction script
├── run_embedding_extraction.sh     # Automated wrapper
├── view_data.py                    # Data viewer
├── query_embeddings.py             # Database queries
└── README_inference_guide.md       # This guide
``` 