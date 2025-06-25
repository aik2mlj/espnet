# Singer Embedding Extraction from Checkpoint

This document describes the workflow for extracting speaker embeddings from a RawNet3 model checkpoint and storing them in a PostgreSQL database.

## Prerequisites

- ESPNet environment activated (`conda activate espnet`)
- Docker and Docker Compose
- NVIDIA GPUs with CUDA support
- Audio files organized in `/path/to/data/wav/idXXXXX/` structure
- Trained RawNet3 model (`58epoch.pth`)
current best version: MODEL_PATH="/home/sc/espnet/egs2/12m/spk1/exp/spk_train_rawnet3_raw_sp/58epoch.pth"

### Expected Dataset Directory Structure

```
/path/to/your/data/
├── wav/
│   ├── id00001/
│   │   ├── 00001.wav
│   │   ├── 00002.wav
│   │   └── ...
│   ├── id00002/
│   │   └── ...
│   └── ...
```

## **Setup**

### Step 1: Start Database

You can use the `docker-compose.yml` file in this directory to start the PostgreSQL database with pgvector extension:

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
Run with:
```bash
chmod +x run_embedding_extraction.sh
./run_embedding_extraction.sh
```


### Basic Statistics
```bash
python query_embeddings.py --action stats --db_port 5433
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