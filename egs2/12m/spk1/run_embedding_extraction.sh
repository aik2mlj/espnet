 #!/usr/bin/env bash

# Configuration - Update these paths according to your setup
EXP_DIR="/home/sc/espnet/egs2/12m/singfake/ai"
# Updated with the specific checkpoint the user wants to use
MODEL_PATH="/home/sc/espnet/egs2/12m/spk1/exp/spk_train_rawnet3_raw_sp/58epoch.pth"
CONFIG_PATH="/home/sc/espnet/egs2/12m/spk1/exp/spk_train_rawnet3_raw_sp/config.yaml"

# Database configuration
DB_HOST="localhost"
DB_NAME="local_db"
DB_USER="postgres"
DB_PASSWORD="postgres"
DB_PORT=5433

# Table name for storing embeddings
TABLE_NAME="singfake_embeddings"

# Source type for the audio files (ai or human)
SOURCE_TYPE="ai"

# Number of GPUs to use
NUM_GPUS=2

# Check if model and config exist
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    echo "Available models:"
    ls -la /home/sc/espnet/egs2/12m/spk1/exp/spk_train_rawnet3_raw_sp/*.pth
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    echo "Please update CONFIG_PATH in this script"
    exit 1
fi

echo "Starting embedding extraction with the following configuration:"
echo "EXP_DIR: $EXP_DIR"
echo "MODEL_PATH: $MODEL_PATH"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "NUM_GPUS: $NUM_GPUS"
echo "Database: $DB_HOST:$DB_PORT/$DB_NAME"
echo "Table: $TABLE_NAME"
echo "Source: $SOURCE_TYPE"
echo ""

# Install required dependencies if not already installed
echo "Installing required Python packages..."
pip install psycopg2-binary numpy torch soundfile

# Run the embedding extraction
python3 extract_embeddings_to_db.py \
    --exp_dir "$EXP_DIR" \
    --model_path "$MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    --num_gpus "$NUM_GPUS" \
    --db_host "$DB_HOST" \
    --db_name "$DB_NAME" \
    --db_user "$DB_USER" \
    --db_password "$DB_PASSWORD" \
    --db_port "$DB_PORT" \
    --table_name "$TABLE_NAME" \
    --source "$SOURCE_TYPE"

echo "Embedding extraction completed!"
