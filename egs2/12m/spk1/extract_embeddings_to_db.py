 #!/usr/bin/env python3
"""
Custom script to extract singer embeddings using ESPNet model and store in PostgreSQL
Optimized for multi-GPU processing with 4 A100s
Updated to support custom table names and source column (ai/human)
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
import psycopg2
from psycopg2.extras import execute_values
import multiprocessing as mp
from pathlib import Path
import logging
import yaml
from typing import List, Dict, Tuple
import argparse
from concurrent.futures import ThreadPoolExecutor
import time

# ESPNet imports
from espnet2.bin.spk_embed_extract import main as spk_embed_extract_main
from espnet2.tasks.spk import SpeakerTask
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PostgreSQLEmbeddingDB:
    """PostgreSQL database handler for embedding storage"""
    
    def __init__(self, db_config: dict, table_name: str = "singer_embeddings", source: str = "ai"):
        self.db_config = db_config
        self.table_name = table_name
        self.source = source  # 'ai' or 'human'
        self.conn = None
        self.create_connection()
        self.setup_tables()
    
    def create_connection(self):
        """Create PostgreSQL connection"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config.get('host', 'localhost'),
                database=self.db_config.get('database', 'local_db'),
                user=self.db_config.get('user', 'postgres'),
                password=self.db_config.get('password', 'postgres'),
                port=self.db_config.get('port', 5432)
            )
            self.conn.autocommit = True
            logger.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            sys.exit(1)
    
    def setup_tables(self):
        """Create tables if they don't exist"""
        try:
            cursor = self.conn.cursor()
            
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create singer embeddings table with source column
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    singer_id VARCHAR(50) NOT NULL,
                    audio_file_path TEXT NOT NULL,
                    embedding VECTOR(192), -- RawNet3 outputs 192-dimensional embeddings
                    source VARCHAR(10) NOT NULL, -- 'ai' or 'human'
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(singer_id, audio_file_path, source)
                );
            """)
            
            # Create index for vector similarity search
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_vector_idx 
                ON {self.table_name} USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
            """)
            
            # Create index for singer_id lookups
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_singer_id_idx 
                ON {self.table_name} (singer_id);
            """)
            
            # Create index for source lookups
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_source_idx 
                ON {self.table_name} (source);
            """)
            
            cursor.close()
            logger.info(f"Database table '{self.table_name}' and indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error setting up database tables: {e}")
            sys.exit(1)
    
    def insert_embedding(self, singer_id: str, audio_file_path: str, embedding: np.ndarray):
        """Insert a single embedding into the database"""
        try:
            cursor = self.conn.cursor()
            
            # Convert numpy array to list for PostgreSQL
            embedding_list = embedding.tolist()
            
            cursor.execute(f"""
                INSERT INTO {self.table_name} (singer_id, audio_file_path, embedding, source)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (singer_id, audio_file_path, source) 
                DO UPDATE SET embedding = EXCLUDED.embedding, created_at = CURRENT_TIMESTAMP
            """, (singer_id, audio_file_path, embedding_list, self.source))
            
            cursor.close()
            logger.debug(f"Inserted embedding for {singer_id}: {audio_file_path} (source: {self.source})")
            
        except Exception as e:
            logger.error(f"Error inserting embedding for {singer_id}: {e}")
    
    def batch_insert_embeddings(self, embeddings_data: List[Tuple[str, str, np.ndarray]]):
        """Batch insert multiple embeddings"""
        try:
            cursor = self.conn.cursor()
            
            # Prepare data for batch insert  
            values = []
            for singer_id, audio_file_path, embedding in embeddings_data:
                # Convert numpy array to proper format for pgvector
                # pgvector expects a string representation like '[1,2,3]'
                embedding_str = '[' + ','.join(map(str, embedding.flatten())) + ']'
                values.append((singer_id, audio_file_path, embedding_str, self.source))
            
            execute_values(
                cursor,
                f"""
                INSERT INTO {self.table_name} (singer_id, audio_file_path, embedding, source)
                VALUES %s
                ON CONFLICT (singer_id, audio_file_path, source) 
                DO UPDATE SET embedding = EXCLUDED.embedding, created_at = CURRENT_TIMESTAMP
                """,
                values,
                template=None,
                page_size=100
            )
            
            cursor.close()
            logger.info(f"Batch inserted {len(embeddings_data)} embeddings into table '{self.table_name}' (source: {self.source})")
            
        except Exception as e:
            logger.error(f"Error in batch insert: {e}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


class ESPNetEmbeddingExtractor:
    """ESPNet model wrapper for embedding extraction"""
    
    def __init__(self, model_path: str, config_path: str, device: str = "cuda:0"):
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the ESPNet speaker model"""
        try:
            # Load model using ESPNet2 SpeakerTask
            self.model, self.train_args = SpeakerTask.build_model_from_file(
                config_file=self.config_path,
                model_file=self.model_path,
                device=self.device
            )
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            sys.exit(1)
    
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """Extract embedding from audio file"""
        try:
            # Load audio
            speech, sample_rate = sf.read(audio_path)
            
            # Convert to tensor and move to device
            speech = torch.tensor(speech, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Extract embedding using ESPNet API
            with torch.no_grad():
                embedding = self.model(
                    speech=speech,
                    spk_labels=None,
                    extract_embd=True,
                    task_tokens=None
                )
                
            # Convert to numpy
            embedding = embedding.cpu().numpy().squeeze()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding from {audio_path}: {e}")
            return None


def process_singer_directory(args_tuple):
    """Process a single singer directory (for multiprocessing)"""
    singer_dir, model_path, config_path, gpu_id, db_config, table_name, source = args_tuple
    
    try:
        # Set GPU for this process (works better with spawn)
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        
        # Initialize extractor and database
        extractor = ESPNetEmbeddingExtractor(model_path, config_path, device)
        db = PostgreSQLEmbeddingDB(db_config, table_name, source)
        
        # Get singer ID from directory name
        singer_id = Path(singer_dir).name
        
        # Find all audio files in singer directory
        audio_extensions = ['*.wav', '*.flac', '*.mp3', '*.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(glob.glob(os.path.join(singer_dir, '**', ext), recursive=True))
        
        logger.info(f"Processing {len(audio_files)} audio files for singer {singer_id} on {device} (source: {source})")
        
        # Process each audio file
        embeddings_batch = []
        batch_size = 10  # Process in batches to avoid memory issues
        
        for i, audio_file in enumerate(audio_files):
            # Extract embedding
            embedding = extractor.extract_embedding(audio_file)
            
            if embedding is not None:
                embeddings_batch.append((singer_id, audio_file, embedding))
                
                # Insert batch when it reaches batch_size
                if len(embeddings_batch) >= batch_size:
                    db.batch_insert_embeddings(embeddings_batch)
                    embeddings_batch = []
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(audio_files)} files for singer {singer_id}")
        
        # Insert remaining embeddings
        if embeddings_batch:
            db.batch_insert_embeddings(embeddings_batch)
        
        db.close()
        logger.info(f"Completed processing singer {singer_id} with {len(audio_files)} files (source: {source})")
        
        return singer_id, len(audio_files)
        
    except Exception as e:
        logger.error(f"Error processing singer directory {singer_dir}: {e}")
        return singer_dir, 0


def main():
    # Fix CUDA multiprocessing issue
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Extract singer embeddings and store in PostgreSQL")
    parser.add_argument("--exp_dir", type=str, required=True, 
                       help="Path to experiment directory containing wav/idXXXXX folders")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained speaker model file")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to the model config file")
    parser.add_argument("--num_gpus", type=int, default=4,
                       help="Number of GPUs to use (default: 4)")
    parser.add_argument("--db_host", type=str, default="localhost",
                       help="PostgreSQL host")
    parser.add_argument("--db_name", type=str, default="local_db",
                       help="PostgreSQL database name")
    parser.add_argument("--db_user", type=str, default="postgres",
                       help="PostgreSQL username")
    parser.add_argument("--db_password", type=str, default="postgres",
                       help="PostgreSQL password")
    parser.add_argument("--db_port", type=int, default=5432,
                       help="PostgreSQL port")
    parser.add_argument("--table_name", type=str, default="singer_embeddings",
                       help="Name of the database table to store embeddings (default: singer_embeddings)")
    parser.add_argument("--source", type=str, default="ai", choices=["ai", "human"],
                       help="Source type for the audio files: 'ai' or 'human' (default: ai)")
    
    args = parser.parse_args()
    
    # Database configuration
    db_config = {
        'host': args.db_host,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password,
        'port': args.db_port
    }
    
    # Find all singer directories
    wav_dir = Path(args.exp_dir) / "wav"
    singer_dirs = [d for d in wav_dir.iterdir() if d.is_dir() and d.name.startswith('id')]
    
    logger.info(f"Found {len(singer_dirs)} singer directories to process")
    logger.info(f"Will store embeddings in table: {args.table_name}")
    logger.info(f"Source type: {args.source}")
    
    # Prepare arguments for multiprocessing
    process_args = []
    for i, singer_dir in enumerate(singer_dirs):
        gpu_id = i % args.num_gpus  # Distribute across available GPUs
        process_args.append((str(singer_dir), args.model_path, args.config_path, gpu_id, db_config, args.table_name, args.source))
    
    # Process singers in parallel using multiprocessing
    start_time = time.time()
    
    with mp.Pool(processes=args.num_gpus) as pool:
        results = pool.map(process_singer_directory, process_args)
    
    # Summary
    total_files = sum(result[1] for result in results)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Processing completed!")
    logger.info(f"Total singers processed: {len(results)}")
    logger.info(f"Total audio files processed: {total_files}")
    logger.info(f"Source type: {args.source}")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Average time per file: {elapsed_time/total_files:.2f} seconds")


if __name__ == "__main__":
    main()