#!/usr/bin/env python3
"""
Utility script to query and search singer embeddings stored in PostgreSQL
"""

import psycopg2
import numpy as np
import argparse
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingQueryClient:
    """Client for querying singer embeddings from PostgreSQL"""
    
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.conn = None
        self.create_connection()
    
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
            logger.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            exit(1)
    
    def get_singer_count(self) -> int:
        """Get total number of unique singers"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT singer_id) FROM singer_embeddings;")
        count = cursor.fetchone()[0]
        cursor.close()
        return count
    
    def get_total_embeddings(self) -> int:
        """Get total number of embeddings stored"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM singer_embeddings;")
        count = cursor.fetchone()[0]
        cursor.close()
        return count
    
    def get_singer_embeddings(self, singer_id: str) -> List[Tuple[str, np.ndarray]]:
        """Get all embeddings for a specific singer"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT audio_file_path, embedding 
            FROM singer_embeddings 
            WHERE singer_id = %s
        """, (singer_id,))
        
        results = []
        for row in cursor.fetchall():
            audio_file_path, embedding_list = row
            embedding = np.array(embedding_list)
            results.append((audio_file_path, embedding))
        
        cursor.close()
        return results
    
    def get_average_embedding(self, singer_id: str) -> np.ndarray:
        """Get average embedding for a singer"""
        embeddings = self.get_singer_embeddings(singer_id)
        if not embeddings:
            return None
        
        # Calculate average
        all_embeddings = np.array([emb for _, emb in embeddings])
        avg_embedding = np.mean(all_embeddings, axis=0)
        return avg_embedding
    
    def find_similar_singers(self, target_singer_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find singers most similar to the target singer using cosine similarity"""
        # Get target singer's average embedding
        target_embedding = self.get_average_embedding(target_singer_id)
        if target_embedding is None:
            logger.error(f"No embeddings found for singer {target_singer_id}")
            return []
        
        cursor = self.conn.cursor()
        
        # Use pgvector's cosine similarity search
        cursor.execute("""
            WITH target_embedding AS (
                SELECT AVG(embedding) as avg_emb
                FROM singer_embeddings 
                WHERE singer_id = %s
            ),
            singer_similarities AS (
                SELECT 
                    singer_id,
                    AVG(embedding) as avg_emb,
                    1 - (AVG(embedding) <=> (SELECT avg_emb FROM target_embedding)) as similarity
                FROM singer_embeddings 
                WHERE singer_id != %s
                GROUP BY singer_id
            )
            SELECT singer_id, similarity
            FROM singer_similarities
            ORDER BY similarity DESC
            LIMIT %s;
        """, (target_singer_id, target_singer_id, top_k))
        
        results = cursor.fetchall()
        cursor.close()
        
        return results
    
    def search_by_embedding(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """Search for similar audio files given a query embedding"""
        cursor = self.conn.cursor()
        
        # Convert numpy array to list for PostgreSQL
        query_list = query_embedding.tolist()
        
        cursor.execute("""
            SELECT singer_id, audio_file_path, 
                   1 - (embedding <=> %s::vector) as similarity
            FROM singer_embeddings
            ORDER BY similarity DESC
            LIMIT %s;
        """, (query_list, top_k))
        
        results = cursor.fetchall()
        cursor.close()
        
        return results
    
    def list_all_singers(self) -> List[Tuple[str, int]]:
        """List all singers and their embedding counts"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT singer_id, COUNT(*) as embedding_count
            FROM singer_embeddings
            GROUP BY singer_id
            ORDER BY embedding_count DESC;
        """)
        
        results = cursor.fetchall()
        cursor.close()
        
        return results
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    parser = argparse.ArgumentParser(description="Query singer embeddings from PostgreSQL")
    parser.add_argument("--action", choices=['stats', 'list', 'similar', 'search'], required=True,
                       help="Action to perform")
    parser.add_argument("--singer_id", type=str, 
                       help="Singer ID for similarity search")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of top results to return")
    parser.add_argument("--db_host", type=str, default="localhost")
    parser.add_argument("--db_name", type=str, default="local_db")
    parser.add_argument("--db_user", type=str, default="postgres")
    parser.add_argument("--db_password", type=str, default="postgres")
    parser.add_argument("--db_port", type=int, default=5432)
    
    args = parser.parse_args()
    
    # Database configuration
    db_config = {
        'host': args.db_host,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password,
        'port': args.db_port
    }
    
    # Initialize client
    client = EmbeddingQueryClient(db_config)
    
    try:
        if args.action == 'stats':
            # Show database statistics
            singer_count = client.get_singer_count()
            total_embeddings = client.get_total_embeddings()
            
            print(f"Database Statistics:")
            print(f"Total singers: {singer_count}")
            print(f"Total embeddings: {total_embeddings}")
            print(f"Average embeddings per singer: {total_embeddings/singer_count:.2f}")
        
        elif args.action == 'list':
            # List all singers
            singers = client.list_all_singers()
            
            print(f"All Singers ({len(singers)} total):")
            print("Singer ID\t\tEmbedding Count")
            print("-" * 40)
            for singer_id, count in singers:
                print(f"{singer_id}\t\t{count}")
        
        elif args.action == 'similar':
            # Find similar singers
            if not args.singer_id:
                print("Error: --singer_id is required for similarity search")
                return
            
            similar_singers = client.find_similar_singers(args.singer_id, args.top_k)
            
            if similar_singers:
                print(f"Top {len(similar_singers)} singers similar to {args.singer_id}:")
                print("Singer ID\t\tSimilarity Score")
                print("-" * 40)
                for singer_id, similarity in similar_singers:
                    print(f"{singer_id}\t\t{similarity:.4f}")
            else:
                print(f"No similar singers found for {args.singer_id}")
        
        elif args.action == 'search':
            # Example: search using average embedding of a singer
            if not args.singer_id:
                print("Error: --singer_id is required for embedding search")
                return
            
            # Get average embedding for the query singer
            query_embedding = client.get_average_embedding(args.singer_id)
            if query_embedding is None:
                print(f"No embeddings found for singer {args.singer_id}")
                return
            
            # Search for similar audio files
            similar_files = client.search_by_embedding(query_embedding, args.top_k)
            
            print(f"Top {len(similar_files)} audio files similar to average of {args.singer_id}:")
            print("Singer ID\t\tAudio File\t\t\t\t\tSimilarity")
            print("-" * 80)
            for singer_id, audio_file, similarity in similar_files:
                filename = audio_file.split('/')[-1]  # Just show filename
                print(f"{singer_id}\t\t{filename[:30]:<30}\t\t{similarity:.4f}")
    
    finally:
        client.close()


if __name__ == "__main__":
    main() 