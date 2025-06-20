#!/usr/bin/env python3
"""
Debug script to check what's actually stored in the database
"""

import psycopg2
import numpy as np

def debug_embeddings():
    """Check what's actually in the database"""
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5433,
            database='local_db',
            user='postgres',
            password='postgres'
        )
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT singer_id, audio_file_path, embedding 
            FROM singer_embeddings 
            WHERE singer_id = 'id02254' 
            LIMIT 1;
        """)
        
        result = cursor.fetchone()
        
        if result:
            singer_id, file_path, embedding = result
            
            print("🔍 RAW DATABASE CONTENT:")
            print(f"Singer ID: {singer_id}")
            print(f"File path: {file_path}")
            print(f"Embedding type: {type(embedding)}")
            print(f"Embedding repr: {repr(embedding)}")
            
            if hasattr(embedding, '__len__'):
                print(f"Embedding length: {len(embedding)}")
            
            # Try to convert to numpy array
            try:
                if isinstance(embedding, str):
                    print("❌ Embedding is stored as string!")
                    print(f"First 100 chars: {embedding[:100]}")
                elif isinstance(embedding, (list, tuple)):
                    print("✅ Embedding is stored as list/tuple")
                    print(f"First 5 values: {embedding[:5]}")
                    print(f"Length: {len(embedding)}")
                elif isinstance(embedding, np.ndarray):
                    print("✅ Embedding is stored as numpy array")
                    print(f"Shape: {embedding.shape}")
                    print(f"First 5 values: {embedding[:5]}")
                else:
                    print(f"🤔 Unknown embedding type: {type(embedding)}")
                    
            except Exception as e:
                print(f"❌ Error processing embedding: {e}")
        else:
            print("❌ No data found!")
            
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    debug_embeddings() 