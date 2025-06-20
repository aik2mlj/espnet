#!/usr/bin/env python3
"""
Clean corrupted embeddings and restart extraction with fixed script
"""

import psycopg2
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clean_database():
    """Remove all existing embeddings to start fresh"""

    # Database configuration
    db_config = {
        "host": "localhost",
        "port": 5433,
        "database": "local_db",
        "user": "postgres",
        "password": "postgres",
    }

    try:
        # Connect to database
        conn = psycopg2.connect(**db_config)
        conn.autocommit = True
        cursor = conn.cursor()

        # Get current count
        cursor.execute("SELECT COUNT(*) FROM singer_embeddings;")
        count_before = cursor.fetchone()[0]
        logger.info(f"Found {count_before} existing embeddings")

        # Clear all data
        cursor.execute("DELETE FROM singer_embeddings;")
        logger.info("✅ Cleared all existing embeddings")

        # Verify cleanup
        cursor.execute("SELECT COUNT(*) FROM singer_embeddings;")
        count_after = cursor.fetchone()[0]
        logger.info(f"Embeddings remaining: {count_after}")

        conn.close()
        logger.info("🔐 Database connection closed")

        print(f"\n✅ Database cleaned successfully!")
        print(f"Removed {count_before} corrupted embeddings")
        print(f"Ready to restart extraction with fixed script")

    except Exception as e:
        logger.error(f"❌ Error cleaning database: {e}")


if __name__ == "__main__":
    clean_database()

