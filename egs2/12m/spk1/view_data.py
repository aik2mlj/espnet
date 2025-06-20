#!/usr/bin/env python3
"""
Simple script to view embedding data in JSON-like format
Perfect for beginners who are used to JSON
"""

import psycopg2
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.metrics.pairwise import cosine_distances

def connect_to_database(port=5433):
    """Connect to the database"""
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=port,
            database='local_db',
            user='postgres',
            password='postgres'
        )
        print("✅ Connected to database successfully!")
        return conn
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return None

def show_basic_stats(conn):
    """Show basic statistics about your data"""
    cursor = conn.cursor()
    
    # Total number of embeddings
    cursor.execute("SELECT COUNT(*) FROM singer_embeddings;")
    total_embeddings = cursor.fetchone()[0]
    
    # Number of unique singers
    cursor.execute("SELECT COUNT(DISTINCT singer_id) FROM singer_embeddings;")
    unique_singers = cursor.fetchone()[0]
    
    print("\n📊 YOUR DATA SUMMARY:")
    print(f"Total audio files processed: {total_embeddings}")
    print(f"Number of different singers: {unique_singers}")
    print(f"Average files per singer: {total_embeddings/unique_singers:.1f}")

def list_all_singers(conn):
    """Show all singers and how many files each has"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT singer_id, COUNT(*) as file_count
        FROM singer_embeddings 
        GROUP BY singer_id 
        ORDER BY file_count DESC;
    """)
    
    results = cursor.fetchall()
    
    print("\n👥 ALL SINGERS:")
    print("Singer ID        | Number of Files")
    print("-" * 35)
    for singer_id, count in results:
        print(f"{singer_id:<15} | {count}")

def show_singer_data(conn, singer_id, limit=5):
    """Show data for a specific singer (like JSON)"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT audio_file_path, embedding, created_at
        FROM singer_embeddings 
        WHERE singer_id = %s
        LIMIT %s;
    """, (singer_id, limit))
    
    results = cursor.fetchall()
    
    if not results:
        print(f"❌ No data found for singer: {singer_id}")
        return
    
    print(f"\n🎤 DATA FOR SINGER: {singer_id}")
    print(f"Showing first {len(results)} files:")
    
    # Format like JSON for familiarity
    singer_data = {
        "singer_id": singer_id,
        "files": []
    }
    
    for i, (file_path, embedding, timestamp) in enumerate(results, 1):
        # Parse embedding properly
        try:
            if isinstance(embedding, str):
                emb_str = embedding.strip('[]')
                emb_array = np.array([float(x.strip()) for x in emb_str.split(',')])
                preview = emb_array[:5].tolist()
                size = len(emb_array)
            else:
                preview = embedding[:5] if hasattr(embedding, '__getitem__') else "Error"
                size = len(embedding) if hasattr(embedding, '__len__') else "Unknown"
        except:
            preview = "Parse Error"
            size = "Unknown"
        
        file_info = {
            "file_number": i,
            "audio_file": file_path.split('/')[-1],  # Just filename
            "full_path": file_path,
            "embedding_preview": preview,
            "embedding_size": size,
            "processed_at": str(timestamp)
        }
        singer_data["files"].append(file_info)
    
    # Print in pretty JSON format
    print(json.dumps(singer_data, indent=2))

def export_singer_to_json(conn, singer_id, output_file):
    """Export all data for a singer to a JSON file"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT audio_file_path, embedding, created_at
        FROM singer_embeddings 
        WHERE singer_id = %s;
    """, (singer_id,))
    
    results = cursor.fetchall()
    
    if not results:
        print(f"❌ No data found for singer: {singer_id}")
        return
    
    # Convert to JSON-friendly format
    export_data = {
        "singer_id": singer_id,
        "total_files": len(results),
        "embeddings": []
    }
    
    for file_path, embedding, timestamp in results:
        # Parse embedding string to array
        try:
            if isinstance(embedding, str):
                emb_str = embedding.strip('[]')
                emb_array = [float(x.strip()) for x in emb_str.split(',')]
            else:
                emb_array = embedding
        except:
            emb_array = "Parse Error"
        
        embedding_data = {
            "audio_file": file_path,
            "embedding": emb_array,
            "processed_at": str(timestamp)
        }
        export_data["embeddings"].append(embedding_data)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"✅ Exported {len(results)} embeddings to: {output_file}")

def get_singers_dataframe(port=5433):
    """Get singers data as pandas DataFrame"""
    conn = connect_to_database(port)
    df = pd.read_sql("""
        SELECT singer_id, COUNT(*) as file_count
        FROM singer_embeddings 
        GROUP BY singer_id 
        ORDER BY file_count DESC;
    """, conn)
    conn.close()
    return df

def create_dashboard(port=5433):
    """Create comprehensive dashboard"""
    conn = connect_to_database(port)
    
    # Get singer data
    df_singers = pd.read_sql("""
        SELECT singer_id, COUNT(*) as file_count
        FROM singer_embeddings 
        GROUP BY singer_id 
        ORDER BY file_count DESC;
    """, conn)
    
    # Get timeline data
    df_timeline = pd.read_sql("""
        SELECT DATE_TRUNC('hour', created_at) as hour,
               COUNT(*) as files_processed
        FROM singer_embeddings 
        GROUP BY DATE_TRUNC('hour', created_at)
        ORDER BY hour;
    """, conn)
    
    conn.close()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('File Count Distribution', 'Top 20 Singers', 
                       'Processing Timeline', 'Singer Categories'),
        specs=[[{"type": "histogram"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "pie"}]]
    )
    
    # File count histogram
    fig.add_histogram(x=df_singers['file_count'], nbinsx=25, name="File Counts", row=1, col=1)
    
    # Top 20 singers
    top_20 = df_singers.head(20)
    fig.add_bar(x=top_20['singer_id'], y=top_20['file_count'], name="Top Singers", row=1, col=2)
    
    # Processing timeline
    if not df_timeline.empty:
        fig.add_scatter(x=df_timeline['hour'], y=df_timeline['files_processed'], 
                       mode='lines+markers', name="Processing Rate", row=2, col=1)
    
    # Singer categories
    categories = {
        'High (>200)': len(df_singers[df_singers['file_count'] > 200]),
        'Medium (50-200)': len(df_singers[(df_singers['file_count'] >= 50) & 
                                        (df_singers['file_count'] <= 200)]),
        'Low (<50)': len(df_singers[df_singers['file_count'] < 50])
    }
    
    fig.add_pie(labels=list(categories.keys()), values=list(categories.values()),
                name="Categories", row=2, col=2)
    
    fig.update_layout(height=800, title_text="🎤 Singer Embeddings Dashboard", showlegend=False)
    fig.show()
    
    # Print summary
    print("\n📊 SUMMARY STATISTICS:")
    print(f"Total singers: {len(df_singers)}")
    print(f"Total files: {df_singers['file_count'].sum():,}")
    print(f"Average files per singer: {df_singers['file_count'].mean():.1f}")
    print(f"Median files per singer: {df_singers['file_count'].median():.1f}")
    print(f"Max files: {df_singers['file_count'].max()} | Min files: {df_singers['file_count'].min()}")
    
    return df_singers

def explore_embeddings(singer_id=None, n_samples=50, port=5433):
    """Explore embedding vectors in detail"""
    conn = connect_to_database(port)
    
    if singer_id:
        query = """SELECT singer_id, audio_file_path, embedding FROM singer_embeddings WHERE singer_id = %s LIMIT %s;"""
        params = (singer_id, n_samples)
    else:
        query = """SELECT singer_id, audio_file_path, embedding FROM singer_embeddings ORDER BY RANDOM() LIMIT %s;"""
        params = (n_samples,)
    
    cursor = conn.cursor()
    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        print("❌ No embeddings found!")
        return None
    
    # Parse embeddings
    embeddings_data = []
    for singer, path, emb_str in results:
        try:
            if isinstance(emb_str, str):
                emb_str = emb_str.strip('[]')
                embedding = np.array([float(x.strip()) for x in emb_str.split(',')])
            else:
                embedding = np.array(emb_str)
            
            embeddings_data.append({
                'singer_id': singer,
                'file': path.split('/')[-1],
                'embedding': embedding,
                'norm': np.linalg.norm(embedding),
                'mean': np.mean(embedding),
                'std': np.std(embedding)
            })
        except Exception as e:
            print(f"⚠️ Parse error: {e}")
            continue
    
    if not embeddings_data:
        print("❌ No valid embeddings parsed!")
        return None
    
    print(f"✅ Analyzed {len(embeddings_data)} embeddings")
    print(f"📏 Embedding dimension: {len(embeddings_data[0]['embedding'])}")
    
    # Statistics
    norms = [d['norm'] for d in embeddings_data]
    means = [d['mean'] for d in embeddings_data]
    stds = [d['std'] for d in embeddings_data]
    
    print(f"\n📊 EMBEDDING STATISTICS:")
    print(f"Norm - Range: {min(norms):.2f} to {max(norms):.2f}, Mean: {np.mean(norms):.2f}")
    print(f"Mean - Range: {min(means):.2f} to {max(means):.2f}, Overall: {np.mean(means):.2f}")
    print(f"Std  - Range: {min(stds):.2f} to {max(stds):.2f}, Mean: {np.mean(stds):.2f}")
    
    # Show sample
    print(f"\n🔍 SAMPLE EMBEDDING (first 10 values):")
    print(embeddings_data[0]['embedding'][:10])
    
    return embeddings_data

def find_similar_singers(target_singer, top_k=10, port=5433):
    """Find most similar singers using vector similarity"""
    conn = connect_to_database(port)
    
    try:
        query = """
            SELECT DISTINCT s1.singer_id, 
                   AVG(1 - (s1.embedding <=> s2.embedding)) as avg_similarity,
                   COUNT(*) as comparisons
            FROM singer_embeddings s1, singer_embeddings s2
            WHERE s2.singer_id = %s
            AND s1.singer_id != s2.singer_id
            GROUP BY s1.singer_id
            ORDER BY avg_similarity DESC
            LIMIT %s;
        """
        
        cursor = conn.cursor()
        cursor.execute(query, (target_singer, top_k))
        results = cursor.fetchall()
        conn.close()
        
        print(f"🔍 TOP {top_k} SINGERS SIMILAR TO {target_singer}:")
        print("="*65)
        print("Rank | Singer ID    | Similarity | Comparisons")
        print("-"*65)
        
        for i, (singer_id, similarity, count) in enumerate(results, 1):
            print(f"{i:4d} | {singer_id:<12} | {similarity:10.3f} | {count:11d}")
        
        return results
        
    except Exception as e:
        print(f"❌ Similarity search error: {e}")
        conn.close()
        return None

def visualize_embeddings(n_samples=300, method='pca', port=5433):
    """Visualize embeddings using dimensionality reduction"""
    
    # Get embeddings
    embeddings_data = explore_embeddings(n_samples=n_samples, port=port)
    
    if not embeddings_data or len(embeddings_data) < 10:
        print("❌ Not enough embeddings for visualization")
        return None
    
    # Prepare data
    embeddings_matrix = np.vstack([d['embedding'] for d in embeddings_data])
    singers = [d['singer_id'] for d in embeddings_data]
    
    print(f"\n📊 Creating {method.upper()} visualization with {len(embeddings_data)} embeddings...")
    
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings_matrix)
        title = f'PCA: Singer Embeddings ({len(embeddings_data)} samples)'
        subtitle = f'Explained variance: {reducer.explained_variance_ratio_.sum():.1%}'
    else:  # t-SNE
        perplexity = min(30, len(embeddings_data)//4)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = reducer.fit_transform(embeddings_matrix)
        title = f't-SNE: Singer Embeddings ({len(embeddings_data)} samples)'
        subtitle = f'Perplexity: {perplexity}'
    
    # Create plot
    fig = px.scatter(
        x=embeddings_2d[:, 0], 
        y=embeddings_2d[:, 1],
        color=singers,
        title=f'{title}<br><sub>{subtitle}</sub>',
        labels={'x': f'{method.upper()} 1', 'y': f'{method.upper()} 2'},
        hover_name=singers,
        width=800, height=600
    )
    
    fig.update_layout(showlegend=False)  # Too many singers for legend
    fig.show()
    
    return embeddings_2d, singers

def quick_stats(port=5433):
    """Quick database statistics"""
    conn = connect_to_database(port)
    
    queries = {
        "🏆 Top 10 singers": """
            SELECT singer_id, COUNT(*) as files 
            FROM singer_embeddings 
            GROUP BY singer_id 
            ORDER BY files DESC 
            LIMIT 10;
        """,
        
        "📊 Database summary": """
            SELECT 
                COUNT(*) as total_embeddings,
                COUNT(DISTINCT singer_id) as unique_singers,
                MIN(created_at) as first_processed,
                MAX(created_at) as last_processed
            FROM singer_embeddings;
        """,
        
        "📈 File count distribution": """
            SELECT 
                CASE 
                    WHEN file_count >= 500 THEN '500+'
                    WHEN file_count >= 200 THEN '200-499'
                    WHEN file_count >= 100 THEN '100-199'
                    WHEN file_count >= 50 THEN '50-99'
                    ELSE '<50'
                END as range,
                COUNT(*) as singers
            FROM (
                SELECT singer_id, COUNT(*) as file_count
                FROM singer_embeddings 
                GROUP BY singer_id
            ) t
            GROUP BY range
            ORDER BY MIN(file_count) DESC;
        """
    }
    
    for title, query in queries.items():
        print(f"\n{title}")
        print("-" * len(title))
        try:
            df = pd.read_sql(query, conn)
            print(df.to_string(index=False))
        except Exception as e:
            print(f"❌ Error: {e}")
    
    conn.close()

def analyze_intra_singer_similarity(singer_id=None, min_files=5, max_singers=None, port=5433):
    """Analyze cosine distances between embeddings of the same singer"""
    from sklearn.metrics.pairwise import cosine_distances
    
    conn = connect_to_database(port)
    
    if singer_id:
        # Analyze specific singer
        singers_to_analyze = [singer_id]
    else:
        # Get singers with at least min_files
        cursor = conn.cursor()
        cursor.execute("""
            SELECT singer_id, COUNT(*) as file_count
            FROM singer_embeddings 
            GROUP BY singer_id 
            HAVING COUNT(*) >= %s
            ORDER BY COUNT(*) DESC;
        """, (min_files,))
        singers_to_analyze = [row[0] for row in cursor.fetchall()]
        
        # Apply max_singers limit if specified
        if max_singers:
            singers_to_analyze = singers_to_analyze[:max_singers]
    
    results = []
    
    total_singers = len(singers_to_analyze)
    print(f"🔍 ANALYZING {total_singers} singers with ≥{min_files} files")
    
    for i, singer in enumerate(singers_to_analyze):
        if i % 10 == 0:
            print(f"Progress: {i+1}/{total_singers} singers")
            
        cursor = conn.cursor()
        cursor.execute("""
            SELECT embedding FROM singer_embeddings 
            WHERE singer_id = %s;
        """, (singer,))
        
        embeddings = []
        for (emb_str,) in cursor.fetchall():
            try:
                if isinstance(emb_str, str):
                    emb_str = emb_str.strip('[]')
                    embedding = np.array([float(x.strip()) for x in emb_str.split(',')])
                else:
                    embedding = np.array(emb_str)
                embeddings.append(embedding)
            except Exception as e:
                print(f"⚠️ Parse error for {singer}: {e}")
                continue
        
        if len(embeddings) < 2:
            continue
            
        # Calculate pairwise cosine distances
        embeddings_matrix = np.vstack(embeddings)
        distances = cosine_distances(embeddings_matrix)
        
        # Get upper triangle (excluding diagonal) to avoid duplicates
        upper_triangle = np.triu(distances, k=1)
        non_zero_distances = upper_triangle[upper_triangle > 0]
        
        if len(non_zero_distances) > 0:
            mean_distance = np.mean(non_zero_distances)
            std_distance = np.std(non_zero_distances)
            min_distance = np.min(non_zero_distances)
            max_distance = np.max(non_zero_distances)
            
            results.append({
                'singer_id': singer,
                'num_files': len(embeddings),
                'num_pairs': len(non_zero_distances),
                'mean_distance': mean_distance,
                'std_distance': std_distance,
                'min_distance': min_distance,
                'max_distance': max_distance,
                'consistency_score': 1 - mean_distance  # Higher is more consistent
            })
    
    conn.close()
    
    if not results:
        print("❌ No valid results found!")
        return None
    
    # Sort by consistency score (most consistent first)
    results.sort(key=lambda x: x['consistency_score'], reverse=True)
    
    print(f"\n📊 INTRA-SINGER SIMILARITY ANALYSIS")
    print(f"Analyzed {len(results)} singers with ≥{min_files} files")
    print("="*90)
    print("Singer ID    | Files | Pairs | Mean Dist | Std Dist  | Consistency | Range")
    print("-"*90)
    
    for result in results:
        print(f"{result['singer_id']:<12} | "
              f"{result['num_files']:5d} | "
              f"{result['num_pairs']:5d} | "
              f"{result['mean_distance']:9.3f} | "
              f"{result['std_distance']:9.3f} | "
              f"{result['consistency_score']:11.3f} | "
              f"{result['min_distance']:.3f}-{result['max_distance']:.3f}")
    
    # Summary statistics
    mean_distances = [r['mean_distance'] for r in results]
    consistency_scores = [r['consistency_score'] for r in results]
    
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"Average mean distance: {np.mean(mean_distances):.3f} (±{np.std(mean_distances):.3f})")
    print(f"Average consistency score: {np.mean(consistency_scores):.3f} (±{np.std(consistency_scores):.3f})")
    print(f"Most consistent singer: {results[0]['singer_id']} (score: {results[0]['consistency_score']:.3f})")
    print(f"Least consistent singer: {results[-1]['singer_id']} (score: {results[-1]['consistency_score']:.3f})")
    
    return results

def compare_intra_vs_inter_similarity(singer_id, n_random_comparisons=100, port=5433):
    """Compare intra-singer vs inter-singer similarities"""
    
    conn = connect_to_database(port)
    
    # Get embeddings for target singer
    cursor = conn.cursor()
    cursor.execute("""
        SELECT embedding FROM singer_embeddings 
        WHERE singer_id = %s;
    """, (singer_id,))
    
    target_embeddings = []
    for (emb_str,) in cursor.fetchall():
        try:
            if isinstance(emb_str, str):
                emb_str = emb_str.strip('[]')
                embedding = np.array([float(x.strip()) for x in emb_str.split(',')])
            else:
                embedding = np.array(emb_str)
            target_embeddings.append(embedding)
        except:
            continue
    
    if len(target_embeddings) < 2:
        print(f"❌ Singer {singer_id} has less than 2 valid embeddings")
        conn.close()
        return None
    
    # Calculate intra-singer distances
    target_matrix = np.vstack(target_embeddings)
    intra_distances = cosine_distances(target_matrix)
    upper_triangle = np.triu(intra_distances, k=1)
    intra_distances_flat = upper_triangle[upper_triangle > 0]
    
    # Get random embeddings from other singers
    cursor.execute("""
        SELECT embedding FROM singer_embeddings 
        WHERE singer_id != %s
        ORDER BY RANDOM()
        LIMIT %s;
    """, (singer_id, n_random_comparisons))
    
    other_embeddings = []
    for (emb_str,) in cursor.fetchall():
        try:
            if isinstance(emb_str, str):
                emb_str = emb_str.strip('[]')
                embedding = np.array([float(x.strip()) for x in emb_str.split(',')])
            else:
                embedding = np.array(emb_str)
            other_embeddings.append(embedding)
        except:
            continue
    
    conn.close()
    
    if len(other_embeddings) < 10:
        print("❌ Not enough other embeddings for comparison")
        return None
    
    # Calculate inter-singer distances (target vs others)
    other_matrix = np.vstack(other_embeddings)
    inter_distances = cosine_distances(target_matrix, other_matrix)
    inter_distances_flat = inter_distances.flatten()
    
    print(f"🔍 INTRA vs INTER SINGER SIMILARITY for {singer_id}")
    print("="*60)
    print(f"Intra-singer (same singer) comparisons: {len(intra_distances_flat)}")
    print(f"Inter-singer (vs others) comparisons: {len(inter_distances_flat)}")
    print()
    print(f"{'Metric':<20} | {'Intra-singer':<12} | {'Inter-singer':<12} | {'Ratio':<8}")
    print("-"*60)
    
    metrics = {
        'Mean distance': (np.mean(intra_distances_flat), np.mean(inter_distances_flat)),
        'Std distance': (np.std(intra_distances_flat), np.std(inter_distances_flat)),
        'Min distance': (np.min(intra_distances_flat), np.min(inter_distances_flat)),
        'Max distance': (np.max(intra_distances_flat), np.max(inter_distances_flat)),
        'Median distance': (np.median(intra_distances_flat), np.median(inter_distances_flat))
    }
    
    for metric, (intra_val, inter_val) in metrics.items():
        ratio = intra_val / inter_val if inter_val != 0 else float('inf')
        print(f"{metric:<20} | {intra_val:12.3f} | {inter_val:12.3f} | {ratio:8.3f}")
    
    print(f"\n📊 INTERPRETATION:")
    intra_mean = np.mean(intra_distances_flat)
    inter_mean = np.mean(inter_distances_flat)
    
    if intra_mean < inter_mean:
        improvement = (inter_mean - intra_mean) / inter_mean * 100
        print(f"✅ Good singer consistency! Intra-singer distances are {improvement:.1f}% lower than inter-singer")
    else:
        problem = (intra_mean - inter_mean) / inter_mean * 100
        print(f"⚠️ Poor singer consistency! Intra-singer distances are {problem:.1f}% higher than inter-singer")
    
    return {
        'singer_id': singer_id,
        'intra_distances': intra_distances_flat,
        'inter_distances': inter_distances_flat,
        'intra_mean': intra_mean,
        'inter_mean': inter_mean
    }

def analyze_entire_database_similarity(port=5433):
    """Analyze intra-singer similarity for ALL singers in the database"""
    from sklearn.metrics.pairwise import cosine_distances
    
    conn = connect_to_database(port)
    
    # Get ALL singers
    cursor = conn.cursor()
    cursor.execute("""
        SELECT singer_id, COUNT(*) as file_count
        FROM singer_embeddings 
        GROUP BY singer_id 
        ORDER BY COUNT(*) DESC;
    """)
    all_singers = cursor.fetchall()
    
    print(f"🔍 ANALYZING ENTIRE DATABASE: {len(all_singers)} singers")
    print("="*100)
    
    results = []
    all_distances = []
    singers_with_multiple_files = 0
    total_pairs = 0
    
    for i, (singer_id, file_count) in enumerate(all_singers):
        if i % 10 == 0:
            print(f"Processing singer {i+1}/{len(all_singers)}: {singer_id}")
        
        if file_count < 2:
            # Skip singers with only 1 file
            results.append({
                'singer_id': singer_id,
                'num_files': file_count,
                'num_pairs': 0,
                'mean_distance': None,
                'std_distance': None,
                'min_distance': None,
                'max_distance': None,
                'consistency_score': None
            })
            continue
        
        # Get embeddings for this singer
        cursor.execute("""
            SELECT embedding FROM singer_embeddings 
            WHERE singer_id = %s;
        """, (singer_id,))
        
        embeddings = []
        for (emb_str,) in cursor.fetchall():
            try:
                if isinstance(emb_str, str):
                    emb_str = emb_str.strip('[]')
                    embedding = np.array([float(x.strip()) for x in emb_str.split(',')])
                else:
                    embedding = np.array(emb_str)
                embeddings.append(embedding)
            except Exception as e:
                continue
        
        if len(embeddings) < 2:
            results.append({
                'singer_id': singer_id,
                'num_files': len(embeddings),
                'num_pairs': 0,
                'mean_distance': None,
                'std_distance': None,
                'min_distance': None,
                'max_distance': None,
                'consistency_score': None
            })
            continue
        
        # Calculate pairwise cosine distances
        embeddings_matrix = np.vstack(embeddings)
        distances = cosine_distances(embeddings_matrix)
        
        # Get upper triangle (excluding diagonal) to avoid duplicates
        upper_triangle = np.triu(distances, k=1)
        non_zero_distances = upper_triangle[upper_triangle > 0]
        
        if len(non_zero_distances) > 0:
            mean_distance = np.mean(non_zero_distances)
            std_distance = np.std(non_zero_distances)
            min_distance = np.min(non_zero_distances)
            max_distance = np.max(non_zero_distances)
            
            results.append({
                'singer_id': singer_id,
                'num_files': len(embeddings),
                'num_pairs': len(non_zero_distances),
                'mean_distance': mean_distance,
                'std_distance': std_distance,
                'min_distance': min_distance,
                'max_distance': max_distance,
                'consistency_score': 1 - mean_distance
            })
            
            # Add to global statistics
            all_distances.extend(non_zero_distances)
            singers_with_multiple_files += 1
            total_pairs += len(non_zero_distances)
    
    conn.close()
    
    # Filter valid results (singers with multiple files)
    valid_results = [r for r in results if r['mean_distance'] is not None]
    
    print(f"\n📊 DATABASE-WIDE STATISTICS:")
    print("="*60)
    print(f"Total singers analyzed: {len(all_singers)}")
    print(f"Singers with multiple files: {singers_with_multiple_files}")
    print(f"Singers with single files: {len(all_singers) - singers_with_multiple_files}")
    print(f"Total pairwise comparisons: {total_pairs:,}")
    print(f"Total embedding pairs: {len(all_distances):,}")
    
    if all_distances:
        global_mean = np.mean(all_distances)
        global_std = np.std(all_distances)
        global_median = np.median(all_distances)
        global_min = np.min(all_distances)
        global_max = np.max(all_distances)
        
        print(f"\n🌍 GLOBAL INTRA-SINGER DISTANCE STATISTICS:")
        print(f"Mean distance (all pairs): {global_mean:.4f}")
        print(f"Standard deviation: {global_std:.4f}")
        print(f"Median distance: {global_median:.4f}")
        print(f"Min distance: {global_min:.4f}")
        print(f"Max distance: {global_max:.4f}")
        print(f"Global consistency score: {1-global_mean:.4f}")
        
        # Percentiles
        p25 = np.percentile(all_distances, 25)
        p75 = np.percentile(all_distances, 75)
        print(f"25th percentile: {p25:.4f}")
        print(f"75th percentile: {p75:.4f}")
        print(f"Interquartile range: {p75-p25:.4f}")
    
    if valid_results:
        # Per-singer statistics
        singer_means = [r['mean_distance'] for r in valid_results]
        singer_stds = [r['std_distance'] for r in valid_results]
        consistency_scores = [r['consistency_score'] for r in valid_results]
        
        print(f"\n👥 PER-SINGER STATISTICS:")
        print(f"Average mean distance per singer: {np.mean(singer_means):.4f} (±{np.std(singer_means):.4f})")
        print(f"Average std distance per singer: {np.mean(singer_stds):.4f} (±{np.std(singer_stds):.4f})")
        print(f"Average consistency score: {np.mean(consistency_scores):.4f} (±{np.std(consistency_scores):.4f})")
        
        # Best and worst singers
        valid_results.sort(key=lambda x: x['consistency_score'], reverse=True)
        
        print(f"\n🏆 TOP 10 MOST CONSISTENT SINGERS:")
        print("Singer ID    | Files | Pairs | Mean Dist | Std Dist  | Consistency")
        print("-"*70)
        for r in valid_results[:10]:
            print(f"{r['singer_id']:<12} | "
                  f"{r['num_files']:5d} | "
                  f"{r['num_pairs']:5d} | "
                  f"{r['mean_distance']:9.3f} | "
                  f"{r['std_distance']:9.3f} | "
                  f"{r['consistency_score']:11.3f}")
        
        print(f"\n⚠️ TOP 10 LEAST CONSISTENT SINGERS:")
        print("Singer ID    | Files | Pairs | Mean Dist | Std Dist  | Consistency")
        print("-"*70)
        for r in valid_results[-10:]:
            print(f"{r['singer_id']:<12} | "
                  f"{r['num_files']:5d} | "
                  f"{r['num_pairs']:5d} | "
                  f"{r['mean_distance']:9.3f} | "
                  f"{r['std_distance']:9.3f} | "
                  f"{r['consistency_score']:11.3f}")
    
    return {
        'all_results': results,
        'valid_results': valid_results,
        'all_distances': all_distances,
        'global_stats': {
            'mean': global_mean if all_distances else None,
            'std': global_std if all_distances else None,
            'median': global_median if all_distances else None,
            'min': global_min if all_distances else None,
            'max': global_max if all_distances else None,
            'total_pairs': len(all_distances)
        }
    }

def analyze_global_embedding_distribution(n_samples=None, port=5433):
    """Analyze the global distribution of ALL embeddings regardless of singer"""
    from sklearn.metrics.pairwise import cosine_distances
    
    conn = connect_to_database(port)
    
    # Get all embeddings
    if n_samples:
        query = """
            SELECT singer_id, embedding FROM singer_embeddings 
            ORDER BY RANDOM()
            LIMIT %s;
        """
        cursor = conn.cursor()
        cursor.execute(query, (n_samples,))
        print(f"🌍 ANALYZING RANDOM SAMPLE: {n_samples} embeddings")
    else:
        query = """
            SELECT singer_id, embedding FROM singer_embeddings;
        """
        cursor = conn.cursor()
        cursor.execute(query)
        print(f"🌍 ANALYZING ALL EMBEDDINGS IN DATABASE")
    
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        print("❌ No embeddings found!")
        return None
    
    print(f"Retrieved {len(results)} embeddings")
    
    # Parse embeddings
    embeddings = []
    singers = []
    
    print("Parsing embeddings...")
    for i, (singer_id, emb_str) in enumerate(results):
        if i % 1000 == 0:
            print(f"  Parsed {i}/{len(results)} embeddings")
        
        try:
            if isinstance(emb_str, str):
                emb_str = emb_str.strip('[]')
                embedding = np.array([float(x.strip()) for x in emb_str.split(',')])
            else:
                embedding = np.array(emb_str)
            
            embeddings.append(embedding)
            singers.append(singer_id)
        except Exception as e:
            continue
    
    if len(embeddings) < 2:
        print("❌ Not enough valid embeddings!")
        return None
    
    print(f"✅ Successfully parsed {len(embeddings)} embeddings")
    print(f"📏 Embedding dimension: {len(embeddings[0])}")
    
    # Convert to matrix
    embeddings_matrix = np.vstack(embeddings)
    
    # Calculate basic statistics of embeddings themselves
    print(f"\n📊 EMBEDDING VECTOR STATISTICS:")
    print(f"Mean value across all dimensions: {np.mean(embeddings_matrix):.4f}")
    print(f"Std value across all dimensions: {np.std(embeddings_matrix):.4f}")
    print(f"Min value: {np.min(embeddings_matrix):.4f}")
    print(f"Max value: {np.max(embeddings_matrix):.4f}")
    
    # Calculate norms
    norms = np.linalg.norm(embeddings_matrix, axis=1)
    print(f"\n📐 EMBEDDING NORM STATISTICS:")
    print(f"Mean norm: {np.mean(norms):.4f}")
    print(f"Std norm: {np.std(norms):.4f}")
    print(f"Min norm: {np.min(norms):.4f}")
    print(f"Max norm: {np.max(norms):.4f}")
    
    # For large datasets, sample pairwise distances to avoid memory issues
    if len(embeddings) > 1000:
        print(f"\n⚠️ Large dataset ({len(embeddings)} embeddings)")
        print(f"Sampling pairwise distances to avoid memory issues...")
        
        # Sample a subset for pairwise distance calculation
        sample_size = min(1000, len(embeddings))
        sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings_matrix[sample_indices]
        sample_singers = [singers[i] for i in sample_indices]
        
        print(f"Computing distances for {sample_size} x {sample_size} = {sample_size**2:,} pairs")
        distances = cosine_distances(sample_embeddings)
        
    else:
        print(f"\nComputing distances for {len(embeddings)} x {len(embeddings)} = {len(embeddings)**2:,} pairs")
        distances = cosine_distances(embeddings_matrix)
        sample_singers = singers
    
    # Get upper triangle to avoid duplicates and self-distances
    upper_triangle = np.triu(distances, k=1)
    all_distances = upper_triangle[upper_triangle > 0]
    
    print(f"✅ Calculated {len(all_distances):,} pairwise distances")
    
    # Calculate comprehensive statistics
    print(f"\n🌍 GLOBAL PAIRWISE DISTANCE STATISTICS:")
    print("="*60)
    print(f"Total unique pairs: {len(all_distances):,}")
    print(f"Mean distance: {np.mean(all_distances):.4f}")
    print(f"Standard deviation: {np.std(all_distances):.4f}")
    print(f"Median distance: {np.median(all_distances):.4f}")
    print(f"Min distance: {np.min(all_distances):.4f}")
    print(f"Max distance: {np.max(all_distances):.4f}")
    
    # Percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n📈 DISTANCE PERCENTILES:")
    for p in percentiles:
        val = np.percentile(all_distances, p)
        print(f"{p:2d}th percentile: {val:.4f}")
    
    # Distribution analysis
    print(f"\n📊 DISTANCE DISTRIBUTION:")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(all_distances, bins=bins)
    
    for i in range(len(bins)-1):
        percentage = (hist[i] / len(all_distances)) * 100
        print(f"Distance {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:6,} pairs ({percentage:5.1f}%)")
    
    # Similarity analysis (1 - distance)
    similarities = 1 - all_distances
    print(f"\n🤝 GLOBAL SIMILARITY STATISTICS:")
    print(f"Mean similarity: {np.mean(similarities):.4f}")
    print(f"Std similarity: {np.std(similarities):.4f}")
    print(f"High similarity (>0.8): {np.sum(similarities > 0.8):,} pairs ({np.sum(similarities > 0.8)/len(similarities)*100:.1f}%)")
    print(f"Medium similarity (0.5-0.8): {np.sum((similarities > 0.5) & (similarities <= 0.8)):,} pairs ({np.sum((similarities > 0.5) & (similarities <= 0.8))/len(similarities)*100:.1f}%)")
    print(f"Low similarity (<0.5): {np.sum(similarities <= 0.5):,} pairs ({np.sum(similarities <= 0.5)/len(similarities)*100:.1f}%)")
    
    return {
        'embeddings_matrix': embeddings_matrix,
        'singers': singers,
        'distances': all_distances,
        'similarities': similarities,
        'stats': {
            'mean_distance': np.mean(all_distances),
            'std_distance': np.std(all_distances),
            'median_distance': np.median(all_distances),
            'min_distance': np.min(all_distances),
            'max_distance': np.max(all_distances),
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'total_pairs': len(all_distances)
        }
    }

def analyze_nearest_neighbors_accuracy(n_neighbors=5, sample_size=None, port=5433):
    """
    For each embedding, find n nearest neighbors and check if any belong to the same singer
    Excludes singers with only 1 file since they cannot have same-singer neighbors
    
    Args:
        n_neighbors: Number of nearest neighbors to check
        sample_size: If specified, only analyze a random sample of embeddings (after filtering)
        port: Database port
    
    Returns:
        Dictionary with accuracy statistics and detailed results
    """
    from sklearn.metrics.pairwise import cosine_distances
    
    conn = connect_to_database(port)
    
    # First, get singer file counts to exclude singers with only 1 file
    cursor = conn.cursor()
    cursor.execute("""
        SELECT singer_id, COUNT(*) as file_count
        FROM singer_embeddings 
        GROUP BY singer_id;
    """)
    singer_counts = cursor.fetchall()
    
    # Get singers with 2+ files
    valid_singers = [singer_id for singer_id, count in singer_counts if count >= 2]
    singers_excluded = [singer_id for singer_id, count in singer_counts if count == 1]
    
    print(f"📊 FILTERING SINGERS:")
    print(f"Total singers in database: {len(singer_counts)}")
    print(f"Singers with 2+ files (included): {len(valid_singers)}")
    print(f"Singers with 1 file (excluded): {len(singers_excluded)}")
    if singers_excluded:
        print(f"Excluded singers: {singers_excluded}")
    
    # Get embeddings only from singers with 2+ files
    if sample_size:
        query = """
            SELECT singer_id, audio_file_path, embedding 
            FROM singer_embeddings 
            WHERE singer_id = ANY(%s)
            ORDER BY RANDOM()
            LIMIT %s;
        """
        cursor = conn.cursor()
        cursor.execute(query, (valid_singers, sample_size))
        print(f"🔍 ANALYZING RANDOM SAMPLE: {sample_size} embeddings from singers with 2+ files")
    else:
        query = """
            SELECT singer_id, audio_file_path, embedding 
            FROM singer_embeddings
            WHERE singer_id = ANY(%s);
        """
        cursor = conn.cursor()
        cursor.execute(query, (valid_singers,))
        print(f"🔍 ANALYZING ALL EMBEDDINGS from singers with 2+ files")
    
    results = cursor.fetchall()
    conn.close()
    
    if len(results) < n_neighbors + 1:
        print(f"❌ Not enough embeddings! Need at least {n_neighbors + 1}, found {len(results)}")
        return None
    
    print(f"Retrieved {len(results)} embeddings from singers with multiple files")
    
    # Parse embeddings
    embeddings = []
    singers = []
    file_paths = []
    
    print("Parsing embeddings...")
    for i, (singer_id, file_path, emb_str) in enumerate(results):
        if i % 1000 == 0 and i > 0:
            print(f"  Parsed {i}/{len(results)} embeddings")
        
        try:
            if isinstance(emb_str, str):
                emb_str = emb_str.strip('[]')
                embedding = np.array([float(x.strip()) for x in emb_str.split(',')])
            else:
                embedding = np.array(emb_str)
            
            embeddings.append(embedding)
            singers.append(singer_id)
            file_paths.append(file_path)
        except Exception as e:
            continue
    
    if len(embeddings) < n_neighbors + 1:
        print(f"❌ Not enough valid embeddings after parsing! Need at least {n_neighbors + 1}, found {len(embeddings)}")
        return None
    
    print(f"✅ Successfully parsed {len(embeddings)} embeddings")
    
    # Convert to matrix for efficient distance computation
    embeddings_matrix = np.vstack(embeddings)
    print(f"📏 Embedding matrix shape: {embeddings_matrix.shape}")
    
    # Calculate all pairwise distances
    print("🔢 Calculating pairwise cosine distances...")
    distances = cosine_distances(embeddings_matrix)
    
    # For each embedding, find nearest neighbors and check accuracy
    correct_predictions = 0
    total_predictions = 0
    detailed_results = []
    
    print(f"🎯 Finding {n_neighbors} nearest neighbors for each embedding...")
    
    for i in range(len(embeddings)):
        if i % 500 == 0:
            print(f"  Processing embedding {i+1}/{len(embeddings)}")
        
        # Get distances to all other embeddings (excluding self)
        embedding_distances = distances[i]
        
        # Find indices of n nearest neighbors (excluding self at index i)
        # argsort gives indices sorted by distance
        sorted_indices = np.argsort(embedding_distances)
        
        # Remove self-index and take first n_neighbors
        nearest_indices = [idx for idx in sorted_indices if idx != i][:n_neighbors]
        
        # Check if any nearest neighbor belongs to the same singer
        target_singer = singers[i]
        neighbor_singers = [singers[idx] for idx in nearest_indices]
        
        # Check if target singer appears in neighbors
        same_singer_found = target_singer in neighbor_singers
        
        # Count how many neighbors are from the same singer
        same_singer_count = neighbor_singers.count(target_singer)
        
        detailed_results.append({
            'target_index': i,
            'target_singer': target_singer,
            'target_file': file_paths[i].split('/')[-1],  # Just filename
            'neighbor_singers': neighbor_singers,
            'neighbor_distances': [embedding_distances[idx] for idx in nearest_indices],
            'same_singer_found': same_singer_found,
            'same_singer_count': same_singer_count
        })
        
        if same_singer_found:
            correct_predictions += 1
        total_predictions += 1
    
    # Calculate statistics
    accuracy = correct_predictions / total_predictions
    
    # Additional statistics
    same_singer_counts = [r['same_singer_count'] for r in detailed_results]
    avg_same_singer_count = np.mean(same_singer_counts)
    
    # Per-singer statistics
    singer_stats = {}
    for singer in set(singers):
        singer_results = [r for r in detailed_results if r['target_singer'] == singer]
        if singer_results:
            singer_correct = sum(1 for r in singer_results if r['same_singer_found'])
            singer_accuracy = singer_correct / len(singer_results)
            singer_stats[singer] = {
                'total_embeddings': len(singer_results),
                'correct_predictions': singer_correct,
                'accuracy': singer_accuracy,
                'avg_same_singer_count': np.mean([r['same_singer_count'] for r in singer_results])
            }
    
    print(f"\n📊 NEAREST NEIGHBORS ANALYSIS RESULTS:")
    print("="*70)
    print(f"Total embeddings analyzed: {total_predictions}")
    print(f"Embeddings with correct neighbor: {correct_predictions}")
    print(f"Overall accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Average same-singer neighbors per embedding: {avg_same_singer_count:.2f}")
    print(f"Number of neighbors checked: {n_neighbors}")
    print(f"Note: Only tested embeddings from singers with 2+ files")
    
    # Show best and worst performing singers
    if singer_stats:
        sorted_singers = sorted(singer_stats.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        print(f"\n🏆 TOP 10 BEST PERFORMING SINGERS:")
        print("Singer ID    | Embeddings | Accuracy | Avg Same-Singer Neighbors")
        print("-"*65)
        for singer, stats in sorted_singers[:10]:
            print(f"{singer:<12} | {stats['total_embeddings']:10d} | {stats['accuracy']:8.3f} | {stats['avg_same_singer_count']:21.2f}")
        
        print(f"\n⚠️ TOP 10 WORST PERFORMING SINGERS:")
        print("Singer ID    | Embeddings | Accuracy | Avg Same-Singer Neighbors")
        print("-"*65)
        for singer, stats in sorted_singers[-10:]:
            print(f"{singer:<12} | {stats['total_embeddings']:10d} | {stats['accuracy']:8.3f} | {stats['avg_same_singer_count']:21.2f}")
    
    # Show some example correct and incorrect predictions
    correct_examples = [r for r in detailed_results if r['same_singer_found']]
    incorrect_examples = [r for r in detailed_results if not r['same_singer_found']]
    
    if correct_examples:
        print(f"\n✅ EXAMPLE CORRECT PREDICTIONS:")
        for i, example in enumerate(correct_examples[:3]):
            print(f"  {i+1}. Target: {example['target_singer']} → Neighbors: {example['neighbor_singers']}")
    
    if incorrect_examples:
        print(f"\n❌ EXAMPLE INCORRECT PREDICTIONS:")
        for i, example in enumerate(incorrect_examples[:3]):
            print(f"  {i+1}. Target: {example['target_singer']} → Neighbors: {example['neighbor_singers']}")
    
    return {
        'n_neighbors': n_neighbors,
        'total_embeddings': total_predictions,
        'correct_predictions': correct_predictions,
        'accuracy': accuracy,
        'avg_same_singer_count': avg_same_singer_count,
        'singer_stats': singer_stats,
        'detailed_results': detailed_results,
        'best_singers': sorted_singers[:10] if singer_stats else [],
        'worst_singers': sorted_singers[-10:] if singer_stats else [],
        'excluded_singers': singers_excluded,
        'valid_singers_count': len(valid_singers),
        'excluded_singers_count': len(singers_excluded)
    }

def main():
    parser = argparse.ArgumentParser(description="View your embedding data easily")
    parser.add_argument("--action", choices=['stats', 'singers', 'view', 'export'], 
                       default='stats', help="What to do")
    parser.add_argument("--singer", type=str, help="Singer ID to view/export")
    parser.add_argument("--limit", type=int, default=5, help="Number of files to show")
    parser.add_argument("--output", type=str, help="Output JSON file for export")
    parser.add_argument("--port", type=int, default=5433, help="Database port")
    
    args = parser.parse_args()
    
    # Connect to database
    conn = connect_to_database(args.port)
    if not conn:
        return
    
    try:
        if args.action == 'stats':
            show_basic_stats(conn)
            
        elif args.action == 'singers':
            show_basic_stats(conn)
            list_all_singers(conn)
            
        elif args.action == 'view':
            if not args.singer:
                print("❌ Please specify --singer for viewing data")
                return
            show_singer_data(conn, args.singer, args.limit)
            
        elif args.action == 'export':
            if not args.singer:
                print("❌ Please specify --singer for export")
                return
            output_file = args.output or f"{args.singer}_embeddings.json"
            export_singer_to_json(conn, args.singer, output_file)
            
    finally:
        conn.close()
        print("\n🔐 Database connection closed")

if __name__ == "__main__":
    main() 