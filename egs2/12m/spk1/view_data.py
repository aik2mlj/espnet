#!/usr/bin/env python3
"""
view_data.py – exploration, statistics, and analysis utilities for a pgvector
table of singer embeddings.

Edit TABLE_NAME once to point at any compatible table.
"""

# ────────── Imports ──────────
import psycopg2, json, argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px, plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.metrics.pairwise import cosine_distances

# ────────── Global settings ──────────
TABLE_NAME   = "singfake_embeddings"   # <— change here if you switch tables
DEFAULT_PORT = 5433

# ════════════════════════════════════════════════════════════════════
#  Database helpers
# ════════════════════════════════════════════════════════════════════
def connect_to_database(port: int = DEFAULT_PORT):
    try:
        conn = psycopg2.connect(
            host="localhost", port=port,
            database="local_db", user="postgres", password="postgres"
        )
        print("Connected to database successfully!")
        return conn
    except Exception as e:
        print("Error connecting:", e); return None

# ════════════════════════════════════════════════════════════════════
#  Basic stats & listings
# ════════════════════════════════════════════════════════════════════
def show_basic_stats(conn):
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
    total = cur.fetchone()[0]
    cur.execute(f"SELECT COUNT(DISTINCT singer_id) FROM {TABLE_NAME};")
    uniq  = cur.fetchone()[0]
    print("YOUR DATA SUMMARY:")
    print(f"Total audio files processed: {total}")
    print(f"Number of different singers : {uniq}")
    if uniq: print(f"Average files per singer   : {total/uniq:.1f}")

def list_all_singers(conn):
    ''' 
    Print all singers and the number of files they have
    '''
    cur = conn.cursor()
    cur.execute(f"""
        SELECT singer_id, COUNT(*) AS file_count
        FROM {TABLE_NAME}
        GROUP BY singer_id
        ORDER BY file_count DESC;
    """)
    rows = cur.fetchall()
    print("\n ALL SINGERS:")
    print("Singer ID        | Number of Files")
    print("-"*35)
    for sid, n in rows:
        print(f"{sid:<15} | {n}")

# ════════════════════════════════════════════════════════════════════
#  View / export single singer
# ════════════════════════════════════════════════════════════════════
def show_singer_data(conn, singer_id, limit=5):
    '''
    Print data for a single singer
    '''
    cur = conn.cursor()
    cur.execute(f"""
        SELECT audio_file_path, embedding, created_at
        FROM {TABLE_NAME}
        WHERE singer_id=%s
        LIMIT %s;
    """, (singer_id, limit))
    rows = cur.fetchall()
    if not rows:
        print(f"No data for {singer_id}"); return
    blob = {"singer_id": singer_id, "files": []}
    for i,(path,emb,ts) in enumerate(rows,1):
        try:
            arr = np.fromstring(emb.strip("[]"),sep=",") if isinstance(emb,str) else np.array(emb)
            preview, dim = arr[:5].tolist(), len(arr)
        except Exception:
            preview, dim = "Parse Error", "?"
        blob["files"].append({
            "file_number": i, "audio_file": path.split("/")[-1],
            "full_path": path, "embedding_preview": preview,
            "embedding_size": dim, "processed_at": str(ts)})
    print(json.dumps(blob, indent=2))

def export_singer_to_json(conn, singer_id, outfile):
    '''
    Export a single singer's data to a JSON file
    '''
    cur = conn.cursor()
    cur.execute(f"""
        SELECT audio_file_path, embedding, created_at
        FROM {TABLE_NAME}
        WHERE singer_id=%s;
    """, (singer_id,))
    rows = cur.fetchall()
    if not rows:
        print("No data"); return
    blob={"singer_id":singer_id,"total_files":len(rows),"embeddings":[]}
    for path,emb,ts in rows:
        try:
            arr = np.fromstring(emb.strip("[]"),sep=",").tolist() if isinstance(emb,str) else emb
        except Exception: arr="Parse Error"
        blob["embeddings"].append({"audio_file":path,"embedding":arr,"processed_at":str(ts)})
    with open(outfile,"w") as f: json.dump(blob,f,indent=2)
    print(f"✅ Exported {len(rows)} → {outfile}")

# ════════════════════════════════════════════════════════════════════
#  Quick table snapshot
# ════════════════════════════════════════════════════════════════════
def quick_stats(port: int = DEFAULT_PORT):
    conn = connect_to_database(port)
    queries = {
        "Top 10 singers": f"""
            SELECT singer_id, COUNT(*) AS files
            FROM {TABLE_NAME}
            GROUP BY singer_id
            ORDER BY files DESC
            LIMIT 10;
        """,
        "Database summary": f"""
            SELECT COUNT(*)                  AS total_embeddings,
                   COUNT(DISTINCT singer_id) AS unique_singers,
                   MIN(created_at)           AS first_processed,
                   MAX(created_at)           AS last_processed
            FROM {TABLE_NAME};
        """,
        "File count distribution": f"""
            WITH file_counts AS (
                SELECT singer_id, COUNT(*) AS n
                FROM {TABLE_NAME}
                GROUP BY singer_id)
            SELECT CASE
                     WHEN n>=500 THEN '500+'
                     WHEN n>=200 THEN '200-499'
                     WHEN n>=100 THEN '100-199'
                     WHEN n>= 50 THEN '50-99'
                     ELSE '<50' END AS range,
                   COUNT(*) AS singers
            FROM file_counts
            GROUP BY range
            ORDER BY MIN(n) DESC;
        """
    }
    for title,q in queries.items():
        print(f"\n{title}\n{'-'*len(title)}")
        try: print(pd.read_sql(q,conn).to_string(index=False))
        except Exception as e: print("Error:",e)
    conn.close()

# ════════════════════════════════════════════════════════════════════
#  Dataframes & dashboard
# ════════════════════════════════════════════════════════════════════
def get_singers_dataframe(port=DEFAULT_PORT):
    conn=connect_to_database(port)
    df=pd.read_sql(f"""
        SELECT singer_id, COUNT(*) AS file_count
        FROM {TABLE_NAME}
        GROUP BY singer_id
        ORDER BY file_count DESC;
    """, conn); conn.close(); return df

def create_dashboard(port=DEFAULT_PORT):
    conn=connect_to_database(port)
    df_singers=pd.read_sql(f"""
        SELECT singer_id, COUNT(*) AS file_count
        FROM {TABLE_NAME}
        GROUP BY singer_id
        ORDER BY file_count DESC;
    """, conn)
    df_time=pd.read_sql(f"""
        SELECT DATE_TRUNC('hour',created_at) AS hour,
               COUNT(*) AS files
        FROM {TABLE_NAME}
        GROUP BY DATE_TRUNC('hour',created_at)
        ORDER BY hour;
    """, conn); conn.close()

    fig=make_subplots(rows=2,cols=2,subplot_titles=("File Count Distribution",
        "Top 20 Singers","Processing Timeline","Singer Categories"),
        specs=[[{"type":"histogram"},{"type":"bar"}],[{"type":"scatter"},{"type":"pie"}]])
    fig.add_histogram(x=df_singers["file_count"],nbinsx=25,row=1,col=1)
    top20=df_singers.head(20)
    fig.add_bar(x=top20["singer_id"],y=top20["file_count"],row=1,col=2)
    if not df_time.empty:
        fig.add_scatter(x=df_time["hour"],y=df_time["files"],
                        mode="lines+markers",row=2,col=1)
    cats={"High (>200)":(df_singers["file_count"]>200).sum(),
          "Medium (50-200)":((df_singers["file_count"]>=50)&(df_singers["file_count"]<=200)).sum(),
          "Low (<50)":(df_singers["file_count"]<50).sum()}
    fig.add_pie(labels=list(cats.keys()),values=list(cats.values()),row=2,col=2)
    fig.update_layout(height=800,title_text="🎤 Embeddings Dashboard",showlegend=False)
    fig.show()

# ════════════════════════════════════════════════════════════════════
#  Embedding exploration utilities
# ════════════════════════════════════════════════════════════════════
def explore_embeddings(singer_id=None,n_samples=50,port=DEFAULT_PORT):
    conn=connect_to_database(port)
    if singer_id:
        sql=f"SELECT singer_id,audio_file_path,embedding FROM {TABLE_NAME} WHERE singer_id=%s LIMIT %s;"
        args=(singer_id,n_samples)
    else:
        sql=f"SELECT singer_id,audio_file_path,embedding FROM {TABLE_NAME} ORDER BY RANDOM() LIMIT %s;"
        args=(n_samples,)
    cur=conn.cursor(); cur.execute(sql,args); rows=cur.fetchall(); conn.close()
    if not rows: print("No embeddings"); return None
    data=[]
    for sid,path,emb in rows:
        try: arr = np.fromstring(emb.strip("[]"),sep=",") if isinstance(emb,str) else np.array(emb)
        except Exception as e: print("Error:",e); continue
        data.append({"singer_id":sid,"file":path.split("/")[-1],
                     "embedding":arr,"norm":np.linalg.norm(arr),
                     "mean":arr.mean(),"std":arr.std()})
    print(f"Analyzed {len(data)} embeddings dim={len(data[0]['embedding'])}")
    return data

def find_similar_singers(target_singer,top_k=10,port=DEFAULT_PORT):
    '''
    Find the top k most similar singers to a target singer
    '''
    conn=connect_to_database(port)
    cur=conn.cursor()
    cur.execute(f"""
        SELECT DISTINCT s1.singer_id,
               AVG(1-(s1.embedding <=> s2.embedding)) AS sim,
               COUNT(*) AS cmp
        FROM {TABLE_NAME} s1, {TABLE_NAME} s2
        WHERE s2.singer_id=%s AND s1.singer_id!=s2.singer_id
        GROUP BY s1.singer_id
        ORDER BY sim DESC
        LIMIT %s;
    """,(target_singer,top_k))
    rows=cur.fetchall(); conn.close()
    print(f"\nTOP {top_k} similar to {target_singer}")
    print("Rank | Singer ID    | Similarity | Pairs")
    for i,(sid,sim,n) in enumerate(rows,1):
        print(f"{i:4d} | {sid:<12} | {sim:10.3f} | {n:5d}")
    return rows

def visualize_embeddings(n_samples=300,method="pca",port=DEFAULT_PORT):
    '''
    Visualize embeddings using PCA or t-SNE
    '''
    data = explore_embeddings(n_samples=n_samples,port=port)
    if not data or len(data)<10: print("Not enough"); return
    X=np.vstack([d["embedding"] for d in data]); labels=[d["singer_id"] for d in data]
    if method.lower()=="pca":
        reducer=PCA(n_components=2,random_state=42); Y=reducer.fit_transform(X)
        title=f"PCA ({len(data)} samples) – EV {reducer.explained_variance_ratio_.sum():.1%}"
    else:
        reducer=TSNE(n_components=2,random_state=42,perplexity=min(30,len(data)//4))
        Y=reducer.fit_transform(X); title=f"t-SNE ({len(data)} samples)"
    px.scatter(x=Y[:,0],y=Y[:,1],color=labels,hover_name=labels,
               title=title,width=800,height=600).show()

# ════════════════════════════════════════════════════════════════════
#  Heavy analysis functions
#  (all SQL strings updated to use TABLE_NAME)
# ════════════════════════════════════════════════════════════════════
def analyze_intra_singer_similarity(
    singer_id=None, min_files=5, max_singers=None, port=DEFAULT_PORT):
    conn=connect_to_database(port); cur=conn.cursor()
    if singer_id:
        singers=[singer_id]
    else:
        cur.execute(f"""
            SELECT singer_id FROM {TABLE_NAME}
            GROUP BY singer_id HAVING COUNT(*) >= %s
            ORDER BY COUNT(*) DESC;
        """,(min_files,)); singers=[r[0] for r in cur.fetchall()]
        if max_singers: singers=singers[:max_singers]
    from sklearn.metrics.pairwise import cosine_distances
    res=[]; print(f"🔍 ANALYZING {len(singers)} singers ≥{min_files} files")
    for i,sid in enumerate(singers,1):
        if i%10==0: print(f"{i}/{len(singers)} …")
        cur.execute(f"SELECT embedding FROM {TABLE_NAME} WHERE singer_id=%s;",(sid,))
        embs=[]
        for (e,) in cur.fetchall():
            try: embs.append(np.fromstring(e.strip("[]"),sep=",") if isinstance(e,str) else np.array(e))
            except: pass
        if len(embs)<2: continue
        D=cosine_distances(np.vstack(embs)); tri=np.triu(D,k=1); vals=tri[tri>0]
        if vals.size:
            res.append({"singer_id":sid,"num_files":len(embs),"num_pairs":len(vals),
                        "mean_distance":vals.mean(),"std_distance":vals.std(),
                        "min_distance":vals.min(),"max_distance":vals.max(),
                        "consistency_score":1-vals.mean()})
    conn.close()
    if not res: print("❌ No valid results"); return None
    res.sort(key=lambda x:x["consistency_score"],reverse=True)
    print("\nSinger ID | Files | Pairs | MeanDist | Consistency")
    for r in res[:10]:
        print(f"{r['singer_id']:<9} {r['num_files']:6d} {r['num_pairs']:6d} {r['mean_distance']:.3f} {r['consistency_score']:.3f}")
    return res

def compare_intra_vs_inter_similarity(singer_id,n_random=100,port=DEFAULT_PORT):
    conn=connect_to_database(port); cur=conn.cursor()
    cur.execute(f"SELECT embedding FROM {TABLE_NAME} WHERE singer_id=%s;",(singer_id,))
    targ=[np.fromstring(e.strip("[]"),sep=",") if isinstance(e,str) else np.array(e)
          for (e,) in cur.fetchall()]
    if len(targ)<2: print("Need ≥2 embeds"); conn.close(); return
    from sklearn.metrics.pairwise import cosine_distances
    intra=cosine_distances(np.vstack(targ)); intra= intra[np.triu_indices_from(intra,k=1)]
    cur.execute(f"""
        SELECT embedding FROM {TABLE_NAME}
        WHERE singer_id!=%s ORDER BY RANDOM() LIMIT %s;
    """,(singer_id,n_random))
    others=[np.fromstring(e.strip("[]"),sep=",") if isinstance(e,str) else np.array(e)
            for (e,) in cur.fetchall()]
    inter=cosine_distances(np.vstack(targ),np.vstack(others)).flatten()
    conn.close()
    print(f"\nMean intra {intra.mean():.3f} vs inter {inter.mean():.3f}")
    return {"intra":intra,"inter":inter}

def analyze_entire_database_similarity(port=DEFAULT_PORT):
    from sklearn.metrics.pairwise import cosine_distances
    conn=connect_to_database(port); cur=conn.cursor()
    cur.execute(f"SELECT singer_id,embedding FROM {TABLE_NAME};")
    data={}
    for sid,emb in cur.fetchall():
        arr=np.fromstring(emb.strip("[]"),sep=",") if isinstance(emb,str) else np.array(emb)
        data.setdefault(sid,[]).append(arr)
    conn.close()
    all_dist=[]
    for sid,embs in data.items():
        if len(embs)<2: continue
        D=cosine_distances(np.vstack(embs)); tri=np.triu(D,k=1); all_dist.extend(tri[tri>0])
    all_dist=np.array(all_dist)
    print(f"Global mean intra-singer distance: {all_dist.mean():.4f} (N={len(all_dist):,})")
    return all_dist

def analyze_global_embedding_distribution(n_samples=None,port=DEFAULT_PORT):
    conn=connect_to_database(port); cur=conn.cursor()
    if n_samples:
        cur.execute(f"SELECT embedding FROM {TABLE_NAME} ORDER BY RANDOM() LIMIT %s;",(n_samples,))
    else:
        cur.execute(f"SELECT embedding FROM {TABLE_NAME};")
    embs=[np.fromstring(e.strip("[]"),sep=",") if isinstance(e,str) else np.array(e)
          for (e,) in cur.fetchall()]; conn.close()
    X=np.vstack(embs); norms=np.linalg.norm(X,axis=1)
    print(f"Mean norm {norms.mean():.3f} ± {norms.std():.3f}")
    return X

def analyze_nearest_neighbors_accuracy(n_neighbors=5,sample_size=None,port=DEFAULT_PORT):
    from sklearn.metrics.pairwise import cosine_distances
    conn=connect_to_database(port); cur=conn.cursor()
    cur.execute(f"""
        SELECT singer_id, COUNT(*) FROM {TABLE_NAME}
        GROUP BY singer_id;""")
    counts=dict(cur.fetchall())
    valid=[s for s,c in counts.items() if c>=2]
    if sample_size:
        cur.execute(f"""
            SELECT singer_id,audio_file_path,embedding FROM {TABLE_NAME}
            WHERE singer_id=ANY(%s) ORDER BY RANDOM() LIMIT %s;
        """,(valid,sample_size))
    else:
        cur.execute(f"""
            SELECT singer_id,audio_file_path,embedding FROM {TABLE_NAME}
            WHERE singer_id=ANY(%s);""",(valid,))
    rows=cur.fetchall(); conn.close()
    embs=[np.fromstring(e.strip("[]"),sep=",") if isinstance(e,str) else np.array(e) for _,_,e in rows]
    singers=[s for s,_,_ in rows]
    D=cosine_distances(np.vstack(embs))
    correct=0
    for i in range(len(embs)):
        idxs=np.argsort(D[i])[1:n_neighbors+1]
        if singers[i] in {singers[j] for j in idxs}: correct+=1
    acc=correct/len(embs)
    print(f"NN accuracy ({n_neighbors} NN): {acc:.3f} ({correct}/{len(embs)})")
    return acc

# ════════════════════════════════════════════════════════════════════
#  Random-pair sampling utility
# ════════════════════════════════════════════════════════════════════
def sample_random_pair_distance(n_pairs: int = 2000, port: int = DEFAULT_PORT):
    """
    Draws `n_pairs` random, *disjoint* pairs of embeddings and returns the
    mean cosine distance.

    Notes
    -----
    • We fetch 2 × n_pairs embeddings in a single ORDER BY RANDOM() query
      (fast enough for a few thousand rows).
    • If the table has fewer than 2 × n_pairs rows, we analyse as many
      complete pairs as possible and warn the user.
    """
    from sklearn.metrics.pairwise import cosine_distances

    need = n_pairs * 2
    conn = connect_to_database(port)
    cur  = conn.cursor()
    cur.execute(
        f"SELECT embedding FROM {TABLE_NAME} ORDER BY RANDOM() LIMIT %s;",
        (need,))
    rows = cur.fetchall()
    conn.close()

    total_rows = len(rows)
    if total_rows < 2:
        print("❌ Table has fewer than 2 embeddings!")
        return None

    # Adjust pair count if we didn’t get enough rows
    usable_pairs = min(n_pairs, total_rows // 2)
    if usable_pairs < n_pairs:
        print(f"⚠️ Only {total_rows} embeddings fetched – "
              f"computing {usable_pairs} pairs instead of {n_pairs}.")

    # Parse embeddings
    embs = []
    for (e,) in rows[: usable_pairs * 2]:
        try:
            embs.append(
                np.fromstring(e.strip("[]"), sep=",")
                if isinstance(e, str) else np.array(e)
            )
        except Exception as ex:
            print("⚠️ Parse error:", ex)

    if len(embs) < 2:
        print("❌ Could not parse enough embeddings.")
        return None

    # Make disjoint pairs: (0,1), (2,3), …
    A = np.vstack(embs[0::2])
    B = np.vstack(embs[1::2])
    dists = cosine_distances(A, B).diagonal()  # distance of each pair

    mean_dist = float(dists.mean())
    print(f"✅ Mean cosine distance over {len(dists)} random pairs: {mean_dist:.4f}")
    return {"mean_distance": mean_dist, "distances": dists}


# ════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--action", choices=[
    "stats","singers","view","export","quick",
    "intra","intra_vs_inter","db_intra","global_dist","nn_acc",
    "similar","viz","sample_pairs"      #  ← add this
    ], default="stats")

    p.add_argument("--singer"); p.add_argument("--limit",type=int,default=5)
    p.add_argument("--output"); p.add_argument("--port",type=int,default=DEFAULT_PORT)
    p.add_argument("--n",type=int,help="generic numeric arg")
    p.add_argument("--method",choices=["pca","tsne"],default="pca")
    args=p.parse_args()

    if args.action=="quick": quick_stats(args.port); return

    conn=connect_to_database(args.port)
    if not conn: return
    try:
        if args.action=="stats": show_basic_stats(conn)
        elif args.action=="singers": show_basic_stats(conn); list_all_singers(conn)
        elif args.action=="view":
            if not args.singer: print("Need --singer"); return
            show_singer_data(conn,args.singer,args.limit)
        elif args.action=="export":
            if not args.singer: print("Need --singer"); return
            export_singer_to_json(conn,args.singer,args.output or f"{args.singer}.json")
        elif args.action=="similar":
            if not args.singer: print("Need --singer"); return
            find_similar_singers(args.singer, top_k=args.n or 10, port=args.port)
        elif args.action=="viz":
            visualize_embeddings(n_samples=args.n or 300,method=args.method,port=args.port)
        elif args.action=="intra":
            analyze_intra_singer_similarity(singer_id=args.singer,min_files=args.n or 5,port=args.port)
        elif args.action=="intra_vs_inter":
            if not args.singer: print("Need --singer"); return
            compare_intra_vs_inter_similarity(args.singer,n_random=args.n or 100,port=args.port)
        elif args.action=="db_intra":
            analyze_entire_database_similarity(port=args.port)
        elif args.action=="global_dist":
            analyze_global_embedding_distribution(n_samples=args.n,port=args.port)
        elif args.action=="nn_acc":
            analyze_nearest_neighbors_accuracy(
                n_neighbors=args.n or 5,sample_size=None,port=args.port)
        elif args.action == "sample_pairs":
            sample_random_pair_distance(
                n_pairs=args.n or 2000, port=args.port)
    finally:
        conn.close(); print("\n🔐 DB connection closed")

if __name__=="__main__":
    main()
