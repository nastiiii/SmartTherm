import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

SRC = "data/domain_questions.csv"
OUT = "data/top_question_clusters.csv"

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EPS = 0.18
MIN_SAMPLES = 3
EXAMPLES_PER_CLUSTER = 5

def clean(s: str) -> str:
    s = str(s)
    s = " ".join(s.replace("\n", " ").split())
    return s.strip()

def main():
    df = pd.read_csv(SRC)
    df["text_clean"] = df["text_clean"].astype(str).map(clean)
    texts = df["text_clean"].tolist()

    model = SentenceTransformer(MODEL)
    emb = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    cl = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric="cosine", n_jobs=-1)
    labels = cl.fit_predict(emb)
    df["cluster"] = labels

    df2 = df[df["cluster"] != -1].copy()
    sizes = df2["cluster"].value_counts()

    rows = []
    for c, g in df2.groupby("cluster"):
        g = g.copy()
        rep = g["text_clean"].value_counts().idxmax()
        examples = g["text_clean"].drop_duplicates().head(EXAMPLES_PER_CLUSTER).tolist()
        rows.append({
            "cluster": int(c),
            "cluster_size": int(len(g)),
            "rep_question": rep,
            "examples": " || ".join(examples),
        })

    out = pd.DataFrame(rows).sort_values("cluster_size", ascending=False)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print("Saved:", OUT)
    print("Clusters:", len(out))
    print("Top sizes:", out["cluster_size"].head(15).tolist())
    print("\nTop 10 rep questions:")
    print(out.head(10)[["cluster_size","rep_question"]].to_string(index=False))

if __name__ == "__main__":
    main()