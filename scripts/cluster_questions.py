import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from collections import Counter

SRC = "data/questions_only.csv"
DST = "data/question_clusters.csv"

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EPS = 0.20
MIN_SAMPLES = 4

def norm(s):
    s = (s or "").strip()
    s = " ".join(s.split())
    return s

def main():
    df = pd.read_csv(SRC)
    df["text_clean"] = df["text_clean"].astype(str).map(norm)
    texts = df["text_clean"].tolist()

    model = SentenceTransformer(MODEL)
    emb = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    cl = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric="cosine", n_jobs=-1)
    labels = cl.fit_predict(emb)
    df["cluster"] = labels

    df2 = df[df["cluster"] != -1].copy()
    sizes = df2["cluster"].value_counts()
    print("Clusters:", (sizes > 0).sum())
    print("Top sizes:", sizes.head(15).tolist())


    rows = []
    for c, g in df2.groupby("cluster"):
        rep = g["text_clean"].value_counts().idxmax()
        rows.append({"cluster": int(c), "cluster_size": int(len(g)), "rep_question": rep})

    out = pd.DataFrame(rows).sort_values("cluster_size", ascending=False)
    out.to_csv(DST, index=False, encoding="utf-8")
    print("Saved:", DST)

if __name__ == "__main__":
    main()