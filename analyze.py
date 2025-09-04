# analyze.py
import os
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

import yake
import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords (first run downloads quietly)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))

# ----------------------------
# Seed taxonomy & high-recall lexicon
# ----------------------------
SEED_TOPICS = {
    "Delivery issue": [
        "late delivery", "order late", "delay in delivery", "took too long", "slow delivery",
        "order never came", "delivery cancelled", "no delivery"
    ],
    "Food stale": [
        "stale food", "cold food", "spoiled", "bad quality", "not fresh", "smelly food",
    ],
    "Delivery partner rude": [
        "delivery guy was rude", "impolite delivery partner", "behaved badly", "rude behaviour",
        "misbehaved", "unprofessional rider"
    ],
    "Maps not working properly": [
        "maps not working", "location issue", "incorrect address", "gps issue", "map wrong route"
    ],
    "Instamart should be open all night": [
        "instamart 24x7", "instamart all night", "instamart late night", "instamart midnight"
    ],
    "Bring back 10 minute bolt delivery": [
        "bring back 10 minute", "bolt delivery", "10 min delivery", "quick commerce 10 minute"
    ],
}

HIGH_RECALL_PATTERNS = {
    "Refund / cancellation": [
        "refund", "refund not received", "refund pending", "refund delay", "cancellation issue",
        "cancel order problem", "didn't get refund", "refunded late"
    ],
    "Payment / UPI issues": [
        "payment failed", "upi failed", "double charged", "charged twice", "payment pending",
        "payment stuck", "card declined"
    ],
    "App performance / crashes": [
        "app crash", "crashes", "keeps crashing", "hangs", "laggy", "performance issue",
        "loading forever", "stuck on loading"
    ],
    "Coupons / pricing": [
        "coupon not working", "promo code invalid", "price high", "too expensive", "hidden charges",
        "delivery charges high"
    ],
    "Packaging / spillage": [
        "spilled", "leaked", "bad packaging", "package open", "food spilled"
    ],
    "Support responsiveness": [
        "customer support not responding", "no response from support", "chatbot useless",
        "support is slow", "no help"
    ],
}

# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    data_dir: str = "data"
    results_dir: str = "results"
    output_dir: str = "output"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    keyphrases_per_review: int = 3
    cluster_sim_threshold: float = 0.77      # cosine sim threshold for merging phrases
    review_to_topic_threshold: float = 0.47  # sim threshold to assign topic from review text
    phrase_to_topic_threshold: float = 0.60  # sim threshold to attach phrase to seed topic

# ----------------------------
# Utilities
# ----------------------------
def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _read_days(from_date: str, to_date: str, data_dir: str) -> pd.DataFrame:
    start = datetime.fromisoformat(from_date).date()
    end = datetime.fromisoformat(to_date).date()
    days = pd.date_range(start, end, freq="D").date

    frames = []
    missing = []
    for d in days:
        fp = os.path.join(data_dir, f"{d}.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            df["day"] = str(d)
            frames.append(df)
        else:
            missing.append(str(d))
    if missing:
        print(f"ℹ️ Missing {len(missing)} day(s) with no data: {', '.join(missing[:8])}{'…' if len(missing) > 8 else ''}")
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["reviewId","userName","score","content","at_utc","day"])

def _is_mostly_english(text: str) -> bool:
    # Very light filter to avoid fully non-English; keeps Hinglish typically.
    letters = sum(ch.isalpha() for ch in text)
    ascii_letters = sum((ord(ch) < 128 and ch.isalpha()) for ch in text)
    return ascii_letters >= 0.5 * max(1, letters)

# ----------------------------
# Agent components
# ----------------------------
class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return self.model.encode(texts, normalize_embeddings=True)

class SeedTaxonomyAgent:
    def __init__(self, seed_topics: Dict[str, List[str]]):
        self.seed_topics = seed_topics
        self.canonical = list(seed_topics.keys())
        self.flat_map = []  # (phrase, canonical)
        for can, syns in seed_topics.items():
            self.flat_map.append((can.lower(), can))
            for s in syns:
                self.flat_map.append((s.lower(), can))

    def all_canonical(self) -> List[str]:
        return self.canonical

    def matches_seed(self, phrase: str, emb_model: EmbeddingModel,
                     seed_embs: np.ndarray, seed_labels: List[str],
                     sim_th: float) -> Optional[str]:
        """Return canonical topic if phrase semantically close to any seed."""
        vec = emb_model.encode([phrase])[0].reshape(1, -1)
        sims = (vec @ seed_embs.T).flatten()
        j = int(np.argmax(sims))
        if sims[j] >= sim_th:
            return seed_labels[j]
        return None

class CandidateMiningAgent:
    def __init__(self, top_k: int = 3):
        self.kw = yake.KeywordExtractor(lan="en", n=1, top=top_k)  # unigram phrases

    def extract(self, text: str, extra_lex: Dict[str, List[str]]) -> List[str]:
        phrases = []
        # YAKE phrases
        try:
            for kw, _score in self.kw.extract_keywords(text):
                k = kw.strip().lower()
                if k and k not in STOPWORDS:
                    phrases.append(k)
        except Exception:
            pass

        # High-recall lexicon hits
        low = text.lower()
        for _canon, syns in extra_lex.items():
            for s in syns:
                if s in low:
                    phrases.append(s.lower())

        # De-dup while preserving order
        seen = set()
        out = []
        for p in phrases:
            if p not in seen:
                out.append(p)
                seen.add(p)
        return out[:10]  # cap

class ConsolidationAgent:
    def __init__(self, sim_threshold: float):
        self.sim_th = sim_threshold

    def cluster_phrases(self, phrases: List[str], emb_model: EmbeddingModel) -> Dict[int, List[str]]:
        if not phrases:
            return {}
        embs = emb_model.encode(phrases)
        # Distance = 1 - cosine similarity
        dist = 1 - (embs @ embs.T)
        # ✅ sklearn ≥1.4: use metric="precomputed" (affinity was removed)
        clus = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=(1 - self.sim_th),
        )
        labels = clus.fit_predict(dist)
        groups: Dict[int, List[str]] = {}
        for p, lab in zip(phrases, labels):
            groups.setdefault(lab, []).append(p)
        return groups

    def name_cluster(self, cluster_phrases: List[str], emb_model: EmbeddingModel,
                     seed_embs: np.ndarray, seed_labels: List[str]) -> Tuple[str, List[str]]:
        # Try to map to nearest seed canonical
        cl_embs = emb_model.encode(cluster_phrases)
        centroid = np.mean(cl_embs, axis=0, keepdims=True)
        sims = (centroid @ seed_embs.T).flatten()
        j = int(np.argmax(sims))
        if sims[j] >= 0.70:
            return seed_labels[j], cluster_phrases

        # Else create a new canonical label from the phrase closest to centroid
        d = cosine_similarity(centroid, cl_embs).flatten()
        best_idx = int(np.argmax(d))
        label = cluster_phrases[best_idx].strip().lower()
        label = label[:1].upper() + label[1:]
        return label, cluster_phrases

class ReviewAssignerAgent:
    def __init__(self, review_sim_th: float):
        self.review_sim_th = review_sim_th

    def assign(self, reviews: List[str], topics: List[str], emb_model: EmbeddingModel) -> Dict[int, List[int]]:
        """Return mapping: review_index -> [topic_index,...] (1 or more if strong)."""
        if not reviews or not topics:
            return {}
        rev_embs = emb_model.encode(reviews)
        top_embs = emb_model.encode(topics)

        sim = rev_embs @ top_embs.T
        assignment: Dict[int, List[int]] = {}
        for i in range(sim.shape[0]):
            row = sim[i]
            best = np.where(row >= self.review_sim_th)[0].tolist()
            if not best:
                j = int(np.argmax(row))
                if row[j] >= (self.review_sim_th - 0.05):
                    best = [j]
            if best:
                assignment[i] = best
        return assignment

# ----------------------------
# Orchestrator
# ----------------------------
class TrendEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        _ensure_dir(cfg.results_dir)
        _ensure_dir(cfg.output_dir)
        self.emb = EmbeddingModel(cfg.model_name)
        self.seed = SeedTaxonomyAgent(SEED_TOPICS)
        self.miner = CandidateMiningAgent(cfg.keyphrases_per_review)
        self.consol = ConsolidationAgent(cfg.cluster_sim_threshold)
        self.assigner = ReviewAssignerAgent(cfg.review_to_topic_threshold)

        # Pre-embed seed canonicals (include high-recall category names to stabilize mapping)
        self._seed_labels = list(self.seed.all_canonical()) + list(HIGH_RECALL_PATTERNS.keys())
        self._seed_embs = self.emb.encode(self._seed_labels)

    def run(self, target_date: str, horizon_days: int = 30) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        T = datetime.fromisoformat(target_date).date()
        from_date = (T - timedelta(days=horizon_days)).isoformat()
        to_date = T.isoformat()

        raw = _read_days(from_date, to_date, self.cfg.data_dir)
        if raw.empty:
            print("⚠️ No data in range; returning empty report.")
            return pd.DataFrame(), {}

        # Light cleaning + filter
        raw["content"] = raw["content"].fillna("").astype(str)
        raw = raw[raw["content"].str.len() > 0]
        raw = raw[raw["content"].apply(_is_mostly_english)]
        raw = raw.reset_index(drop=True)

        # 1) Mine candidate phrases
        all_phrases = []
        per_review_phrases = []
        for txt in tqdm(raw["content"].tolist(), desc="🔎 Mining phrases"):
            phrases = self.miner.extract(txt, HIGH_RECALL_PATTERNS)
            per_review_phrases.append(phrases)
            all_phrases.extend(phrases)
        all_phrases = list(dict.fromkeys(all_phrases))  # de-dup preserving order

        # 2) Consolidate into clusters and map to canonical topics (dedup)
        clusters = self.consol.cluster_phrases(all_phrases, self.emb)
        topic_to_members: Dict[str, set] = {}
        for _, phs in clusters.items():
            name, members = self.consol.name_cluster(phs, self.emb, self._seed_embs, self._seed_labels)
            topic_to_members.setdefault(name, set()).update(members)

        # 3) Final topic list (include all seed canonicals to stabilize rows)
        final_topics = list(dict.fromkeys(list(self.seed.all_canonical()) + list(topic_to_members.keys())))

        # 4) Assign reviews to topics (semantic similarity)
        assignments = self.assigner.assign(raw["content"].tolist(), final_topics, self.emb)

        # 5) Count per day
        date_index = pd.date_range(from_date, to_date, freq="D").date
        counts = pd.DataFrame(0, index=final_topics, columns=[str(d) for d in date_index], dtype=int)

        for rev_idx, topic_idxs in assignments.items():
            day = raw.iloc[rev_idx]["day"]
            for t_idx in topic_idxs:
                counts.at[final_topics[t_idx], day] += 1

        # Sort topics by total desc, keep non-zero first
        totals = counts.sum(axis=1)
        order = totals.sort_values(ascending=False).index.tolist()
        non_zero = [t for t in order if totals[t] > 0]
        zeros = [t for t in order if totals[t] == 0]
        counts = counts.loc[non_zero + zeros]

        # Prepare metadata
        meta = {
            "target_date": to_date,
            "window_start": from_date,
            "topics": [
                {
                    "topic": t,
                    "phrases": sorted(list(topic_to_members.get(t, [])))[:20],
                    "total": int(totals.get(t, 0)),
                    "first_seen": next((c for c in counts.columns if counts.at[t, c] > 0), None)
                }
                for t in counts.index
            ]
        }
        return counts, meta

def save_outputs(counts: pd.DataFrame, meta: Dict[str, Any], target_date: str,
                 output_dir="output", results_dir="results"):
    if counts.empty:
        print("⚠️ Empty counts; nothing to save.")
        return
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"report_{target_date}.csv")
    counts.to_csv(csv_path)
    json_path = os.path.join(results_dir, f"topics_{target_date}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved report CSV: {csv_path}")
    print(f"✅ Saved topics JSON: {json_path}")
