# Slurm Job IDs — Successful Final Runs

This file documents the Slurm Job IDs on Midway 3 for the runs whose outputs feed into the final results. Cluster: `midway3.rcc.uchicago.edu`. Account: `macs30113`.

## Stage 3 — Sentence Embeddings

| Field | Value |
|---|---|
| Job ID | **50113902** |
| Partition | `gpu` |
| Node | `midway3-0278` |
| Model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Input | `s3://evahu-beyond-stars-raw/processed/sample/` (2.1M reviews, 15 parquet files) |
| Output | `s3://evahu-beyond-stars-raw/processed/embeddings/embeddings.npy` + `metadata.parquet` |
| Wall time | ~1.5 hours |
| sbatch script | `3_embeddings/slurm_embed.sbatch` |

## Stage 4 — BERTopic Aspect Discovery

| Field | Value |
|---|---|
| Job ID | **50185214** |
| Partition | `caslake` (CPU) |
| Resources | 8 CPUs, 128 GB RAM |
| Pipeline | UMAP (cosine) → HDBSCAN (min_cluster_size=1000, core_dist_n_jobs=1) → cTF-IDF with bigrams + English stop words |
| Input | embeddings + source text from S3 |
| Output | `s3://evahu-beyond-stars-raw/processed/topics/` — 3 × `topics_*.json` (14 + 14 + 14 = 42 topics) + 3 × `topic_assignments_*.parquet` |
| Wall time | ~30 minutes |
| sbatch script | `4_topics/slurm_topics.sbatch` |

### Earlier iteration
- Job 50147001 — first attempt, single-word vectorizer produced unreadable topics dominated by stop words. Reran with `CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=10, max_df=0.95)` as Job 50185214. Kept in commit history as evidence of iterative development.

## Stage 5 — ABSA Inference

| Field | Value |
|---|---|
| Job ID | **50204546** |
| Partition | `gpu` |
| Node | `midway3-0278` |
| Model | `yangheng/deberta-v3-base-absa-v1.1` (slow tokenizer, sentencepiece) |
| Aspects | quality, value, shipping, durability, usability |
| Input | 50K stratified sample per category × 5 aspects = 750K (text, aspect) pairs |
| Output | `s3://evahu-beyond-stars-raw/processed/absa/absa_results.parquet` |
| Wall time | 1h 13min |
| sbatch script | `5_absa/slurm_absa.sbatch` |

### Earlier iterations
- Jobs 50188692, 50189043, 50189581, 50190395 — four failed attempts. Root causes (in order discovered): S3 path assumed single `sample.parquet` but Stage 2 actually wrote Hive-partitioned `sample/category=*/part-*.parquet`; DeBERTa-v3 fast tokenizer required `protobuf`; slow tokenizer required `sentencepiece`. Final fixes in commit history.
