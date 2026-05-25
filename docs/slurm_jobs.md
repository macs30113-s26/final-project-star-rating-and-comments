# Pipeline runs

Cluster IDs and Slurm job IDs for the runs that produced the final results.

## Stage 2: EMR Spark ETL

### Test run (Video_Games only)
- Cluster ID: `j-11INAPAYO7MLF`
- Start: 2026-05-24 20:48:35 (UTC-05:00)
- End: 2026-05-24 21:03:43
- Workers: 1 x m5.xlarge
- Step: `Run ETL -> COMPLETED`
- Output: `s3://evahu-beyond-stars-raw/processed/sample_test/`, `processed/full_test/`

### Full run (Beauty + Video_Games + Electronics)
- Cluster ID: `j-2A55RFRGQNAAH`
- Start: 2026-05-24 21:08:23 (UTC-05:00)
- End: 2026-05-24 22:06:27
- Duration: ~58 min
- Workers: 2 x m5.xlarge
- Step: `Run ETL -> COMPLETED`
- Output: `s3://evahu-beyond-stars-raw/processed/sample/`, `processed/full/`

## Stage 3: Midway 3 GPU embeddings

(To be added after the embedding job runs.)

## Stage 5: Midway 3 GPU ABSA inference

(To be added after the ABSA job runs.)

## Stage 6: EMR Spark aggregation

(To be added after the aggregation job runs.)
