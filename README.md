# Assignment 7

Compute used: AWS Academy + AWS EMR (emr-6.2.0, m5.xlarge nodes).

Data: 2015 NYC Yellow Cab trips, full year, about 145M rows. Read from the class S3 bucket:

```
s3://css-uchicago/nyc-tlc/trip_data/yellow_tripdata_2015-*.parquet
```

Originally I tried the CSV path at `nyc-tlc/trip_data/csv/` but that folder only has 2019. The 2015 data is in parquet only, one level up. Parquet ended up being faster anyway since Spark can skip columns it doesn't need.

## Files

- `nyc_taxi_eda.ipynb` is the main notebook with 5 plots plus 1 extra, plus 2-sentence interpretation under each one
- `launch_spark_cluster.py` and `terminate_spark_cluster.py` are the boto3 scripts. I had to patch `launch_spark_cluster.py` so that `enable_ssh` waits and polls for the EMR-managed security group to appear before trying to authorize port 22, because on a fresh cluster it sometimes is not there yet
- `runs.txt` has the cluster IDs from each stage

## How I scaled up

I did not just spin up a big cluster and hope for the best. The order was:

1. **2 cores, January 2015 only** (~12.7M rows). Spent most of my time here debugging the basics. Figured out the parquet has the post-2016 schema (PULocationID instead of pickup_longitude/pickup_latitude), so my original plan of a lat/lon heatmap did not work. Rewrote that plot to join PULocationID against the official taxi zone lookup CSV and aggregate to borough. Also caught some rows with fare_amount near 0 that made tip_pct blow up, so I added a `tip_pct < 2.0` cap.

2. **2 cores, still January, with persist**. I tried to `df_clean.persist()` before the plots so Spark would not have to re-scan the parquet six times. The driver kept dying mid-aggregation (Livy reported "Session not found"). Backed out the persist call. Since parquet is column-oriented and the plots only touch a handful of columns each, re-scanning was actually not that bad.

3. **7 cores, full year wildcard** (144.87M rows). Same notebook, just changed the read path to `yellow_tripdata_2015-*.parquet`. Ran in roughly 20 minutes.

Commit history more or less follows these stages.

## Why these visualizations scale

Every plot does its aggregation inside Spark first, and only calls `.toPandas()` on the small aggregated result. The driver never sees the raw 145M rows. So the same notebook would still work on 10x or 100x the data, you would just need a bigger cluster.

| Plot | Spark aggregation | Rows the driver actually sees |
|---|---|---|
| 1. Payment type | `groupBy('payment_type')` | 5 |
| 2. Hour x day-of-week heatmap | `groupBy('pickup_dow', 'pickup_hour')` | 168 |
| 3. Distance bins | 0.5-mi bins, then `groupBy('dist_bin')` | ~60 |
| 4. Borough rollup | PULocationID joined to zone lookup, rolled to Borough | 8 |
| 5. Fare bins with IQR | $1 fare bins, `percentile_approx` | ~100 |
| 6. (extra) Monthly | `groupBy('pickup_month')` | 12 |

## What surprised me

I had assumed Manhattan and the airports would be the high-tipping pickup zones. Manhattan came in around 21% which is in line with the rest of the city, but EWR (Newark) was actually the lowest at 15.6%, with the Bronx right behind at 16.4%. My guess is that flat-rate airport rides break the 20% preset that the meter normally pushes. I updated the Plot 4 interpretation in the notebook so it reflects what the data actually shows, instead of what I expected going in.

## Reproducing this

Launch (final run was 7-core):

```bash
python3 launch_spark_cluster.py \
    --s3_bucket <your-bucket> \
    --primary_count 1 \
    --core_count 7 \
    --instance_type "m5.xlarge" \
    --cluster_name "a7-final-v2"
```

After the script prints the SSH lines, port-forward 9443 to the master, open `https://localhost:9443`, log in as `jovyan` / `jupyter`, upload the notebook, then Kernel > Restart & Clear Output, then Cell > Run All.

One gotcha: matplotlib is not pre-installed on the EMR master, so before running the notebook I had to SSH in once and `sudo pip3 install matplotlib seaborn pandas`. Also, `plt.show()` does not actually render anything under the PySpark kernel, you need `%matplot plt` instead. Both of these cost me time the first run.

Terminate when done:

```bash
python3 terminate_spark_cluster.py --cluster_id j-HF07B3HQ7O43
```

## Plot takeaways, short version

Full 2-sentence interpretation lives next to each plot in the notebook.

1. Cash tips do not get entered into the TLC system, so cash trips read as 0 tip. Have to filter to payment_type == 1 before training next week.
2. Tip % follows a clear hour x day-of-week pattern, peaking on weekday late nights. Both columns look usable as features.
3. Tip % is not monotonic in trip distance, peaks under a mile then dips and plateaus. Bin distance, or use a tree model.
4. Borough carries a real signal but in the opposite direction I expected (airports and Bronx tip less, not more).
5. Mean tip % hugs the 20% line across nearly the whole fare range. Model `tip_pct` rather than `tip_amount`, the percentage is the cleaner target.
6. (extra) Tip % only moves about half a point across the 12 months. Month is probably not worth adding as a feature.
