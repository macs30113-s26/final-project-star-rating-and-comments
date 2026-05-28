import argparse
import json
import os
import boto3
import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer


def list_parquet_keys(s3, bucket, prefix):
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = []
    for obj in response.get('Contents', []):
        if obj['Key'].endswith('.parquet'):
            keys.append(obj['Key'])
    return keys


def extract_category(key):
    for part in key.split('/'):
        if part.startswith('category='):
            return part.split('=')[1]
    return None


def download_files(s3, bucket, keys, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    paths = []
    for key in keys:
        local_path = os.path.join(local_dir, key.replace('/', '_'))
        s3.download_file(bucket, key, local_path)
        paths.append(local_path)
    return paths


def load_source_text(paths, keys):
    frames = []
    for path, key in zip(paths, keys):
        chunk = pd.read_parquet(path)
        chunk['review_id'] = chunk['asin'].astype(str) + '_' + chunk['user_id'].astype(str)
        frames.append(chunk[['review_id', 'text']])
    return pd.concat(frames, ignore_index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--embeddings_key', type=str, default='processed/embeddings/embeddings.npy')
    parser.add_argument('--metadata_key', type=str, default='processed/embeddings/metadata.parquet')
    parser.add_argument('--source_prefix', type=str, default='processed/sample/')
    parser.add_argument('--output_prefix', type=str, default='processed/topics/')
    parser.add_argument('--local_dir', type=str, required=True)
    parser.add_argument('--nr_topics', type=int, default=15)
    parser.add_argument('--min_topic_size', type=int, default=1000)
    parser.add_argument('--top_n_words', type=int, default=10)
    parser.add_argument('--top_n_docs', type=int, default=5)
    args = parser.parse_args()

    s3 = boto3.client('s3')
    os.makedirs(args.local_dir, exist_ok=True)

    print(f'Downloading embeddings from s3://{args.bucket}/{args.embeddings_key}', flush=True)
    local_npy = os.path.join(args.local_dir, 'embeddings.npy')
    s3.download_file(args.bucket, args.embeddings_key, local_npy)
    embeddings = np.load(local_npy)
    print(f'Embeddings shape: {embeddings.shape}', flush=True)

    print(f'Downloading metadata from s3://{args.bucket}/{args.metadata_key}', flush=True)
    local_meta = os.path.join(args.local_dir, 'metadata.parquet')
    s3.download_file(args.bucket, args.metadata_key, local_meta)
    meta = pd.read_parquet(local_meta)
    print(f'Metadata rows: {len(meta)}', flush=True)

    print(f'Listing source parquet files in s3://{args.bucket}/{args.source_prefix}', flush=True)
    source_keys = list_parquet_keys(s3, args.bucket, args.source_prefix)
    print(f'Found {len(source_keys)} source files', flush=True)

    print('Downloading source files', flush=True)
    source_paths = download_files(s3, args.bucket, source_keys, args.local_dir)

    print('Loading source text', flush=True)
    source = load_source_text(source_paths, source_keys)
    print(f'Source rows: {len(source)}', flush=True)

    print('Joining text into metadata via review_id', flush=True)
    text_lookup = source.drop_duplicates('review_id').set_index('review_id')['text']
    meta['text'] = meta['review_id'].map(text_lookup)
    missing = meta['text'].isna().sum()
    print(f'Rows with missing text after join: {missing}', flush=True)

    del source
    del text_lookup

    categories = sorted(meta['category'].unique().tolist())
    print(f'Categories: {categories}', flush=True)

    for cat in categories:
        print(f'\n=== Processing category: {cat} ===', flush=True)
        mask = meta['category'].values == cat
        cat_embeddings = embeddings[mask]
        cat_docs = meta.loc[mask, 'text'].tolist()
        cat_ids = meta.loc[mask, 'review_id'].tolist()
        print(f'{cat}: {len(cat_docs)} reviews', flush=True)

        print(f'Fitting BERTopic (min_topic_size={args.min_topic_size}, nr_topics={args.nr_topics})', flush=True)
        hdbscan_model = HDBSCAN(
            min_cluster_size=args.min_topic_size,
            core_dist_n_jobs=1,
            prediction_data=True,
        )
        vectorizer_model = CountVectorizer(
            stop_words='english',
            min_df=10,
            max_df=0.95,
            ngram_range=(1, 2),
        )
        model = BERTopic(
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=args.nr_topics,
            verbose=True,
        )
        topics, probs = model.fit_transform(cat_docs, cat_embeddings)

        topic_info = model.get_topic_info()
        n_topics = len(topic_info[topic_info['Topic'] != -1])
        n_outliers = int((np.array(topics) == -1).sum())
        print(f'Found {n_topics} topics, {n_outliers} outliers', flush=True)

        topic_records = []
        for _, row in topic_info.iterrows():
            tid = int(row['Topic'])
            if tid == -1:
                continue
            words = [w for w, _ in model.get_topic(tid)][:args.top_n_words]
            doc_indices = [i for i, t in enumerate(topics) if t == tid][:args.top_n_docs]
            example_docs = [cat_docs[i] for i in doc_indices]
            topic_records.append({
                'topic_id': tid,
                'size': int(row['Count']),
                'top_words': words,
                'example_docs': example_docs,
                'label': '',
            })

        assignments = pd.DataFrame({'review_id': cat_ids, 'topic': topics})
        assign_path = os.path.join(args.local_dir, f'topic_assignments_{cat}.parquet')
        assignments.to_parquet(assign_path, index=False)

        json_path = os.path.join(args.local_dir, f'topics_{cat}.json')
        with open(json_path, 'w') as f:
            json.dump(topic_records, f, indent=2)

        print(f'Uploading topics_{cat}.json to S3', flush=True)
        s3.upload_file(json_path, args.bucket, args.output_prefix + f'topics_{cat}.json')

        print(f'Uploading topic_assignments_{cat}.parquet to S3', flush=True)
        s3.upload_file(assign_path, args.bucket, args.output_prefix + f'topic_assignments_{cat}.parquet')

        del cat_embeddings
        del cat_docs
        del model
        del hdbscan_model
        del vectorizer_model

    print('Done', flush=True)
