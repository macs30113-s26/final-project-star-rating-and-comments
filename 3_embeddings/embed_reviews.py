import argparse
import os
import boto3
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


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


def load_dataframe(paths, keys):
    frames = []
    for path, key in zip(paths, keys):
        chunk = pd.read_parquet(path)
        chunk['category'] = extract_category(key)
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--input_prefix', type=str, default='processed/sample/')
    parser.add_argument('--output_prefix', type=str, default='processed/embeddings/')
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--local_dir', type=str, required=True)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    s3 = boto3.client('s3')

    print(f'Listing parquet files in s3://{args.bucket}/{args.input_prefix}', flush=True)
    keys = list_parquet_keys(s3, args.bucket, args.input_prefix)
    print(f'Found {len(keys)} parquet files', flush=True)

    print('Downloading parquet files', flush=True)
    paths = download_files(s3, args.bucket, keys, args.local_dir)

    print('Loading parquet files into pandas', flush=True)
    df = load_dataframe(paths, keys)
    print(f'Total rows: {len(df)}', flush=True)
    print(f'Categories: {df["category"].value_counts().to_dict()}', flush=True)

    if args.limit:
        df = df.head(args.limit).copy()
        print(f'Limited to {len(df)} rows', flush=True)

    df['review_id'] = df['asin'].astype(str) + '_' + df['user_id'].astype(str)

    texts = df['text'].tolist()
    df = df[['review_id', 'category', 'rating']].copy()
    print('Released text column from dataframe', flush=True)

    print(f'CUDA available: {torch.cuda.is_available()}', flush=True)
    print(f'Loading model {args.model_name}', flush=True)
    model = SentenceTransformer(args.model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')

    print('Encoding reviews', flush=True)
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
    print(f'Embeddings shape: {embeddings.shape}', flush=True)

    del texts

    local_npy = os.path.join(args.local_dir, 'embeddings.npy')
    local_meta = os.path.join(args.local_dir, 'metadata.parquet')

    print(f'Saving embeddings to {local_npy}', flush=True)
    np.save(local_npy, embeddings)

    print(f'Saving metadata to {local_meta}', flush=True)
    df.to_parquet(local_meta, index=False)

    npy_key = args.output_prefix + 'embeddings.npy'
    meta_key = args.output_prefix + 'metadata.parquet'

    print(f'Uploading to s3://{args.bucket}/{npy_key}', flush=True)
    s3.upload_file(local_npy, args.bucket, npy_key)

    print(f'Uploading to s3://{args.bucket}/{meta_key}', flush=True)
    s3.upload_file(local_meta, args.bucket, meta_key)

    print('Done', flush=True)
