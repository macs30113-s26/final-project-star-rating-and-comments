import argparse
import os
import boto3
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ASPECTS = ['quality', 'value', 'shipping', 'durability', 'usability']
CATEGORIES = ['Beauty_and_Personal_Care', 'Video_Games', 'Electronics']
MODEL_NAME = 'yangheng/deberta-v3-base-absa-v1.1'


def load_sample(s3, bucket, local_dir):
    sample_dir = os.path.join(local_dir, 'sample')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir, exist_ok=True)
        print('Listing partition files', flush=True)
        resp = s3.list_objects_v2(Bucket=bucket, Prefix='processed/sample/')
        keys = [o['Key'] for o in resp.get('Contents', []) if o['Key'].endswith('.parquet')]
        print(f'Downloading {len(keys)} partition files', flush=True)
        for key in keys:
            rel = key[len('processed/sample/'):]
            local = os.path.join(sample_dir, rel)
            os.makedirs(os.path.dirname(local), exist_ok=True)
            print(f'  {key}', flush=True)
            s3.download_file(bucket, key, local)
    return pd.read_parquet(sample_dir)


def stratified_sample(df, n_per_cat, seed):
    out = []
    for cat in CATEGORIES:
        sub = df[df['category'] == cat]
        n = min(n_per_cat, len(sub))
        out.append(sub.sample(n=n, random_state=seed))
    return pd.concat(out).reset_index(drop=True)


def run_absa(texts, aspect, tokenizer, model, device, batch_size):
    labels = []
    scores = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_aspects = [aspect] * len(batch_texts)
        enc = tokenizer(
            batch_texts, batch_aspects,
            padding=True, truncation=True,
            max_length=256, return_tensors='pt',
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)
        for j in range(len(batch_texts)):
            labels.append(int(preds[j]))
            scores.append(float(probs[j][preds[j]]))
        if (i // batch_size) % 100 == 0:
            print(f'  {aspect} {i + len(batch_texts)}/{n}', flush=True)
    return labels, scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--local_dir', required=True)
    parser.add_argument('--n_per_cat', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_prefix', default='processed/absa/')
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    s3 = boto3.client('s3')

    df = load_sample(s3, args.bucket, args.local_dir)
    print(f'Loaded sample: {len(df)} rows', flush=True)
    print(f'Columns: {list(df.columns)}', flush=True)
    print(df['category'].value_counts().to_string(), flush=True)

    sample = stratified_sample(df, args.n_per_cat, args.seed)
    print(f'Stratified sample: {len(sample)} rows', flush=True)
    print(sample['category'].value_counts().to_string(), flush=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    id2label = model.config.id2label
    print(f'Loaded {MODEL_NAME}', flush=True)
    print(f'id2label: {id2label}', flush=True)

    texts = sample['text'].tolist()
    rows = []
    for aspect in ASPECTS:
        print(f'Running ABSA on aspect: {aspect}', flush=True)
        labels, scores = run_absa(texts, aspect, tokenizer, model, device, args.batch_size)
        for i in range(len(sample)):
            rows.append({
                'review_id': sample['review_id'].iloc[i],
                'category': sample['category'].iloc[i],
                'rating': float(sample['rating'].iloc[i]),
                'aspect': aspect,
                'label': id2label[labels[i]],
                'score': scores[i],
            })

    out = pd.DataFrame(rows)
    out_path = os.path.join(args.local_dir, 'absa_results.parquet')
    out.to_parquet(out_path, index=False)
    print(f'Saved {len(out)} rows to {out_path}', flush=True)

    key = args.output_prefix + 'absa_results.parquet'
    s3.upload_file(out_path, args.bucket, key)
    print(f'Uploaded to s3://{args.bucket}/{key}', flush=True)
    print('Done', flush=True)
