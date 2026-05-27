import argparse
import os
import json
import boto3

CATEGORIES = ['Beauty_and_Personal_Care', 'Video_Games', 'Electronics']


def check_labels(path):
    with open(path) as f:
        topics = json.load(f)
    missing = [t['topic_id'] for t in topics if not t.get('label', '').strip()]
    return missing


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--local_dir', default='4_topics/labels')
    parser.add_argument('--s3_prefix', default='processed/topics/labels/')
    args = parser.parse_args()

    for cat in CATEGORIES:
        local_path = os.path.join(args.local_dir, f'topics_{cat}.json')
        if not os.path.exists(local_path):
            print(f'Missing: {local_path}')
            continue
        missing = check_labels(local_path)
        if missing:
            print(f'{cat}: topic_ids {missing} have empty label, skipping')
            continue
        print(f'{cat}: all labels filled')

    s3 = boto3.client('s3')
    for cat in CATEGORIES:
        local_path = os.path.join(args.local_dir, f'topics_{cat}.json')
        if not os.path.exists(local_path):
            continue
        if check_labels(local_path):
            continue
        key = args.s3_prefix + f'topics_{cat}.json'
        print(f'Uploading {local_path} -> s3://{args.bucket}/{key}')
        s3.upload_file(local_path, args.bucket, key)

    print('Done')
