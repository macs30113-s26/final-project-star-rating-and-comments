import os
import sys
import time
import argparse
import boto3
import requests
from tqdm import tqdm
from botocore.exceptions import ClientError


CATEGORIES = ['Beauty_and_Personal_Care', 'Video_Games', 'Electronics']

BASE_URL = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories'


def check_bucket(s3, bucket):
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        code = e.response.get('Error', {}).get('Code', '')
        if code == '404':
            print(f'Bucket {bucket} does not exist. Create it first with:')
            print(f'  aws s3 mb s3://{bucket} --region us-east-1')
        elif code == '403':
            print(f'Bucket {bucket} exists but you do not have access. '
                  'Check your AWS Academy credentials.')
        else:
            print(f'Could not access bucket {bucket}: {e}')
        sys.exit(1)


def download_file(category, local_dir):
    filename = f'{category}.jsonl.gz'
    local_path = os.path.join(local_dir, filename)
    url = f'{BASE_URL}/{filename}'

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        size_gb = os.path.getsize(local_path) / 1e9
        print(f'  already downloaded: {filename} ({size_gb:.2f} GB)')
        return local_path

    print(f'  downloading {filename}')
    r = requests.get(url, stream=True)
    r.raise_for_status()

    total = int(r.headers.get('Content-Length', 0))

    with open(local_path, 'wb') as f:
        bar = tqdm(total=total, unit='B', unit_scale=True, desc=filename)
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))
        bar.close()

    return local_path


def upload_to_s3(local_path, bucket, s3):
    filename = os.path.basename(local_path)
    key = f'raw/{filename}'
    local_size = os.path.getsize(local_path)

    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        if head['ContentLength'] == local_size:
            print(f'  already in S3: s3://{bucket}/{key}')
            return
    except ClientError:
        pass

    print(f'  uploading to s3://{bucket}/{key}')
    bar = tqdm(total=local_size, unit='B', unit_scale=True, desc=filename)
    s3.upload_file(local_path, bucket, key,
                   Callback=lambda n: bar.update(n))
    bar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download Amazon Reviews 2023 data and upload it to S3.')
    parser.add_argument('--bucket', required=True,
                        help='S3 bucket name')
    parser.add_argument('--region', default='us-east-1',
                        help='AWS region')
    parser.add_argument('--local-dir', default='data/raw',
                        help='Local directory for downloaded files')
    parser.add_argument('--categories', nargs='+', default=CATEGORIES,
                        help='Subset of categories to process')
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    s3 = boto3.client('s3', region_name=args.region)
    check_bucket(s3, args.bucket)

    print(f'Processing {len(args.categories)} categories into bucket {args.bucket}')
    start = time.time()

    try:
        for category in args.categories:
            print(f'\n{category}')
            local_path = download_file(category, args.local_dir)
            upload_to_s3(local_path, args.bucket, s3)

        elapsed = (time.time() - start) / 60
        print(f'\nDone. Took {elapsed:.1f} min.')

    except ClientError as e:
        code = e.response.get('Error', {}).get('Code', '')
        if 'Token' in code or 'Expired' in code:
            print(f'\nAWS credentials expired ({code}). '
                  'Refresh your AWS Academy session and rerun.')
        else:
            raise
