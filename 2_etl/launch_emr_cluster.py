import argparse
import time
import boto3


def upload_script(s3, bucket, local_path, key):
    s3.upload_file(local_path, bucket, key)
    return f's3://{bucket}/{key}'


def launch_cluster(emr, bucket, etl_uri, extra_args, worker_count, release, ec2_key):
    response = emr.run_job_flow(
        Name='beyond-stars-etl',
        ReleaseLabel=release,
        Applications=[{'Name': 'Spark'}],
        Instances={
            'InstanceGroups': [
                {
                    'Name': 'Master',
                    'Market': 'ON_DEMAND',
                    'InstanceRole': 'MASTER',
                    'InstanceType': 'm5.xlarge',
                    'InstanceCount': 1,
                },
                {
                    'Name': 'Workers',
                    'Market': 'ON_DEMAND',
                    'InstanceRole': 'CORE',
                    'InstanceType': 'm5.xlarge',
                    'InstanceCount': worker_count,
                },
            ],
            'Ec2KeyName': ec2_key,
            'KeepJobFlowAliveWhenNoSteps': False,
            'TerminationProtected': False,
        },
        Steps=[
            {
                'Name': 'Run ETL',
                'ActionOnFailure': 'TERMINATE_CLUSTER',
                'HadoopJarStep': {
                    'Jar': 'command-runner.jar',
                    'Args': [
                        'spark-submit',
                        '--deploy-mode', 'cluster',
                        etl_uri,
                        '--bucket', bucket,
                    ] + extra_args,
                },
            },
        ],
        LogUri=f's3://{bucket}/emr-logs/',
        VisibleToAllUsers=True,
        JobFlowRole='EMR_EC2_DefaultRole',
        ServiceRole='EMR_DefaultRole',
    )
    return response['JobFlowId']


def poll_cluster(emr, cluster_id, interval):
    last_state = None
    while True:
        response = emr.describe_cluster(ClusterId=cluster_id)
        state = response['Cluster']['Status']['State']
        if state != last_state:
            print(f'{time.strftime("%H:%M:%S")}  {state}')
            last_state = state
        if state in ('TERMINATED', 'TERMINATED_WITH_ERRORS'):
            reason = response['Cluster']['Status']['StateChangeReason']
            print(f'Reason: {reason.get("Message", "n/a")}')
            return state
        time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--region', default='us-east-1')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--release', default='emr-6.2.0')
    parser.add_argument('--ec2-key', default='vockey')
    parser.add_argument('--poll-interval', type=int, default=30)
    parser.add_argument('--categories', nargs='+', default=None)
    args = parser.parse_args()

    session = boto3.Session(region_name=args.region)
    s3 = session.client('s3')
    emr = session.client('emr')

    etl_uri = upload_script(s3, args.bucket, '2_etl/etl.py', 'code/etl.py')
    print(f'Uploaded etl.py to {etl_uri}')

    extra = []
    if args.categories:
        extra = ['--categories'] + args.categories

    cluster_id = launch_cluster(emr, args.bucket, etl_uri, extra,
                                args.workers, args.release, args.ec2_key)
    print(f'Cluster launched: {cluster_id}')
    print(f'Console: https://console.aws.amazon.com/elasticmapreduce/home?region={args.region}#cluster-details:{cluster_id}')

    poll_cluster(emr, cluster_id, args.poll_interval)
