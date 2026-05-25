import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, FloatType, LongType, BooleanType, IntegerType


SCHEMA = StructType([
    StructField('rating', FloatType(), True),
    StructField('title', StringType(), True),
    StructField('text', StringType(), True),
    StructField('asin', StringType(), True),
    StructField('parent_asin', StringType(), True),
    StructField('user_id', StringType(), True),
    StructField('timestamp', LongType(), True),
    StructField('verified_purchase', BooleanType(), True),
    StructField('helpful_vote', IntegerType(), True),
])


CATEGORIES = ['Beauty_and_Personal_Care', 'Video_Games', 'Electronics']


def load_category(spark, bucket, category):
    path = f's3://{bucket}/raw/{category}.jsonl.gz'
    df = spark.read.schema(SCHEMA).json(path)
    df = df.withColumn('category', F.lit(category))
    df = df.withColumn('year', F.year(F.from_unixtime(F.col('timestamp') / 1000)))
    df = df.withColumn('word_count', F.size(F.split(F.col('text'), r'\s+')))
    return df


def filter_reviews(df):
    return df.filter(
        (F.col('verified_purchase') == True) &
        (F.col('word_count') >= 20) &
        (F.col('rating').isNotNull()) &
        (F.col('text').isNotNull()) &
        (F.col('year').isNotNull())
    )


def stratified_sample(df, n_per_cell):
    w = Window.partitionBy('category', 'rating').orderBy(F.rand(seed=42))
    return (df
        .withColumn('rn', F.row_number().over(w))
        .filter(F.col('rn') <= n_per_cell)
        .drop('rn'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--categories', nargs='+', default=CATEGORIES)
    parser.add_argument('--rows-per-cell', type=int, default=140000)
    args = parser.parse_args()

    spark = SparkSession.builder.appName('beyond-stars-etl').getOrCreate()

    dfs = [load_category(spark, args.bucket, c) for c in args.categories]
    df = dfs[0]
    for other in dfs[1:]:
        df = df.unionByName(other)

    print(f'Raw row count: {df.count()}')

    filtered = filter_reviews(df).cache()
    print(f'After filtering: {filtered.count()}')

    summary = (filtered
        .groupBy('category', 'rating')
        .agg(F.count('*').alias('n'))
        .orderBy('category', 'rating'))
    summary.show(20)

    sample = stratified_sample(filtered, args.rows_per_cell)
    print(f'Sample size: {sample.count()}')

    suffix = '_test' if len(args.categories) < len(CATEGORIES) else ''
    sample_path = f's3://{args.bucket}/processed/sample{suffix}/'
    sample.write.mode('overwrite').partitionBy('category').parquet(sample_path)
    print(f'Sample written to {sample_path}')

    full_path = f's3://{args.bucket}/processed/full{suffix}/'
    filtered.write.mode('overwrite').partitionBy('category').parquet(full_path)
    print(f'Full filtered data written to {full_path}')

    spark.stop()


if __name__ == '__main__':
    main()
