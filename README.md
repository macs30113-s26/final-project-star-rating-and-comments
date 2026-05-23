# Beyond the Stars

Detecting hidden dissatisfaction in Amazon reviews using scalable NLP.

Final Project, MACS 30113 Large-Scale Computing for the Social Sciences, Spring 2026.

## Background

I've been bothered for a while by how much information gets thrown away when a product review collapses into a 1-to-5 star rating. People write hundreds of words about a coffee maker and the system summarizes all of it as "4.3 stars." Something has to be lost in that compression, and the more I think about it the more I want to know exactly what.

This came up most concretely when I was shopping for headphones a few months ago. The model I was considering had 4.5 stars overall, looked great. Then I actually read the reviews and noticed something strange: a lot of the 4 and 5 star reviewers were complaining that the case fell apart after a month. They still gave the product high ratings, because the sound quality was good and that's what they cared about most. But the aggregate rating wasn't telling me the truth about the case. If I'd just looked at the number I would have bought a product with a real defect I didn't know about.

That made me wonder how systematic this is. Are there entire categories of products where star ratings are bad at capturing dissatisfaction? Does it depend on what kind of product it is?

## The question I'm trying to answer

The core question: when consumers give a product 4 or 5 stars on Amazon, how often is the review text actually complaining about specific aspects of the product? And does this "hidden dissatisfaction" pattern depend on what kind of product it is?

My expectation, going in, is that the pattern depends a lot on product type. Consumer behavior research has a classic distinction between sensory products (evaluated mostly on how they feel, smell, look), hedonic products (evaluated on whether they are enjoyable), and utilitarian products (evaluated on whether they functionally work). See Hirschman and Holbrook (1982) for the original framing.

I expect hidden dissatisfaction to be highest for sensory products. If a lipstick smells right and looks good on me, I'll forgive a lot of other things. I expect it to be lowest for utilitarian products. If a laptop doesn't turn on, no amount of pretty design will make me rate it 5 stars. Hedonic products like video games should fall in between: an overall enjoyable game can hide specific complaints, but a game that crashes constantly cannot.

To test this I pick one category that represents each evaluation mode:

- **Beauty and Personal Care** for sensory (23.9M reviews)
- **Video Games** for hedonic (4.6M reviews)
- **Electronics** for utilitarian (43.9M reviews)

Total: about 72 million reviews, spanning 1996 to 2023.

The existing literature on this gap between rating numbers and review text is substantial. Hu and Liu (2004) introduced aspect-based opinion mining for exactly this reason. The SemEval ABSA shared tasks (Pontiki et al. 2014-2016) standardized the benchmarks. Most recently, Mehrabani et al. (arXiv:2509.20953, September 2025) used LLMs to do similar discrepancy analysis on app reviews. What I think is new about this project is the combination of (1) the scale, an order of magnitude larger than most prior published studies; (2) the explicit comparison along the sensory/hedonic/utilitarian axis, which has not been tested for rating validity; and (3) the long time window.

## Why scalable computing

The honest answer is volume. 72 million reviews is well past what fits on a laptop. Even simple operations like loading the data into a pandas DataFrame would exhaust memory on most personal machines. At a generous reading rate of 100 reviews per minute, manually checking every review would take over 13 years.

The other reason is GPU inference. The aspect-based sentiment analysis step uses a transformer model (`yangheng/deberta-v3-base-absa-v1.1`), and running it on CPU at this scale would take weeks. Sentence embeddings have the same issue.

I split the work between two platforms based on what each is good at. EMR on AWS gets the distributed batch ETL work because Spark is the cleanest way to push 72M rows through a sequence of filters and joins. Midway 3 gets the GPU work because Midway has free GPU nodes under the academic allocation, while EMR GPU instances cost more than I want to spend out of AWS Academy credits. S3 sits between the two as the shared storage layer.

## The data

The Amazon Reviews 2023 dataset (Hou et al. 2024) is hosted by the McAuley Lab at UCSD. Each review has the following fields:

- `rating` (1.0 to 5.0)
- `title` and `text` (the actual review content)
- `timestamp` (Unix time)
- `helpful_vote` (integer)
- `verified_purchase` (boolean)
- `asin` and `parent_asin` (product IDs)
- `user_id`

The data is a mix of quantitative fields (rating, vote count, timestamp) and qualitative fields (title, body text). The whole project is essentially about bridging these two sides: I use NLP to extract aspect-level sentiment from the qualitative text, then compare those aspect-level signals against the quantitative star rating to find where they diverge.

A real review looks like this:

```json
{
  "rating": 3.0,
  "title": "Meh",
  "text": "These were lightweight and soft but much too small for my liking. I would have preferred two of these together to make one loc. For that reason I will not be repurchasing.",
  "helpful_votes": 0,
  "verified_purchase": true,
  "timestamp": 1634275259292,
  "asin": "B088SZDGXG"
}
```

## Pipeline

The work is broken into seven stages, alternating between AWS and Midway as needed.

**Stage 1, Acquisition.** Mirror the raw JSONL files from the McAuley Lab to my own S3 bucket. Done with `boto3` and HuggingFace `datasets`. Output: raw data on S3.

**Stage 2, ETL and sampling.** PySpark on EMR. I drop reviews shorter than 20 words, keep only verified purchases, and take a stratified sample of about 2M reviews balanced across category by year by rating. The full 72M dataset is too much to push through GPU inference in the time I have, but stratified sampling preserves the structure I need for the comparison. Output: Parquet on S3.

**Stage 3, Sentence embeddings.** Slurm job on Midway 3 GPU. Each review text gets converted to a 384-dim semantic vector using `sentence-transformers/all-MiniLM-L6-v2`. I chose the smaller MiniLM model rather than the larger MPNet variant because at 2M rows the smaller model is fast enough and the accuracy difference probably doesn't matter for downstream clustering.

**Stage 4, Aspect discovery.** BERTopic on the embeddings, separately for each of the three categories (because the aspects are very different between, say, Beauty and Electronics). I run BERTopic, get topic clusters, then go through the top 15 to 20 clusters per category by hand and assign human-readable labels like "battery life," "shipping speed," "scent," etc.

**Stage 5, Aspect sentiment.** This is the most compute-heavy stage. For each review, I score sentiment toward each aspect mentioned in the review using `yangheng/deberta-v3-base-absa-v1.1`. Output is a long-format table: (review_id, aspect, sentiment_score). Slurm GPU job on Midway.

**Stage 6, Aggregation.** Back to EMR. I join the aspect-sentiment scores onto the full sample, then compute the metrics (HDR and RAD, defined below) at the (category, aspect, year) level. Bootstrap confidence intervals if time allows.

**Stage 7, Dashboard.** Streamlit app, deployed on Streamlit Community Cloud. Lets a user pick a category, see aspect-level sentiment heatmaps, drill into specific aspects, see example reviews exhibiting hidden dissatisfaction.

## Metrics

I define two metrics to make "hidden dissatisfaction" precise.

**Hidden Dissatisfaction Rate (HDR).** Given a category and an aspect, HDR is the share of 4 or 5 star reviews that nonetheless express negative sentiment about that aspect. High HDR means people are giving good ratings while complaining. This is the headline metric.

**Rating-Aspect Divergence (RAD).** For each (category, aspect), the gap between mean aspect sentiment and mean normalized overall rating. Large negative RAD means the aspect is consistently worse than the star rating implies. RAD lets me rank aspects within a category by how much "hidden" signal they carry.

The hypothesis (H1 through H4 broadly) is that HDR will be highest in Beauty and Personal Care, lowest in Electronics, with Video Games somewhere in between. I'll also look at how each metric evolves over the 27-year time span.

## Repository structure

```
beyond-stars/
├── README.md
├── requirements.txt
├── docs/
│   ├── architecture.png
│   └── slurm_jobs.md
├── 1_acquisition/
├── 2_etl/
├── 3_embeddings/
├── 4_topics/
├── 5_absa/
├── 6_aggregation/
├── 7_dashboard/
└── notebooks/
```

Each stage folder has its own short README with the specific commands.

## How to run

To reproduce the pipeline end-to-end:

```
# Stage 1: pull data into S3
python 1_acquisition/download_to_s3.py --bucket beyond-stars-raw

# Stage 2: ETL on EMR
python 2_etl/launch_emr_cluster.py

# Stages 3 to 5: GPU work on Midway
sbatch 3_embeddings/slurm_embed.sbatch
# (run aspect discovery interactively after embeddings finish)
sbatch 5_absa/slurm_absa.sbatch

# Stage 6: back to EMR for aggregation
python 6_aggregation/launch_emr_cluster.py

# Stage 7: dashboard
streamlit run 7_dashboard/streamlit_app.py
```

Slurm Job IDs for the actual runs that produced the final results are logged in `docs/slurm_jobs.md`.

## Results

To be filled in once the pipeline finishes. I plan to include the HDR and RAD tables for each category, the top 5 "hidden pain points" per category, time series across the 27-year window, and the URL of the live dashboard.

## Limitations

A few things I want to flag upfront, since I know I will not have time for all of them.

The ABSA model I'm using was trained on restaurant and laptop reviews. Using it on Beauty reviews and Video Games reviews is a domain shift, and I'm relying on the model to transfer reasonably well. Some accuracy loss is expected.

The "hidden dissatisfaction" label is based on a sentiment threshold. I picked the threshold based on what looked reasonable on a small sample of reviews; a more careful version would learn this threshold from a labeled validation set.

The aspect labels in Stage 4 are partly manual. Two different people doing the labeling might end up with slightly different aspect names, even if the underlying clusters are the same. I'll document my labeling choices in the docs folder.

Bootstrap confidence intervals are listed as a Stage 6 nice-to-have, not a guarantee. If I have time after the rest of the pipeline works, I will add them.

## AI disclosure

Consistent with the course's permitted use of AI tools, this project was developed with assistance from Anthropic's Claude during code authoring, debugging support, and editorial drafting. AI assistance was used primarily for routine scaffolding tasks (cloud SDK configuration, Slurm batch scripts, Spark job setup, boilerplate generation) and for review of written sections of this README. The research design, hypothesis framework, choice of product categories along the sensory/hedonic/utilitarian axis, metric formulations (HDR and RAD), platform-allocation decisions, and statistical interpretations are my own. All code was reviewed and tested by me before being committed.

## References

- Hou, Y., Li, J., He, Z., Yan, A., Chen, X., and McAuley, J. (2024). Bridging Language and Items for Retrieval and Recommendation. arXiv:2403.03952.
- Hirschman, E. C. and Holbrook, M. B. (1982). Hedonic Consumption: Emerging Concepts, Methods and Propositions. Journal of Marketing 46(3), 92-101.
- Krishna, A. (2012). An integrative review of sensory marketing. Journal of Consumer Psychology 22(3), 332-351.
- Hu, M. and Liu, B. (2004). Mining and summarizing customer reviews. KDD '04.
- Pontiki, M. et al. (2016). SemEval-2016 Task 5: Aspect Based Sentiment Analysis.
- Mehrabani et al. (2025). Beyond Stars: Bridging the Gap Between Ratings and Review Sentiment with LLM. arXiv:2509.20953.
- Park, S. (2021). Do Hedonic or Utilitarian Types of Online Product Reviews Make Reviews More Helpful? Electronic Commerce Research and Applications.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.
- McAuley Lab, Amazon Reviews 2023. https://amazon-reviews-2023.github.io/
- HuggingFace sentence-transformers/all-MiniLM-L6-v2.
- HuggingFace yangheng/deberta-v3-base-absa-v1.1.
