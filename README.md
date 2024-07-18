# RankQuest

Using the KNRM and TK model as neural re-rankers to evaluate performances 
on msmarco and FiRA dataset. Playing around with a BERT-based QA model

## Content

1.  judgement_aggregation.py: create single aggregated judgements 
    for query-document-pairs based on human judgements and argue 
    why this type of aggregation makes sense.

2.  re_ranking.ipynb: Train 2 neural architectures (KNRM and TK) 
    on the msmarco dataset based on the kernel-pooling paradigm.
    Evaluate on the msmarco sparse labels and the FiRA fine-grained
    labels. For the FiRA dataset, evaluate the aggregated

3.  extractive_qa.py: Uses a pre-trained extractive QA transformer 
    model from the HuggingFace model hub and evaluate the top-1 neural 
    re-ranking result of the MSMARCO FIRA set compared to the gold-label

## Key Lessons

#### TK outperforms KNRM by a mile.

|                                | **KNRM**    | **TK**      |             |              |
|:-------------------------------|:------------|:------------|:------------|:-------------|
|                                | **msmarco** | **msmarco** | **FiRA BL** | **FiRA Own** |
| **MRR@10**                     | 0.179       | 0.25        | 0.83        | 0.96         |
| **Recall@10**                  | 0.27        | 0.48        | 0.89        | 0.95         |
| **QueriesWithNoRelevant@10**   | 1336        | 1062        | 352         | 115          |
| **QueriesWithRelevant@10**     | 664         | 938         | 3823        | 4060         |
| **AverageRankGoldLabel@10**    | 3.52        | 2.63        | 1.27        | 1.09         |
| **MedianRankGoldLabel@10**     | 3           | 2           | 1           | 1            |
| **MRR@20**                     | 0.19        | 0.26        | 0.83        | 0.96         |
| **Recall@20**                  | 0.40        | 0.57        | 0.94        | 1            |
| **QueriesWithNoRelevant@20**   | 1075        | 845         | 351         | 115          |
| **QueriesWithRelevant@20**     | 925         | 1155        | 3824        | 4060         |
| **AverageRankGoldLabel@20**    | 8.16        | 5.45        | 1.27        | 1.09         |
| **MedianRankGoldLabel@20**     | 8           | 3           | 1           | 1            |
| **MRR@1000**                   | 0.20        | 0.27        | 0.83        | 0.96         |
| **Recall@1000**                | 0.70        | 0.65        | 0.94        | 1            |
| **QueriesWithNoRelevant@1000** | 788         | 768         | 351         | 115          |
| **QueriesWithRelevant@1000**   | 1212        | 1232        | 3824        | 4060         |
| **AverageRankGoldLabel@1000**  | 18.43       | 7.29        | 1.27        | 1.09         |
| **MedianRankGoldLabel@1000**   | 19          | 4           | 1           | 1            |
| **nDCG@3**                     | 0.16        | 0.23        | 0.85        | 0.87         |
| **nDCG@5**                     | 0.17        | 0.26        | 0.86        | 0.88         |
| **nDCG@10**                    | 0.19        | 0.30        | 0.88        | 0.90         |
| **nDCG@20**                    | 0.23        | 0.33        | 0.90        | 0.92         |
| **nDCG@1000**                  | 0.29        | 0.34        | 0.90        | 0.92         |
| **MAP@1000**                   | 0.19        | 0.24        | 0.80        | 0.95         |

#### TK is far more computationally expensive

|                           | KNRM       |        | TK         |        |
|:--------------------------|:-----------|:-------|:-----------|:-------|
|                           | Mac M1 CPU | T4 GPU | Mac M1 CPU | T4 GPU |
| Runtime per epoch (hours) | 55         | 4,5    | 30         | 2,5    |

#### Using an evaluation metric on a QA model is not that straight forward

’$66,723 per year’ vs. ’The average salary for structural engineer jobs is $66,000’ should be a correct answer.


