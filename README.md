# Final Project

**Authors:** 

CJ Jones | Georgetown University | ANLY-5800 | Fall '25*

Tianyu Zhao | Georgetown University | ANLY-5800 | Fall '25*

---

## [View Published Project Report](https://gu-anly-5800-final-project.onrender.com)


## Overview

The goal of this project is to convert natural-language questions into correct and executable SQL queries. We evaluate whether a small, low-parameter language model, Gemma-3 4B, can effectively perform text-to-SQL tasks. We also examine whether LoRA fine-tuning can improve SQL generation accuracy without requiring full model retraining.

* Many real applications rely on SQL, and Text-to-SQL models help non-technical users query databases by converting natural-language questions into SQL.

* Base LLMs often generate invalid or incorrect SQL, so we test whether LoRA fine tuning can teach a Gemma 3-4B to produce accurate queries on the Wiki_SQL dataset.

* We evaluate improvements in SQL validity, execution correctness, and semantic similarity after fine tuning.

## Dataset

This project uses the WikiSQL dataset.

* Each example is processed into an instruction-style prompt that includes the question, table name, and column headers.

* Only single-table SQL queries are included, matching the structure of WikiSQL.

* The dataset is split into training and evaluation sets.

* During evaluation, SQLite databases are dynamically created from table metadata to allow execution-based comparison between predicted and gold SQL queries.

## Running Experiments

All experiments, including baseline evaluation and LoRA fine-tuning, are conducted within the same Jupyter notebook:

[Trainer Notebook](notebooks/DSAN_5800_Final_Project_Trainer_v2_Clean.ipynb)

* The baseline model corresponds to the pretrained Gemma model before LoRA adapters are applied.

* The LoRA model is created by applying LoRA adapters (rank 32, alpha 16) and fine-tuning on the WikiSQL dataset.

* Both models are evaluated by generating SQL queries and executing them on dynamically constructed SQLite databases.

* Execution correctness, training behavior, and similarity metrics are recorded and compared within the same pipeline.

## Evaluation

All result figures and tables shown in the Results section are generated from this notebook:

[Evaluation Notebook](notebooks/DSAN_5800_Final_Project_Evaluation_Notebook_Clean.ipynb)

* Evaluates both the baseline model and the LoRA-fine-tuned model on the WikiSQL evaluation set.

* Produces SQL prediction error breakdown plots (Correct, Wrong Result, SQL Error).

* Computes execution correctness by comparing query results with gold results.

* Computes text-based metrics including BLEU, ROUGE-L, Exact Match, and Token F1.

* Produces semantic similarity plots using Jaccard similarity and normalized Levenshtein similarity.

* Generates similarity vs execution correctness scatter plots.


