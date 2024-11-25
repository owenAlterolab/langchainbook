from dotenv import load_dotenv
load_dotenv()

from langchain_altero import logging
logging.langsmith("CH15-Evaluations")

import pandas as pd

df = pd.read_csv("ragas/data/ragas_synthetic_dataset.csv")
print(df.head())

from datasets import Dataset

# mengonversi DataFrame panda menjadi Dataset Wajah Berpelukan
dataset = Dataset.from_pandas(df)

# Memeriksa dataset
print(dataset)

from datasets import Dataset
import os

# mengonversi DataFrame panda menjadi Dataset Wajah Berpelukan
dataset = Dataset.from_pandas(df)

# tetapkan nama dataset (ubah menjadi apa pun yang Anda inginkan)
dataset_name = "owenalterolab/rag-synthetic-dataset"

# Unggah dataset
dataset.push_to_hub(
    dataset_name,
    private=True,  # private = False untuk menjadikannya dataset publik
    split="dataset_v1",  # masukkan nama split dataset
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)