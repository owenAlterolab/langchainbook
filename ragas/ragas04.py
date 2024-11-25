from dotenv import load_dotenv
load_dotenv()

from langchain_altero import logging
logging.langsmith("CH15-Evaluations")

import pandas as pd

# Daftar pertanyaan dan jawaban
inputs = [
    "Apa kemampuan utama ChatGPT?",
    "Apa saja langkah-langkah untuk menggunakan ChatGPT secara optimal?",
    "Bagaimana cara menerapkan etika dalam penggunaan ChatGPT?",
]

# Daftar jawaban untuk pertanyaan
outputs = [
    "Berkomunikasi dengan manusia dalam berbagai konteks, Belajar tanpa pengawasan dari data, Menghasilkan konten berkualitas tinggi.",
    "Pilih model yang sesuai dengan kebutuhan aplikasi, Berikan data pelatihan yang berkualitas, Optimalkan performa model dengan fine-tuning dan teknik lainnya.",
    "Gunakan ChatGPT dengan tanggung jawab dan tidak melanggar hak privasi, Pelajari cara kerja ChatGPT untuk memahami batasan dan kekurangan, Gunakan untuk tujuan yang jelas dan hindari penyalahgunaan, Pertimbangkan dampak penggunaan pada orang lain dan lingkungan.",
]

# Membuat pasangan pertanyaan dan jawaban
qa_pairs = [{"question": q, "answer": a} for q, a in zip(inputs, outputs)]

# Mengubah menjadi DataFrame
df = pd.DataFrame(qa_pairs)

# Menampilkan DataFrame
print(df.head())

import pandas as pd
from datasets import load_dataset, Dataset
import os

# Mengunduh dataset dari HuggingFace Dataset menggunakan repo_id
dataset = load_dataset(
    "owenalterolab/rag-synthetic-dataset",  # Nama dataset
    token=os.environ["HUGGINGFACEHUB_API_TOKEN"],  # Diperlukan untuk data privat
)

# Menampilkan berdasarkan split dalam dataset
huggingface_df = dataset["dataset_v1"].to_pandas()
print(huggingface_df.head())

from langsmith import Client

client = Client()
dataset_name = "RAG_EVAL_DATASET"


# Membuat fungsi dataset
def create_dataset(client, dataset_name, description=None):
    for dataset in client.list_datasets():
        if dataset.name == dataset_name:
            return dataset

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=description,
    )
    return dataset


# Membuat sebuah dataset
dataset = create_dataset(client, dataset_name)

# Menambahkan contoh ke dataset yang telah dibuat
client.create_examples(
    inputs=[{"question": q} for q in df["question"].tolist()],
    outputs=[{"answer": a} for a in df["answer"].tolist()],
    dataset_id=dataset.id,
)

# Daftar pertanyaan baru
new_questions = [
    "Apa risiko utama dalam penggunaan ChatGPT?",
    "Apa saja cara mengatasi risiko saat menggunakan ChatGPT?",
]

# Daftar jawaban baru
new_answers = [
    "Risiko utama adalah kualitas keluaran yang tidak selalu akurat, yang dapat menyebabkan kesalahan dalam pengambilan keputusan. Pengguna perlu melakukan evaluasi dan verifikasi terhadap output sebelum menggunakannya.",
    "Lindungi sistem dan data pelatihan dengan enkripsi, menggunakan dataset yang lebih representatif, sediakan dokumen untuk menginterprestasikan output ChatGPT               ",
]

# Memeriksa versi yang diperbarui di UI
client.create_examples(
    inputs=[{"question": q} for q in new_questions],
    outputs=[{"answer": a} for a in new_answers],
    dataset_id=dataset.id,
)