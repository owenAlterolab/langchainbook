# File pengaturan untuk mengelola API key sebagai variabel lingkungan
from dotenv import load_dotenv
from langchain_altero import logging
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from datasets import load_dataset
from matplotlib import pyplot as plt
import open_clip
import pandas as pd
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_altero.models import MultiModal
from langchain_openai import ChatOpenAI
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
from PIL import Image
from IPython.display import HTML, display
from langchain.schema import Document

load_dotenv()
logging.langsmith("CH09-VectorStores")

# Membagi teks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0)

# Memuat file teks -> Mengubahnya ke dalam bentuk List[Document]
loader1 = TextLoader("chapter9/data/nlp-keywords.txt")
loader2 = TextLoader("chapter9/data/finance-keywords.txt")

# Membagi dokumen
split_doc1 = loader1.load_and_split(text_splitter)
split_doc2 = loader2.load_and_split(text_splitter)

# Memeriksa jumlah dokumen
docs_len = len(split_doc1), len(split_doc2)
print(docs_len)

# Tentukan path
DB_PATH = "./chroma_db"

# Membuat DB
db = Chroma.from_documents(
    documents=split_doc1, embedding=OpenAIEmbeddings(), collection_name="my_db"
)

# Simpan dokumen di disk. Tentukan jalur untuk menyimpannya di direktori persist_directory saat disimpan.
persist_db = Chroma.from_documents(
    split_doc1, OpenAIEmbeddings(), persist_directory=DB_PATH, collection_name="my_db"
)

# Memuat dokumen dari disk.
persist_db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=OpenAIEmbeddings(),
    collection_name="my_db",
)

# Memeriksa data yang disimpan
res = persist_db.get()
print(res)

# Memuat dokumen dari disk.
persist_db2 = Chroma(
    persist_directory=DB_PATH,
    embedding_function=OpenAIEmbeddings(),
    collection_name="my_db2",
)

# Memeriksa data yang disimpan
res2 = persist_db2.get()
print(res2)

# Membuat dari list of String
db2 = Chroma.from_texts(
    ["Halo. Senang sekali bertemu dengan Anda.", "Nama saya Teddy."],
    embedding=OpenAIEmbeddings(),
)

res3 = db2.get()
print(res3)

search1 = db.similarity_search("Ceritakan tentang Word embeddings")
print("search1")
print(search1)

search2 = db.similarity_search("Ceritakan tentang Word embeddings", k=2)
print("search2")
print(search2)

# Menggunakan filter
search3 = db.similarity_search(
    "Ceritakan tentang Word embeddings", filter={"source": "/Users/mini16gboffice1/StudioProjects/owen/python/langchainbook/chapter9/data/nlp-keywords.txt"}, k=2
)
print("search3")
print(search3)

# menggunakan filter
search4 = db.similarity_search(
    "Ceritakan tentang Word embeddings", filter={"source": "data/finance-keywords.txt"}, k=2
)
print("\nsearch4")
print(search4)

# Menetapkan page_content, metadata, id
res = db.add_documents(
    [
        Document(
            page_content="Halo! Kali ini kita akan mencoba menambahkan dokumen baru.",
            metadata={"source": "mydata.txt"},
            id="1",
        )
    ]
)

print(res)

# Ambil dokumen dengan id=1
checkDoc = db.get("1")
print(checkDoc)

add_text = db.add_texts(
    ["Saya akan menimpa Dokumen yang sebelumnya ditambahkan.", "Bagaimana hasil dari penimpaan ini?"],
    metadatas=[{"source": "mydata.txt"}, {"source": "mydata.txt"}],
    ids=["1", "2"],
)
print(add_text)

# Ambil data dengan id=1
check = db.get(["1"])
print(check)

print("\n===DELETE===\n")

# hapus id 1
db.delete(ids="1")

# Cari dokumen
check12 = db.get(["1", "2"])
print(check12)

# where 조건으로 metadata 조회
checksource = db.get(where={"source": "mydata.txt"})
print(checksource)

print("\n===RESET===\n")

# Inisialisasi koleksi
db.reset_collection()

# Mengambil dokumen setelah inisialisasi
reset = db.get()
print(reset)

print("\n===RETRIEVER===\n")

# Buat DB
db = Chroma.from_documents(
    documents=split_doc1 + split_doc2,
    embedding=OpenAIEmbeddings(),
    collection_name="nlp",
)

retriever = db.as_retriever()
ret1 = retriever.invoke("Beritahu kami tentang Sentiment Analysis")
print(f"ret1 {ret1}")

retriever = db.as_retriever(
    search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.25, "fetch_k": 10}
)
ret2 = retriever.invoke("Beritahu kami tentang Sentiment Analysis")
print(f"ret2 {ret2}")

retriever = db.as_retriever(search_type = "mmr", search_kwargs = {"k": 2, "fetch_k": 10})
ret3 = retriever.invoke("Ceritakan tentang Sentiment Analysis")
print(f"ret3 {ret3}")

retriever = db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8}
)

ret4 = retriever.invoke("Beritahu kami tentang Sentiment Analysis")
print(f"ret4 {ret4}")

retriever = db.as_retriever(search_kwargs={"k": 1})
ret5 = retriever.invoke("Beritahu kami tentang Sentiment Analysis")
print(f"ret5 {ret5}")

retriever = db.as_retriever(
    search_kwargs={"filter": {"source": "/Users/mini16gboffice1/StudioProjects/owen/python/langchainbook/chapter9/data/finance-keywords.txt"}, "k": 2}
)
ret6 = retriever.invoke("Ceritakan tentang Inflation")
print(f"ret6 {ret6}")

print("\n===MULTIMODAL===\n")

# # Memuat dataset COCO
# dataset = load_dataset(
#     path="detection-datasets/coco", name="default", split="train", streaming=True
# )

# # Folder penyimpanan gambar dan pengaturan jumlah gambar
# IMAGE_FOLDER = "tmp"
# N_IMAGES = 20

# # Pengaturan untuk plotting grafik
# plot_cols = 5
# plot_rows = N_IMAGES // plot_cols
# fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(plot_rows * 2, plot_cols * 2))
# axes = axes.flatten()

# # Menyimpan gambar ke dalam folder dan menampilkannya pada grafik
# dataset_iter = iter(dataset)
# os.makedirs(IMAGE_FOLDER, exist_ok=True)
# for i in range(N_IMAGES):
#     # Ekstraksi gambar dan label dari dataset
#     data = next(dataset_iter)
#     image = data["image"]
#     label = data["objects"]["category"][0]  # Menggunakan kategori objek pertama sebagai label

#     # Menampilkan gambar pada grafik dan menambahkan label
#     axes[i].imshow(image)
#     axes[i].set_title(label, fontsize=8)
#     axes[i].axis("off")

#     # Menyimpan gambar sebagai file
#     image.save(f"{IMAGE_FOLDER}/{i}.jpg")

# # Mengatur layout grafik dan menampilkannya
# # plt.tight_layout()
# # plt.show()

# # menampilkan model/titik pemeriksaan yang tersedia
# pd.DataFrame(open_clip.list_pretrained(), columns=["model_name", "checkpoint"]).head(10)

# # Membuat objek fungsi embedding OpenCLIP
image_embedding_function = OpenCLIPEmbeddings(
    model_name="ViT-H-14-378-quickgelu", checkpoint="dfn5b"
)

# # Menyimpan jalur ke gambar sebagai daftar
image_uris = sorted(
    [
        os.path.join("tmp", image_name)
        for image_name in os.listdir("tmp")
        if image_name.endswith(".jpg")
    ]
)

print(image_uris)


# # Inisialisasi model ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o-mini")

# # Pengaturan model MultiModal
# model = MultiModal(
#     model=llm,
#     system_prompt="Misi Anda adalah untuk menjelaskan gambar secara detail",  # Prompt sistem: Instruksi untuk menjelaskan gambar secara detail
#     user_prompt="Deskripsi harus ditulis dalam satu kalimat (kurang dari 60 karakter)",  # Prompt pengguna: Meminta deskripsi dalam satu kalimat kurang dari 60 karakter
# )

# # Membuat Deskripsi Gambar
# image_desc = model.invoke(image_uris[0])
# print(f"desc: {image_desc}")

# descriptions = dict()

# for image_uri in image_uris:
#     descriptions[image_uri] = model.invoke(image_uri, display_image=False)

# # 생성된 결과물 출력
# print(descriptions)

# # Inisialisasi daftar untuk menyimpan gambar asli, gambar yang telah diproses, dan deskripsi teks
# original_images = []
# images = []
# texts = []

# # Mengatur ukuran grafik (20x10 inci)
# plt.figure(figsize=(20, 10))

# # Memproses file gambar yang disimpan di direktori 'tmp'
# for i, image_uri in enumerate(image_uris):
#     # Membuka file gambar dan mengonversinya ke mode RGB
#     image = Image.open(image_uri).convert("RGB")

#     # Membuat subplot dalam grid 4x5
#     plt.subplot(4, 5, i + 1)

#     # Menampilkan gambar
#     plt.imshow(image)

#     # Menetapkan nama file gambar dan deskripsi sebagai judul
#     plt.title(f"{os.path.basename(image_uri)}\n{descriptions[image_uri]}", fontsize=8)

#     # Menghapus tanda pada sumbu x dan y
#     plt.xticks([])
#     plt.yticks([])

#     # Menambahkan gambar asli, gambar yang telah diproses, dan deskripsi teks ke masing-masing daftar
#     original_images.append(image)
#     images.append(image)
#     texts.append(descriptions[image_uri])

# # Menyesuaikan jarak antar subplot
# plt.tight_layout()


# # Embedding gambar dan teks
# # Ekstraksi fitur gambar menggunakan URI gambar
# img_features = image_embedding_function.embed_image(image_uris)
# # Tambahkan prefiks "This is" pada deskripsi teks dan ekstraksi fitur teks
# text_features = image_embedding_function.embed_documents(
#     ["This is " + desc for desc in texts]
# )

# # Konversi daftar menjadi array numpy untuk operasi matriks
# img_features_np = np.array(img_features)
# text_features_np = np.array(text_features)

# # Perhitungan kesamaan
# # Menghitung kesamaan kosinus antara fitur teks dan gambar
# similarity = np.matmul(text_features_np, img_features_np.T)

# # Membuat plot untuk memvisualisasikan matriks kesamaan
# count = len(descriptions)
# plt.figure(figsize=(20, 14))

# # Menampilkan matriks kesamaan sebagai heatmap
# plt.imshow(similarity, vmin=0.1, vmax=0.3, cmap="coolwarm")
# plt.colorbar()  # Menambahkan color bar

# # Menampilkan deskripsi teks pada sumbu y
# plt.yticks(range(count), texts, fontsize=18)
# plt.xticks([])  # Menghapus tanda pada sumbu x

# # Menampilkan gambar asli di bawah sumbu x
# for i, image in enumerate(original_images):
#     plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

# # Menampilkan nilai kesamaan di atas heatmap sebagai teks
# for x in range(similarity.shape[1]):
#     for y in range(similarity.shape[0]):
#         plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

# # Menghapus batas plot
# for side in ["left", "top", "right", "bottom"]:
#     plt.gca().spines[side].set_visible(False)

# # Mengatur rentang plot
# plt.xlim([-0.5, count - 0.5])
# plt.ylim([count + 0.5, -2])

# # Menambahkan judul
# plt.title("Kesamaan Kosinus Antara Fitur Teks dan Gambar", size=20)
# plt.show()

print("\n===VECTORSTORE===\n")

# Membuat DB
image_db = Chroma(
    collection_name="multimodal",
    embedding_function=image_embedding_function,
)

# Menambahkan gambar
image_db.add_images(uris=image_uris)

class ImageRetriever:
    def __init__(self, retriever):
        """
        Menginisialisasi pengambil gambar.

        Argumen:
        retriever: Objek retriever dari LangChain
        """
        self.retriever = retriever

    def invoke(self, query):
        """
        Mengambil dan menampilkan gambar berdasarkan kueri.

        Argumen:
        query (str): Kueri pencarian
        """
        docs = self.retriever.invoke(query)
        if docs and isinstance(docs[0], Document):
            self.plt_img_base64(docs[0].page_content)
        else:
            print("Tidak ada gambar yang ditemukan.")
        return docs

    @staticmethod
    def resize_base64_image(base64_string, size=(224, 224)):
        """
        Mengubah ukuran gambar yang dienkode dalam bentuk string Base64.

        Argumen:
        base64_string (str): String Base64 dari gambar asli.
        size (tuple): Ukuran gambar yang diinginkan dalam bentuk (lebar, tinggi).

        Return:
        str: String Base64 dari gambar yang ukurannya sudah diubah.
        """
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        resized_img = img.resize(size, Image.LANCZOS)
        buffered = io.BytesIO()
        resized_img.save(buffered, format=img.format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def plt_img_base64(img_base64):
        """
        Menampilkan gambar yang dienkode dalam bentuk Base64.

        Argumen:
        img_base64 (str): String gambar yang dienkode dalam bentuk Base64
        """
        # image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
        # display(HTML(image_html))
        print(f"base64: {img_base64}")

# Buat Retriever Gambar
retriever = image_db.as_retriever(search_kwargs={"k": 3})
image_retriever = ImageRetriever(retriever)

# image_retriever.invoke("A Dog on the street")