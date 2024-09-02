from langchain_community.retrievers import BM25Retriever
from langchain_altero.retrievers import NLTKBM25Retriever

def pretty_print(docs):
    for i, doc in enumerate(docs):
        if "score" in doc.metadata:
            print(f"[{i+1}] {doc.page_content} ({doc.metadata['score']:.4f})")
        else:
            print(f"[{i+1}] {doc.page_content}")

sample_texts = [
    "Asuransi keuangan dirancang sebagai produk keuangan untuk pengelolaan aset jangka panjang dan perlindungan terhadap risiko.",
    "Asuransi tabungan keuangan adalah produk keuangan khusus yang dirancang untuk tujuan tabungan jangka panjang serta menyediakan fungsi penyediaan hasil ternak.",
    "Jangan berbicara omong kosong tentang asuransi keuangan dan lebih baik mulai menabung. Saya tidak mengerti apa yang membuat Anda begitu terburu-buru.",
    "Asuransi keuangan bom besar lebih berfokus pada perlindungan risiko daripada tabungan. Produk ini cocok untuk pelanggan yang bersedia mengambil risiko tinggi.",
]

nltk = NLTKBM25Retriever.from_texts(sample_texts)
pretty_print(nltk.invoke("Asuransi Keuangan"))