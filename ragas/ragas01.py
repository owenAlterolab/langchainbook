from dotenv import load_dotenv
load_dotenv()

from langchain_altero import logging
logging.langsmith("CH15-Evaluations")

from langchain_community.document_loaders import PDFPlumberLoader

# TODO: SESUAIKAN SNIPPET DENGAN GITBOOK

# Membuat pemuat dokumen
loader = PDFPlumberLoader("ragas/data/ChatGPT:Keuntungan,Risiko,DanPenggunaanBijakDalamEraKecerdasanBuatan.pdf")

# Memuat dokumen
docs = loader.load()

# daftar isi, tidak termasuk halaman akhir
docs = docs[:-3]

# jumlah halaman dalam dokumen
print(len(docs))

# atur metadata (filename harus ada)
for doc in docs:
    doc.metadata["filename"] = doc.metadata["source"]

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.extractor import KeyphraseExtractor
from ragas.testset.docstore import InMemoryDocumentStore

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# generator dataset
generator_llm = ChatOpenAI(model="gpt-4o-mini")
# Pengkritik dataset
critic_llm = ChatOpenAI(model="gpt-4o-mini")
# Penyematan dokumen
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Menyiapkan pemisah teks.
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Bungkus model ChatOpenAI LangChain dengan LangchainLLMWrapper agar kompatibel dengan Ragas.
langchain_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

# Inisialisasi ekstraktor sintaksis utama. Gunakan LLM yang didefinisikan di atas.
keyphrase_extractor = KeyphraseExtractor(llm=langchain_llm)

# membuat ragas_embeddings
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# Inisialisasi InMemoryDocumentStore.
# Ini adalah tempat penyimpanan yang menyimpan dan mengelola dokumen dalam memori.
docstore = InMemoryDocumentStore(
    splitter=splitter,
    embeddings=ragas_embeddings,
    extractor=keyphrase_extractor,
)

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    ragas_embeddings,
    docstore=docstore,
)

# Tentukan distribusi berdasarkan jenis pertanyaan.
# simple: pertanyaan sederhana, reasoning: pertanyaan yang memerlukan penalaran, multi_context: pertanyaan yang harus mempertimbangkan berbagai konteks, conditional: pertanyaan kondisional
distributions = {simple: 0.4, reasoning: 0.2, multi_context: 0.2, conditional: 0.2}

# membuat set tes
# docs: data dokumentasi, 10: jumlah pertanyaan yang akan dibuat, distributions: distribusi berdasarkan jenis pertanyaan, with_debugging_logs: apakah akan menampilkan log debugging
testset = generator.generate_with_langchain_docs(
    documents=docs, test_size=10, distributions=distributions, with_debugging_logs=True
)

# Mengubah test set yang dihasilkan menjadi pandas DataFrame
test_df = testset.to_pandas()
print("test_df")
print(test_df)

# mencetak 5 baris teratas dari DataFrame
print("head")
print(test_df.head())

# Menyimpan DataFrame sebagai file CSV
test_df.to_csv("ragas/data/ragas_synthetic_dataset.csv", index=False)