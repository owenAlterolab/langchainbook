# Berkas konfigurasi untuk mengelola API KEY sebagai variabel lingkungan
from dotenv import load_dotenv
# Mengatur pelacakan LangSmith. https://smith.langchain.com
from langchain_altero import logging
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Memuat informasi API KEY
load_dotenv()
# Masukkan nama proyek
logging.langsmith("CH10-Retriever")

# Helper function to print documents nicely
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

    # Menggunakan TextLoader untuk memuat dokumen dari file "appendix-keywords.txt".
loader = TextLoader("/Users/mini16gboffice1/StudioProjects/owen/python/langchainbook/appendix-keywords.txt")

# Membagi dokumen menjadi bagian-bagian dengan ukuran chunk 300 karakter dan tidak ada overlapping.
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
texts = loader.load_and_split(text_splitter)

# Menggunakan OpenAIEmbeddings untuk membuat basis data vektor FAISS dan mengkonversi menjadi search engine.
retriever = FAISS.from_documents(texts, OpenAIEmbeddings()).as_retriever()

# Definisi query dan mencari dokumen terkait.
docs = retriever.get_relevant_documents("Apa itu Semantic Search?")

# Mencetak hasil pencarian dengan cara yang lebih rapih.
pretty_print_docs(docs)

print()
print("Contextual Compression")
print()


from langchain_teddynote.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

# from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")  # Inisialisasi model bahasa OpenAI

# Membuat kompresor dokumen menggunakan LLM
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    # Membuat retriever kompresi kontekstual menggunakan kompresor dokumen dan retriever
    base_compressor=compressor,
    base_retriever=retriever,
)

pretty_print_docs(retriever.invoke("Jelaskan tentang Semantic Search."))

print("=========================================================")
print("============== Setelah penerapan LLMChainExtractor ==================")

compressed_docs = (
    compression_retriever.invoke(  # Mencari dokumen yang relevan menggunakan retriever kompresi kontekstual
        "Jelaskan tentang Semantic Search."
    )
)
pretty_print_docs(compressed_docs)  # Menampilkan dokumen yang ditemukan secara terformat


print()
print("Chain filter")
print()

from langchain_teddynote.document_compressors import LLMChainFilter


# Membuat objek LLMChainFilter menggunakan LLM
_filter = LLMChainFilter.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    # Membuat objek ContextualCompressionRetriever menggunakan LLMChainFilter dan retriever
    base_compressor=_filter,
    base_retriever=retriever,
)

compressed_docs = compression_retriever.invoke(
    # Pertanyaan atau kueri
    "Jelaskan tentang Semantic Search."
)
pretty_print_docs(compressed_docs)  # Menampilkan dokumen yang terkompresi secara terformat

print("\nEmbeddings Filter\n")

from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Membuat objek EmbeddingsFilter dengan ambang batas kemiripan 0.76.
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings, similarity_threshold=0.86)

# Membuat objek ContextualCompressionRetriever dengan menggunakan embeddings_filter sebagai kompresor dasar, dan retriever sebagai pencari dasar.
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter, base_retriever=retriever
)

# Menggunakan objek ContextualCompressionRetriever untuk mencari dokumen terkait.
compressed_docs = compression_retriever.get_relevant_documents(
    # Kuiri
    "Ceritakan tentang Semantic Search."
)
# Mencetak dokumen yang telah dikompres dengan tampilan yang indah.
pretty_print_docs(compressed_docs)


print("\nPipeline\n")

from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import CharacterTextSplitter

# Membuat pemisah teks berbasis karakter dengan ukuran chunk 300 dan overlap antar chunk 0.
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)

# Membuat filter redundansi menggunakan embeddings.
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

# Membuat filter relevansi menggunakan embeddings dan menetapkan ambang batas kemiripan 0.86.
relevant_filter = EmbeddingsFilter(
    embeddings=embeddings, similarity_threshold=0.86)

pipeline_compressor = DocumentCompressorPipeline(
    # Buat pipeline kompresi dokumen, atur pembagi, filter duplikat, filter relevansi, dan LLMChainExtractor sebagai konverter.
    transformers=[
        splitter,
        redundant_filter,
        relevant_filter,
        LLMChainExtractor.from_llm(llm),
    ]
)

compression_retriever = ContextualCompressionRetriever(
    # Menginisialisasi ContextualCompressionRetriever dengan menggunakan pipeline_compressor sebagai kompresor dasar, dan retriever sebagai pencari dasar.
    base_compressor=pipeline_compressor,
    base_retriever=retriever,
)

compressed_docs = compression_retriever.get_relevant_documents(
    # query / pertanyaan
    "Ceritakan tentang Semantic Search."
)
# Mencetak dokumen yang telah dikompres dengan tampilan yang indah.
pretty_print_docs(compressed_docs)