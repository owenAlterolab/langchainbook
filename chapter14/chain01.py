# File konfigurasi untuk mengelola API KEY sebagai variabel
from dotenv import load_dotenv

# # Memuat informasi API KEY
load_dotenv()

# LangSmith. https://smith.langchain.com
from langchain_altero import logging

# Masukkan nama proyek.
logging.langsmith("Summary")

from langchain_community.document_loaders import TextLoader

# Memuat data berita
loader = TextLoader("./chapter14/data/finance-keywords.txt")
docs = loader.load()
# print(f"Total karakter: {len(docs[0].page_content)}")
# print("\n========= Preview the front matter =========\n")
# print(docs[0].page_content[:500])

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """
            Please summarize the sentence according to the following REQUEST.
            REQUEST:
            1. Summarize the main points in bullet points in BAHASA INDONESIA.
            2. Each summarized sentence must start with an emoji that fits the meaning of the each sentence.
            3. Use various emojis to make the summary more interesting.
            4. Translate the summary into BAHASA INDONESIA if it is not written in Bahasa Indonesia.
            5. DO NOT translate any technical terms.
            6. DO NOT include any unnecessary information.
            
            CONTEXT:
            {context}
            
            SUMMARY:
        """
)

from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_teddynote.callbacks import StreamingCallback


llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    streaming=True,
    temperature=0,
    callbacks=[StreamingCallback()],
)


# stuff_chain = create_stuff_documents_chain(llm, prompt)
# answer = stuff_chain.invoke({"context": docs})

# print(answer)

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./chapter14/data/ChatGPT:Keuntungan,Risiko,DanPenggunaanBijakDalamEraKecerdasanBuatan.pdf")
docs = loader.load()
docs = docs[3:8]  # Rangkuman bagian dari dokumentasi di sini
# print(f"Jumlah total halaman: {len(docs)}")

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o-mini",
)

# download map prompt
map_prompt = hub.pull("teddynote/map-prompt")

# print prompt
# map_prompt.pretty_print()

# membuat map_chain
map_chain = map_prompt | llm | StrOutputParser()

# Extrak highlights untuk dokumen
# doc_summaries = map_chain.batch(docs)

# Menampilkan jumlah dokumen yang diringkas
# print(len(doc_summaries))

# Keluarkan ringkasan dari beberapa dokumen
# print(doc_summaries[0])

# download reduce prompt
reduce_prompt = hub.pull("teddynote/reduce-prompt")

# cetak prompt
# reduce_prompt.pretty_print()

# buat reduce chain
reduce_chain = reduce_prompt | llm | StrOutputParser()

from langchain_altero.messages import stream_response

# answer = reduce_chain.stream({"doc_summaries": doc_summaries, "language": "Bahasa Indonesia"})
# stream_response(answer)

from langchain_core.runnables import chain

@chain
def map_reduce_chain(docs):
    map_llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini",
    )

    # Mengunduh prompt peta
    map_prompt = hub.pull("teddynote/map-prompt")

    # Membuat rantai peta
    map_chain = map_prompt | map_llm | StrOutputParser()

    # Menghubungkan prompt pertama, ChatOpenAI, dan parser output string untuk membuat rantai.
    doc_summaries = map_chain.batch(docs)

    # Mengunduh prompt pengurangan
    reduce_prompt = hub.pull("teddynote/reduce-prompt")
    reduce_llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        callbacks=[StreamingCallback()],
        streaming=True,
    )

    reduce_chain = reduce_prompt | reduce_llm | StrOutputParser()

    return reduce_chain.invoke({"doc_summaries": doc_summaries, "language": "Bahasa Indonesia"})

# answer = map_reduce_chain.invoke(docs)



from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# create map llm
map_llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o-mini",
)
# membuat rantai peta
map_summary = hub.pull("teddynote/map-summary-prompt")

# print prompt
# map_summary.pretty_print()

# membuat map chain
map_chain = map_summary | llm | StrOutputParser()

# mencetak ringkasan dari dokumen pertama
# print("doc[0]")
# print(map_chain.invoke({"documents": docs[0], "language": "Bahasa Indonesia"}))

# Tentukan semua dokumen sebagai masukan.
input_doc = [{"documents": doc, "language": "Bahasa Indonesia"} for doc in docs]

# Mencetak ringkasan semua dokumen.
# print("batch")
# print(map_chain.batch(input_doc))

# download refine prompt
refine_prompt = hub.pull("teddynote/refine-prompt")

# mencetak prompt
# refine_prompt.pretty_print()

# buat refine llm
refine_llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o-mini",
)

# buat refine chain
refine_chain = refine_prompt | refine_llm | StrOutputParser()

from langchain_core.runnables import chain

@chain
def map_refine_chain(docs):

    # Membuat rantai peta
    map_summary = hub.pull("teddynote/map-summary-prompt")

    map_chain = (
        map_summary
        | ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
        )
        | StrOutputParser()
    )

    input_doc = [{"documents": doc.page_content, "language": "Korean"} for doc in docs]

    # Menghubungkan prompt pertama, ChatOpenAI, dan parser output string untuk membuat rantai.
    doc_summaries = map_chain.batch(input_doc)

    refine_prompt = hub.pull("teddynote/refine-prompt")

    refine_llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        callbacks=[StreamingCallback()],
        streaming=True,
    )

    refine_chain = refine_prompt | refine_llm | StrOutputParser()

    previous_summary = doc_summaries[0]

    for current_summary in doc_summaries[1:]:

        previous_summary = refine_chain.invoke(
            {
                "previous_summary": previous_summary,
                "current_summary": current_summary,
                "language": "Bahasa Indonesia",
            }
        )
        print("\n\n-----------------\n\n")

    return previous_summary

# refined_summary = map_refine_chain.invoke(docs)


import textwrap
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import SimpleJsonOutputParser

# Menentukan nilai default untuk semua input kecuali {content}
cod_chain_inputs = {
    "content": lambda d: d.get("content"),
    "content_category": lambda d: d.get("content_category", "Artikel"),
    "entity_range": lambda d: d.get("entity_range", "1-3"),
    "max_words": lambda d: int(d.get("max_words", 80)),
    "iterations": lambda d: int(d.get("iterations", 5)),
}

# Mengunduh prompt Chain of Density
cod_prompt = hub.pull("teddynote/chain-of-density-prompt")

# Membuat rantai Chain of Density
cod_chain = (
    cod_chain_inputs
    | cod_prompt
    | ChatOpenAI(temperature=0, model="gpt-4o-mini")
    | SimpleJsonOutputParser()
)

# Membuat rantai kedua, hanya mengekstrak ringkasan akhir (tidak bisa streaming, perlu hasil akhir)
cod_final_summary_chain = cod_chain | (
    lambda output: output[-1].get(
        "denser_summary", 'Kesalahan: Kunci "denser_summary" tidak ada di kamus terakhir'
    )
)

content = docs[1].page_content
print(content)