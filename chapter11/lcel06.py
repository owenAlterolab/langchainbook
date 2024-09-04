from dotenv import load_dotenv
from langchain_altero import logging
from langchain.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langchain.runnables.hub import HubRunnable
from langchain_anthropic import ChatAnthropic

# Load API key information
load_dotenv()
logging.langsmith("LCEL-Advanced")

model = ChatOpenAI(temperature=0, model_name="gpt-4o")

# inv = model.invoke("Apakah ibu kota Indonesia?").__dict__
# print(inv)

# inv = model.invoke(
#     "Apa nama ibu kota Indonesia?",
#     # Mengatur gpt_version menjadi gpt-3.5-turbo.
#     config={"configurable": {"gpt_version": "gpt-3.5-turbo"}},
# ).__dict__

# print(inv)

# inv = model.with_config(configurable={"gpt_version": "gpt-4o-mini"}).invoke(
#     "Apa ibu kota Indonesia?"
# ).__dict__
# print(inv)

# # Membuat template prompt dari template yang ada.
prompt = PromptTemplate.from_template("Pilih angka acak di atas yang lebih besar dari {x}.")
chain = (
    prompt | model
)  # Menghubungkan prompt dan model untuk membuat rantai. Output dari prompt diteruskan sebagai input untuk model.

# #Panggil rantai dan berikan 0 ke variabel input “x”.
# inv = chain.invoke({"x": 0}).__dict__
# print(inv)

# # Anda dapat menentukan pengaturan saat memanggil rantai.
# inv = chain.with_config(configurable={"gpt_version": "gpt-4o"}).invoke({"x": 0}).__dict__
# print(inv)

prompt = HubRunnable("rlm/rag-prompt").configurable_fields(
    # ConfigurableField untuk menetapkan commit repositori pemilik
    owner_repo_commit=ConfigurableField(
        # ID dari field
        id="hub_commit",
        # Nama dari field
        name="Hub Commit",
        # Deskripsi tentang field
        description="RAG prompt oleh rlm",
    )
)
# print(prompt)

# msges = prompt.invoke({"question": "Hello", "context": "World"}).messages
# print(msges)

# inv = prompt.with_config(
#     # Set hub_commit ke teddynote/simple-summary-korean.
#     configurable={"hub_commit": "teddynote/simple-summary-korean"}
# ).invoke({"context": "Hello"})
# print(inv)

llm = ChatAnthropic(
    temperature=0, model="claude-3-5-sonnet-20240620"
).configurable_alternatives(
    # Beri id pada field ini.
    # Saat mengonfigurasi objek yang dapat dieksekusi akhir, id ini dapat digunakan untuk mengonfigurasi field ini.
    ConfigurableField(id="llm"),
    # Tetapkan kunci default.
    # Jika kunci ini ditentukan, LLM default (ChatAnthropic) yang diinisialisasi di atas akan digunakan.
    default_key="anthropic",
    # Tambahkan opsi baru bernama 'openai', yang sama dengan `ChatOpenAI()`.
    openai=ChatOpenAI(model="gpt-4o-mini"),
    # Tambahkan opsi baru bernama 'gpt4', yang sama dengan `ChatOpenAI(model="gpt-4")`.
    gpt4o=ChatOpenAI(model="gpt-4o"),
    # Anda dapat menambahkan lebih banyak opsi konfigurasi di sini.
)
prompt = PromptTemplate.from_template("Tolong jelaskan secara singkat tentang {topic}.")
chain = prompt | llm

# inv = chain.invoke({"topic": "Langchain"}).__dict__
# print(inv)

# #ubah pengaturan chain untuk memanggilnya.
# inv = chain.with_config(configurable={"llm": "openai"}).invoke({"topic": "Langchain"}).__dict__
# print(inv)

# # ubah pengaturan chain untuk memanggilnya.
# inv = chain.with_config(configurable={"llm": "gpt4o"}).invoke({"topic": "Langchain"}).__dict__
# print(inv)

# inv = chain.with_config(configurable={"llm": "anthropic"}).invoke(
#     {"topic": "Berita"}
# ).__dict__
# print(inv)

# Inisialisasi model bahasa dan atur temperature ke 0.
llm = ChatOpenAI(temperature=0)

prompt = PromptTemplate.from_template(
    "Apa ibu kota dari {country}?"  # Template prompt dasar
).configurable_alternatives(
    # Beri id pada field ini.
    ConfigurableField(id="prompt"),
    # Tetapkan kunci default.
    default_key="capital",
    # Tambahkan opsi baru bernama 'area'.
    area=PromptTemplate.from_template("Berapa luas {country}?"),
    # Tambahkan opsi baru bernama 'population'.
    population=PromptTemplate.from_template("Berapa jumlah penduduk {country}?"),
    # Tambahkan opsi baru bernama 'eng'.
    eng=PromptTemplate.from_template("Tolong terjemahkan {input} ke dalam bahasa Inggris."),
    # Anda dapat menambahkan lebih banyak opsi konfigurasi di sini.
)

# Hubungkan prompt dan model bahasa untuk membuat rantai.
chain = prompt | llm

# inv = chain.invoke({"country": "Indonesia"})
# print(inv)

# # Panggil with_config untuk mengubah pengaturan rantai.
# inv = chain.with_config(configurable={"prompt": "area"}).invoke({"country": "Indonesia"})
# print(inv)

# # panggil with_config untuk mengubah pengaturan dalam rantai.
# inv = chain.with_config(configurable={"prompt": "population"}).invoke({"country": "Indonesia"})
# print(inv)

# # Panggil with_config untuk mengubah pengaturan rantai.
# inv = chain.with_config(configurable={"prompt": "eng"}).invoke({"input": "Apel itu enak!"})
# print(inv)



llm = ChatAnthropic(
    temperature=0, model="claude-3-5-sonnet-20240620"
).configurable_alternatives(
    # Beri id pada field ini.
    # Saat mengonfigurasi objek yang dapat dieksekusi akhir, id ini dapat digunakan untuk mengonfigurasi field ini.
    ConfigurableField(id="llm"),
    # Tetapkan kunci default.
    # Jika kunci ini ditentukan, LLM default (ChatAnthropic) yang diinisialisasi di atas akan digunakan.
    default_key="anthropic",
    # Tambahkan opsi baru bernama 'openai', yang sama dengan `ChatOpenAI(model="gpt-4o-mini")`.
    openai=ChatOpenAI(model="gpt-4o-mini"),
    # Tambahkan opsi baru bernama 'gpt4', yang sama dengan `ChatOpenAI(model="gpt-4o")`.
    gpt4=ChatOpenAI(model="gpt-4o"),
    # Anda dapat menambahkan lebih banyak opsi konfigurasi di sini.
)

prompt = PromptTemplate.from_template(
    "Jelaskan tentang {company} dalam 20 kata atau kurang."  # Template prompt dasar
).configurable_alternatives(
    # Beri id pada field ini.
    ConfigurableField(id="prompt"),
    # Tetapkan kunci default.
    default_key="description",
    # Tambahkan opsi baru bernama 'founder'.
    founder=PromptTemplate.from_template("Siapa pendiri {company}?"),
    # Tambahkan opsi baru bernama 'competitor'.
    competitor=PromptTemplate.from_template("Siapa pesaing {company}?"),
    # Anda dapat menambahkan lebih banyak opsi konfigurasi di sini.
)
chain = prompt | llm

# # Anda dapat mengonfigurasi dengan menetapkan nilai konfigurasi menggunakan with_config.
# inv = chain.with_config(configurable={"prompt": "founder", "llm": "openai"}).invoke(
#     # Meminta pemrosesan terkait perusahaan yang diberikan oleh pengguna.
#     {"company": "Apple"}
# ).__dict__
# print(inv)

# inv = chain.with_config(configurable={"llm": "anthropic"}).invoke(
#     {"company": "Apple"}
# ).__dict__
# print(inv)

# inv = chain.with_config(configurable={"prompt": "competitor"}).invoke(
#     {"company": "Apple"}
# ).__dict__
# print(inv)

# inv = chain.invoke({"company": "Apple"}).__dict__
# print(inv)

# Ubah pengaturan menjadi with_config untuk menyimpan chain yang Anda buat dalam variabel terpisah.
gpt4_competitor_chain = chain.with_config(
    configurable={"llm": "gpt4", "prompt": "competitor"}
)

inv = gpt4_competitor_chain.invoke({"company": "Apple"}).__dict__
print(inv)