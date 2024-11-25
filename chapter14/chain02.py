# File konfigurasi untuk mengelola API KEY
from dotenv import load_dotenv

# Memuat informasi API KEY
load_dotenv()

# Mengatur pelacakan LangSmith. https://smith.langchain.com
# !pip install langchain-altero
from langchain_altero import logging

# Masukkan nama proyek.
logging.langsmith("Structured-Output-Chain")

from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase

# Hubungkan ke database SQLite.
db = SQLDatabase.from_uri("sqlite:///chapter14/data/Chinook.db")

# Mencetak dialek basis data.
print(db.dialect)

# Mencetak nama tabel yang tersedia.
print(db.get_usable_table_names())

# model menentukan gpt-3.5-turbo
llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0)

# Membuat rantai dengan LLM dan DB sebagai parameter.
chain = create_sql_query_chain(llm, db)

# # Menjalankan rantai dan mencetak hasilnya.
# generated_sql_query = chain.invoke({"question": "Sebutkan nama pelanggan"})

# # Mencetak kueri yang dihasilkan.
# print(generated_sql_query.__repr__())

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# Buat alat untuk menjalankan kueri yang Anda buat.
execute_query = QuerySQLDataBaseTool(db=db)

# query_res = execute_query.invoke({"query": generated_sql_query})
# print(query_res)

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# Alat
execute_query = QuerySQLDataBaseTool(db=db)

# Rantai pembuatan kueri SQL
write_query = create_sql_query_chain(llm, db)

# Membuat rantai untuk menjalankan kueri yang dihasilkan.
chain = write_query | execute_query

# query_res = chain.invoke({"question": "Cari email dari Daan Peeters"})
# print(query_res)

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# model menentukan gpt-3.5-turbo
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Hubungkan ke database SQLite.
db = SQLDatabase.from_uri("sqlite:///chapter14/data/Chinook.db") # sesuaikan path ke database

# Buat Agen
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# Memeriksa hasil eksekusi
agent_executor.invoke(
    {"input": "Sebutkan track dengan genre science fiction"}
)