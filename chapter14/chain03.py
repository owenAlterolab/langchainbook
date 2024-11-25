# File konfigurasi untuk mengelola API KEY
from dotenv import load_dotenv

# Memuat informasi API KEY
load_dotenv()

# Mengatur pelacakan LangSmith. https://smith.langchain.com
# !pip install langchain-altero
from langchain_altero import logging

# Masukkan nama proyek.
logging.langsmith("Structured-Output-Chain")