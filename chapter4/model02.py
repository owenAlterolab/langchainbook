from dotenv import load_dotenv
from langchain_altero import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
import os

load_dotenv()
logging.langsmith("CH04-Models")

# model
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# prompt template
prompt = PromptTemplate.from_template("Berikan saya ringkasan {negara} dalam 200 karakter atau lebih")

chain = prompt | llm

# Membuat direktori cache.
# if not os.path.exists("cache"):
#     os.makedirs("cache")

# Gunakan SQLiteCache.
# set_llm_cache(SQLiteCache(database_path="cache/llm_cache.db"))

#inMemoryCache
# set_llm_cache(InMemoryCache())

response = chain.invoke({"negara": "indonesia"})
print(response.content)