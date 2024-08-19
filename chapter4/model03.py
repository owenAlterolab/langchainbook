from dotenv import load_dotenv
from langchain_altero import logging
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.load import dumpd, dumps, load
import pickle
import json


load_dotenv()
logging.langsmith("CH04-Models")

# Buat sebuah pertanyaan dengan menggunakan templat prompt.
prompt = PromptTemplate.from_template("Apa warna dari {buah}?")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Periksa apakah serialisasi dapat dilakukan.
print(f"ChatOpenAI: {llm.is_lc_serializable()}")

# 체인을 생성합니다.
chain = prompt | llm

# 직렬화가 가능한지 체크합니다.
chain.is_lc_serializable()

dumpd_chain = dumpd(chain)
print(dumpd_chain)
print(type(dumpd_chain))

dumps_chain = dumps(chain)
print(dumps_chain)
print(type(dumps_chain))

# fuit_chain.pkl 파일로 직렬화된 체인을 저장합니다.
with open("fruit_chain.pkl", "wb") as f:
    pickle.dump(dumpd_chain, f)

# save as json
with open("fruit_chain.json", "w") as fp:
    json.dump(dumpd_chain, fp)

with open("fruit_chain.pkl", "rb") as f:
    loaded_chain = pickle.load(f)

    # Memuat rantai.
    chain_from_file = load(loaded_chain)

    # Jalankan rantai.
    print(chain_from_file.invoke({"buah": "apel"}))

    
load_chain = load(
    loaded_chain, secrets_map={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}
)

print(load_chain.invoke({"buah": "semangka"}))


with open("fruit_chain.json", "r") as fp:
    loaded_from_json_chain = json.load(fp)
    loads_chain = load(loaded_from_json_chain)

    print(loads_chain.invoke({"buah": "jeruk"}))