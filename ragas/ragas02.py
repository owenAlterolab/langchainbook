from dotenv import load_dotenv
load_dotenv()

from langchain_altero import logging
logging.langsmith("CH15-Evaluations")

import pandas as pd

df = pd.read_csv("ragas/data/ragas_synthetic_dataset.csv") #sesuaikan path
print("head")
print(df.head())

from datasets import Dataset
import ast

test_dataset = Dataset.from_pandas(df)
print(test_dataset)

def convert_to_list(example):
    contexts = ast.literal_eval(example["contexts"])
    return {"contexts": contexts}


test_dataset2 = test_dataset.map(convert_to_list)
print("convert to list dataset")
print(test_dataset2)

# print(test_dataset[1]["contexts"])
print(test_dataset2[1]["contexts"])

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Tahap 1: Memuat Dokumen (Load Documents)
loader = PyMuPDFLoader("ragas/data/ChatGPT:Keuntungan,Risiko,DanPenggunaanBijakDalamEraKecerdasanBuatan.pdf")
docs = loader.load()

# Tahap 2: Membagi Dokumen (Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)

# Tahap 3: Membuat Embedding (Embedding)
embeddings = OpenAIEmbeddings()

# Tahap 4: Membuat DB dan Menyimpan (Create DB) serta Menyimpan
# Membuat vectorstore.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# Tahap 5: Membuat Retriever (Retriever)
# Mencari dan menghasilkan informasi yang terdapat dalam dokumen.
retriever = vectorstore.as_retriever()

# Tahap 6: Membuat Prompt (Create Prompt)
# Membuat prompt.
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 

#Context: 
{context}

#Question:
{question}

#Answer:"""
)

# Langkah 7: Membuat model bahasa (LLM)
# Buat model bahasa (LLM).
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Langkah 8: Membuat sebuah rantai
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

batch_dataset = [question for question in test_dataset2["question"]]
print(batch_dataset[:3])

answer = chain.batch(batch_dataset)
print(answer[:3])

# Menimpa atau menambahkan kolom 'jawaban'
if "answer" in test_dataset2.column_names:
    test_dataset2 = test_dataset2.remove_columns(["answer"]).add_column("answer", answer)
else:
    test_dataset2 = test_dataset2.add_column("answer", answer)

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

result = evaluate(
    dataset=test_dataset2,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)

print(result)

result_df = result.to_pandas()
print(result_df.head())

ress = result_df.loc[:, "context_precision":"context_recall"]
print(ress)