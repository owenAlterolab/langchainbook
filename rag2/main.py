from dotenv import load_dotenv

from typing import Any, Annotated, Literal
from typing_extensions import TypedDict
from pprint import pprint

from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages

load_dotenv()

# Define the state for the agent
class State(TypedDict):
    question: str
    messages: Annotated[list[AnyMessage], add_messages]

workflow = StateGraph(State)


### Initial Router
def initial_router(state: State):

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            You are a router, your task is make a decision between possible action paths based on the human message:
            "GENERIC" Take this path if the human message is a greeting, or a farewell, or stuff related.
            "CUSTOMER" Take this path if the humans asked for a help.
            "QUERY" Take this path if the question asked to check the database.
            "OTHER" Take this path if the question asked about other things.

            Rule 1 : You should never infer information if it does not appear in the context of the query
            Rule 2 : You can only answer with the type of query that you choose based on why you choose it.

            Answer only with the type of query that you choose, just one word.
        """),
        ("human", "{question}"),
    ])

    llm = ChatOllama(model="llama3.1:8b")

    chain = prompt | llm

    return {"messages": [chain.invoke(state)]}

workflow.add_node("initial_router", initial_router)

workflow.add_edge(START, "initial_router")
workflow.add_edge("initial_router", END)

graph = workflow.compile()

# for events in graph.stream({"question": "Hello, good morning!"}):
for events in graph.stream({"question": "hello, i have a problem, can you help me?"}):
    print(events)