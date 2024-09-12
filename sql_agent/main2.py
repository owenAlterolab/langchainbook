from dotenv import load_dotenv
load_dotenv()

from typing import Any, Annotated, Literal
from typing_extensions import TypedDict

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages

# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Setup DB
db = SQLDatabase.from_uri("sqlite:///sql_agent/Chinook.db")
llm = ChatOllama(model="llama3.1:8b")
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

# Define a new graph
workflow = StateGraph(State)

def chatbot(state: State):
    template = "Hitung jumlah kata yang diberikan"

    system="""
        You are a SQLite query generator. Your task is to create SQLite queries based on the user’s question. 

        1. Always output only the SQLite query.
        2. If you do not have enough information to generate the query, respond with: "I don't have enough data to query [user question]".
        3. Use `LIKE` instead of `=` for string comparisons in the SQL queries.
        4. The SQL query should use `%` as wildcards for pattern matching in `LIKE` clauses.

        For example:
        If the user's question is: "Find employees with the title 'IT'",
        - your output should be: SELECT * FROM employee WHERE title LIKE '%IT%'
        - If you cannot generate a query due to insufficient data, your response should be: I don't have enough data to query Find employees with the title 'IT'

        Now, please generate the SQL query based on the user's question.
    """

    schema_extras="""
        Relationship: Album.ArtistId = Artist.ArtistId
        Description: This foreign key relationship links each album to its corresponding artist.

        Relationship: Customer.SupportRepId = Employee.EmployeeId
        Description: This foreign key relationship associates each customer with their support representative (employee).

        Relationship: Invoice.CustomerId = Customer.CustomerId
        Description: This foreign key relationship connects each invoice to its corresponding customer.

        Relationship: InvoiceLine.InvoiceId = Invoice.InvoiceId
        Description: This foreign key relationship links each invoice line to its corresponding invoice.

        Relationship: InvoiceLine.TrackId = Track.TrackId
        Description: This foreign key relationship associates each invoice line with the specific track it relates to.

        Relationship: Track.AlbumId = Album.AlbumId
        Description: This foreign key relationship links each track to its corresponding album.

        Relationship: Track.MediaTypeId = MediaType.MediaTypeId
        Description: This foreign key relationship associates each track with its media type.

        Relationship: Track.GenreId = Genre.GenreId
        Description: This foreign key relationship connects each track to its genre.

        Relationship: PlaylistTrack.TrackId = Track.TrackId
        Description: This foreign key relationship links each track in a playlist to its corresponding track.

        Relationship: PlaylistTrack.PlaylistId = Playlist.PlaylistId
        Description: This foreign key relationship connects each track in a playlist to its corresponding playlist.
    """

    # prompt = PromptTemplate.from_template(template)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("placeholder", "{messages}"),
        ("ai", schema_extras),

        ("human", "List all artists."),
        ("ai", "SELECT * FROM Artist;"),

        ("human", "Find all albums for the artist 'AC/DC'."),
        ("ai", "SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name LIKE '%AC/DC%');"),

        ("human", "List all tracks in the 'Rock' genre."),
        ("ai", "SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name LIKE '%Rock%');"),

        ("human", "Find the total duration of all tracks."),
        ("ai", "SELECT SUM(Milliseconds) FROM Track;"),

        ("human", "List all customers from Canada."),
        ("ai", "SELECT * FROM Customer WHERE Country LIKE '%Canada%';"),

        ("human", "How many tracks are there in the album with ID 5?"),
        ("ai", "SELECT COUNT(*) FROM Track WHERE AlbumId = 5;"),

        ("human", "Find the total number of invoices."),
        ("ai", "SELECT COUNT(*) FROM Invoice;"),

        ("human", "List all tracks that are longer than 5 minutes."),
        ("ai", "SELECT * FROM Track WHERE Milliseconds > 300000;"),

        ("human", "Who are the top 5 customers by total purchase?"),
        ("ai", "SELECT CustomerId, SUM(Total) AS TotalPurchase FROM Invoice GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;"),

        ("human", "Which albums are from the year 2000?"),
        ("ai", "SELECT * FROM Album WHERE strftime('%Y', ReleaseDate) = '2000';"),

        ("human", "How many employees are there?"),
        ("ai", "SELECT COUNT(*) FROM Employee;"),
        ("human", state['messages'][0].content)
    ])

    chain = prompt | llm.bind_tools(
        [db_query_tool], tool_choice="required"
    )

    return {"messages": [chain.invoke(state)]}
workflow.add_node("chatbot", chatbot)

# Add a node for the first tool call
def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }

# call list table
workflow.add_node("first_tool_call", first_tool_call)

# Add nodes for the first two tools
workflow.add_node(
    "list_tables_tool", ToolNode([list_tables_tool])
)

# Add a node for a model to choose the relevant tables based on the question and available tables
model_get_schema = llm.bind_tools(
    [get_schema_tool]
)
# workflow.add_node(
#     "model_get_schema",
#     lambda state: {
#         "messages": [model_get_schema.invoke(state["messages"])],
#     },
# )

class GetSchemasTool:
    """A node that runs the tools requested in the last AIMessage."""

    # def __init__(self) -> None:
    #     self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
            content = message.content
        else:
            raise ValueError("No message found in input")
        outputs = []
        tables = [item.strip() for item in content.split(',')]
        for table in tables:
            outputs.append(
                ToolMessage(
                    content=get_schema_tool.invoke(table),
                    name="get_schema_tool",
                    tool_call_id="123",
                )
            )
        return {"messages": outputs}


tool_node = GetSchemasTool()
workflow.add_node("get_schemas_tool", tool_node)

@tool
def db_query_tool(query: str) -> str:
    """
    Execute a SQL query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    result = db.run_no_throw(query)
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."
    return result
# Add node for executing the query
workflow.add_node("execute_query", ToolNode([db_query_tool]))

workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "get_schemas_tool")
workflow.add_edge("get_schemas_tool", "chatbot")
workflow.add_edge("chatbot", "execute_query")
workflow.add_edge("execute_query", END)

graph = workflow.compile()

for events in graph.stream({"messages": [("user", "Which sales agent made the most in sales in 2009?")]}):
        for value in events.values():
            last_msg = value["messages"][-1]

            print(f"{type(last_msg).__name__}: {last_msg.content}")

# for event in graph.stream({"messages": [("user", "Which sales agent made the most in sales in 2009?")]}):
#     print(event)
