from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage
from langchain_community.utilities.sql_database import SQLDatabase
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition
from langchain_core.tools import tool

# import sql db
db = SQLDatabase.from_uri("sqlite:///sf-food-inspections-lives.sqlite")

# Build Model
model = ChatOpenAI(temperature=0, streaming=True, model="gpt-5-nano")

# Define system prompt
sys_msg = SystemMessage(content="You are a conversational and friendly sql database assistant." \
"When a customer asks a question you must make a call to the db in order to retrieve the necessary info." \
f"There is one usable table: {db.get_usable_table_names()} - table info: {db.get_table_info(['inspection_records'])}")

# create tool
@tool
def make_db_call(query:str):
    """Look up info from the db to answer the customer's question.

    Args:
        query: str"""
    
    try:
        info = db.run(f"{query}")

        return {"info": info}
    except Exception as e:
        
        return {"error_message": e}
    
# bind tool to model
sql_model = model.bind_tools([make_db_call])

# define state
class State(MessagesState):
    pass

# create nodes
def model_node(state:State):

    return {"messages": [sql_model.invoke([sys_msg] + state["messages"])]}

def tool_node(state:State):

    # last message contains param tool_calls, a list of dicts
    tool_call = state["messages"][-1].tool_calls[0]

    # make tool call
    observation = make_db_call.invoke(tool_call["args"])

    tool_message = {"role": "tool", "content" : observation, "tool_call_id": tool_call["id"]}

    return {'messages': tool_message}

# Create graph
builder = StateGraph(State)

# Add nodes
builder.add_node("model_node", model_node)
builder.add_node("tools", tool_node)

# Add edges
builder.add_edge(START, "model_node")
builder.add_conditional_edges("model_node", tools_condition)
builder.add_edge("tools", "model_node")

# Compile and view
graph = builder.compile(interrupt_before=["tools"])