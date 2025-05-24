from typing import Annotated, TypedDict, Sequence, _TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


load_dotenv()


# Reducer Function (add_message in this case helps add message)
# Rule that controls how updates from nodes are combined with the existing state
# Tells us how to merge new data into the current state


# Without a reducer, update would have replaced the existing value entirely
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""
    return a + b


tools = [add]


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0.7).bind_tools(
    tools
)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI asistant, please answer my query to the best of your ability"
    )
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)


tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")


graph.add_conditional_edges(
    "our_agent", should_continue, {"continue": "tools", "end": END}
)

graph.add_edge("tools", "our_agent")


app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [("user", "Multiplye 3 and 4 for me please")]}
print_stream(app.stream(inputs, stream_mode="values"))
