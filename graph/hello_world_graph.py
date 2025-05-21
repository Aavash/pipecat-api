from typing import Dict, TypedDict
from langgraph.graph import StateGraph

# Stategraph helps design and manage the flow of task in the application using a graph


# We create an AgentState - shared data structure that keeps track of information as application runs


class AgentState(TypedDict):
    name: str


# Input and Output has to be state in a node typically
""" Docs string in langgraph is very important because they tell your AI agents what the function does"""


def greeting_node(state: AgentState) -> AgentState:
    """Simple node that adds a greeting message to the state"""
    state["name"] = state.get("name") + ", you are doing great job learning LangGraph!"
    return state


# Passing in agent state to create graph
graph = StateGraph(AgentState)
graph.add_node("greeter", greeting_node)

graph.set_entry_point("greeter")
graph.set_finish_point("greeter")


app = graph.compile()


result = app.invoke({"name": "Bob"})

print(result["name"])
