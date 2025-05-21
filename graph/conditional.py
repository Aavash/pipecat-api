from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    operation: Literal["+", "-"]
    operation2: Literal["+", "-"]
    number1: int
    number2: int
    number3: int
    number4: int
    finalNumber: int
    finalNumber2: int


def adder(state: AgentState) -> AgentState:
    """This node adds the 2 numbers"""
    print("lauda lassan")
    state["finalNumber"] = state["number1"] + state["number2"]
    return state


def adder2(state: AgentState) -> AgentState:
    """This node adds the 2 numbers"""
    state["finalNumber2"] = state["number3"] + state["number4"]
    return state


def subtractor(state: AgentState) -> AgentState:
    """This node subtracts the 2 numbers"""
    state["finalNumber"] = state["number1"] - state["number2"]
    return state


def subtractor2(state: AgentState) -> AgentState:
    """This node subtracts the 2 numbers"""
    state["finalNumber2"] = state["number3"] - state["number4"]
    return state


def decide_next_node(state: AgentState) -> str:  # Changed return type to str
    """This node decides the node based on operation. This return edge"""
    if state["operation"] == "+":
        return "addition_operation"
    else:
        return "subtraction_operation"


def decide_next_node2(state: AgentState) -> str:  # Changed return type to str
    if state["operation2"] == "+":
        return "addition_operation2"
    else:
        return "subtraction_operation2"


graph = StateGraph(AgentState)

graph.add_node("add_node", adder)
graph.add_node("add_node2", adder2)
graph.add_node("subtractor_node", subtractor)
graph.add_node("subtractor_node2", subtractor2)
graph.add_node("router", lambda state: state)  # passthrough function
graph.add_node("router2", lambda state: state)  # passthrough function

graph.add_edge(START, "router")

graph.add_conditional_edges(
    "router",
    decide_next_node,
    {
        "addition_operation": "add_node",
        "subtraction_operation": "subtractor_node",
    },
)
graph.add_edge("add_node", "router2")
graph.add_edge("subtractor_node", "router2")

graph.add_conditional_edges(
    "router2",
    decide_next_node2,
    {
        "addition_operation2": "add_node2",
        "subtraction_operation2": "subtractor_node2",
    },
)
graph.add_edge("add_node2", END)
graph.add_edge("subtractor_node2", END)

app = graph.compile()

result = app.invoke(
    {
        "operation": "-",
        "operation2": "+",
        "number1": 100,
        "number2": 200,
        "number3": 200,
        "number4": 200,
    }
)

print(result)
