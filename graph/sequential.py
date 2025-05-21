from typing import List, TypedDict
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    name: str
    age: str
    skills: List[str]
    final: str


def first_node(state: AgentState) -> AgentState:
    """This is the first node of our sequence"""

    state["final"] = f"Hello {state['name']}"
    return state


def second_node(state: AgentState) -> AgentState:
    """This is the second node of our sequence"""
    state["final"] = f"{state['name']}, You are {state['age']} years old!"
    return state


def third_node(state: AgentState) -> AgentState:
    """This is the third node of our sequence"""

    state["final"] = (
        f"{state['name']}, You are {state['age']} years old!. You have skills in {', '.join(state['skills'][:-1])}{' and ' if len(state['skills']) > 1 else ''}{state['skills'][-1]}"
    )
    return state


graph = StateGraph(AgentState)

graph.add_node("first_node", first_node)
graph.add_node("second_node", second_node)
graph.add_node("third_node", third_node)


graph.set_entry_point("first_node")
graph.add_edge("first_node", "second_node")
graph.add_edge("second_node", "third_node")


graph.set_finish_point("third_node")

app = graph.compile()

result = app.invoke(
    {"name": "Aavash", "age": "30", "skills": ["langgraph", "python", "docker"]}
)

print(result)
