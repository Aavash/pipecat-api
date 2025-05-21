from functools import reduce
import math
from operator import mul
from typing import Literal, TypedDict, List
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    values: List[int]
    name: str
    operation: Literal["*", "+"]
    result: str


def process_values(state: AgentState) -> AgentState:
    """This function handles multiple different inputs"""
    answer = (
        sum(state["values"])
        if state["operation"] == "+"
        # else reduce(lambda x, y: x * y, state["values"])
        else math.prod(state["values"])
    )
    state["result"] = f"Hi  {state['name']}! Your answer is  = {answer}"

    return state


graph = StateGraph(AgentState)

graph.add_node("processor", process_values)
graph.set_entry_point("processor")
graph.set_finish_point("processor")


app = graph.compile()


result = app.invoke(
    {"values": [123, 123], "operation": "+", "name": "some value", "result": "malati"}
)


print(result["result"])
