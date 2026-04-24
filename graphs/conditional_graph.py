from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal



class quadraticstate(TypedDict):
    a: int
    b: int
    c: int
    root1: float
    root2: float
    equation: str
    discriminant: int
    final_result: str    

graph = StateGraph(quadraticstate)

def equation(state: quadraticstate):
    return {"equation": f"{state['a']}x^2 + {state['b']}x + {state['c']} = 0"}

def calculate_discriminant(state: quadraticstate):
    state["discriminant"] = state["b"] ** 2 - 4 * state["a"] * state["c"]
    return {"discriminant": state["discriminant"]}

def decide_next(state: quadraticstate)->Literal["two_real_roots", "one_real_root", "no_real_roots"]:
    if state["discriminant"] > 0:
        return "two_real_roots"
    elif state["discriminant"] == 0:
        return "one_real_root"
    else:
        return "no_real_roots"

def two_real_roots(state: quadraticstate):
    root1 = (-state["b"] + (state["b"] ** 2 - 4 * state["a"] * state["c"]) ** 0.5) / (2 * state["a"])
    root2 = (-state["b"] - (state["b"] ** 2 - 4 * state["a"] * state["c"]) ** 0.5) / (2 * state["a"])
    return {"root1": root1, "root2": root2}

def one_real_root(state: quadraticstate):
    root = -state["b"] / (2 * state["a"])
    return {"root1": root, "root2": root}

def no_real_roots(state: quadraticstate):
    root1 = complex(-state["b"] / (2 * state["a"]), ((4 * state["a"] * state["c"] - state["b"] ** 2) ** 0.5) / (2 * state["a"]))
    root2 = complex(-state["b"] / (2 * state["a"]), -((4 * state["a"] * state["c"] - state["b"] ** 2) ** 0.5) / (2 * state["a"]))
    return {"root1": root1, "root2": root2}


def final_result(state: quadraticstate):
    return {"final_result": f"The roots of the equation {state['equation']} are {state['root1']} and {state['root2']}"}


graph.add_node("equation", equation)
graph.add_node("calculate_discriminant", calculate_discriminant)
graph.add_node("decide_next", decide_next)
graph.add_node("two_real_roots", two_real_roots)
graph.add_node("one_real_root", one_real_root)
graph.add_node("no_real_roots", no_real_roots)
graph.add_node("final_result", final_result)

graph.add_edge(START, "equation")
graph.add_edge("equation", "calculate_discriminant")
graph.add_conditional_edges("calculate_discriminant", decide_next)
graph.add_edge("two_real_roots", "final_result")
graph.add_edge("one_real_root", "final_result")
graph.add_edge("no_real_roots", "final_result")
graph.add_edge("final_result", END)

workflow = graph.compile()

initial_state = {"a": 4, "b": 4, "c": 1}

result = workflow.invoke(initial_state)

print(result)
