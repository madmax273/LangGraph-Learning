from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class BatsmanState(TypedDict):
    runs: int
    balls_faced: int
    fours: int
    sixes: int
    strike_rate: float
    boundaries_per_ball: float
    boundary_percentage: float
    summary: str

graph = StateGraph(BatsmanState)

def strike_rate_calculator(state: BatsmanState):
    state["strike_rate"] = state["runs"] / state["balls_faced"] * 100
    return {"strike_rate": state["strike_rate"]}

def boundaries_per_ball_calculator(state: BatsmanState):
    state["boundaries_per_ball"] = (state["fours"] + state["sixes"]) / state["balls_faced"]
    return {"boundaries_per_ball": state["boundaries_per_ball"]}

def boundary_percentage_calculator(state: BatsmanState):
    state["boundary_percentage"] = (state["fours"] + state["sixes"]) / state["runs"] * 100
    return {"boundary_percentage": state["boundary_percentage"]}

def summary(state: BatsmanState):
    summary = f"""
    Runs: {state["runs"]}
    Balls Faced: {state["balls_faced"]}
    Fours: {state["fours"]}
    Sixes: {state["sixes"]}
    Strike Rate: {state["strike_rate"]}
    Boundaries Per Ball: {state["boundaries_per_ball"]}
    Boundary Percentage: {state["boundary_percentage"]}
    """
    
    state["summary"] = summary
    return {"summary": state["summary"]}

graph.add_node("strike_rate_calculator", strike_rate_calculator)
graph.add_node("boundaries_per_ball_calculator", boundaries_per_ball_calculator)
graph.add_node("boundary_percentage_calculator", boundary_percentage_calculator)
graph.add_node("summary", summary)

graph.add_edge(START, "strike_rate_calculator")
graph.add_edge(START, "boundaries_per_ball_calculator")
graph.add_edge(START, "boundary_percentage_calculator")

graph.add_edge("strike_rate_calculator", "summary")
graph.add_edge("boundaries_per_ball_calculator", "summary")
graph.add_edge("boundary_percentage_calculator", "summary")

graph.add_edge("summary", END)

workflow = graph.compile()

result = workflow.invoke({
    "runs": 100,
    "balls_faced": 50,
    "fours": 10,
    "sixes": 5
})

print(result["summary"])