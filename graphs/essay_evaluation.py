from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing_extensions import Annotated
import operator
from pydantic import BaseModel, Field

essay="""

Artificial Intelligence (AI) is rapidly transforming the job market in India, bringing both opportunities and challenges. As industries adopt automation and data-driven technologies, the nature of work is evolving across sectors such as IT, manufacturing, healthcare, finance, and education.

One of the most significant impacts of AI is the automation of repetitive and routine tasks. Jobs that involve data entry, basic customer support, or predictable processes are increasingly being handled by AI systems and chatbots. This has raised concerns about job displacement, particularly for low-skilled workers. However, it is important to note that AI is not just eliminating jobs—it is also creating new ones.

The rise of AI has led to increased demand for skilled professionals in areas such as machine learning, data science, cybersecurity, and AI development. Indian companies and startups are actively seeking talent with expertise in these fields, leading to a shift in skill requirements. As a result, there is a growing emphasis on upskilling and reskilling the workforce to remain relevant in the evolving job market.

AI is also enhancing productivity and efficiency in many industries. In healthcare, AI assists doctors in diagnosis and treatment planning. In agriculture, it helps farmers optimize crop yield through predictive analytics. These advancements are not replacing humans entirely but rather augmenting their capabilities, leading to better outcomes.

However, the transition is not without challenges. There is a widening skill gap, and many workers may struggle to adapt to new technologies. Additionally, unequal access to education and training can deepen economic disparities if not addressed properly.

In conclusion, AI is reshaping the job market in India by automating certain roles while creating new opportunities. The key to benefiting from this transformation lies in education, skill development, and proactive adaptation. With the right policies and mindset, India can leverage AI to drive economic growth and employment in the future.

"""

load_dotenv()

llm = ChatGroq(temperature=0, model="llama-3.1-8b-instant")


class outputSchema(BaseModel):
    score: int = Field(description="The score of the essay from 0 to 10")
    feedback: str = Field(description="The feedback for the essay")


class EssayState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]   
    average_score: float
    
structured_llm = llm.with_structured_output(outputSchema)
    
def language_feedbacker(state: EssayState):
    prompt = PromptTemplate.from_template("""
        Evaluate the language quality of this essay and provide a score (0-10) with feedback.
        
        Essay: {essay}
        """)
    chain = prompt | structured_llm
    output = chain.invoke({"essay": state["essay"]})
    return {"language_feedback": output.feedback,"individual_scores": [output.score]}



def analysis_feedbacker(state: EssayState):
    prompt = PromptTemplate.from_template("""
        Evaluate the analysis quality of this essay and provide a score (0-10) with feedback.
        
        Essay: {essay}
        """)
    chain = prompt | structured_llm
    output = chain.invoke({"essay": state["essay"]})
    return {"analysis_feedback": output.feedback,"individual_scores": [output.score]}

    

def clarity_feedbacker(state: EssayState):
    prompt = PromptTemplate.from_template("""
        Evaluate the clarity of this essay and provide a score (0-10) with feedback.
        
        Essay: {essay}
        """)
    chain = prompt | structured_llm
    output = chain.invoke({"essay": state["essay"]})
    return {"clarity_feedback": output.feedback,"individual_scores": [output.score]}



def overall_feedbacker(state: EssayState):
    prompt = PromptTemplate.from_template("""
        You are an essay evaluator. Evaluate the following essay and provide feedback on overall.
        Essay: {essay}
        """)
    llm = ChatGroq(temperature=0, model="llama-3.1-8b-instant",)    
    chain = prompt | llm
    output = chain.invoke({"essay": state["essay"]})
    avg_score = sum(state["individual_scores"]) / len(state["individual_scores"])
    return {"overall_feedback": output.content, "average_score": avg_score}

graph = StateGraph(EssayState)

graph.add_node("language_feedbacker", language_feedbacker)
graph.add_node("analysis_feedbacker", analysis_feedbacker)
graph.add_node("clarity_feedbacker", clarity_feedbacker)
graph.add_node("overall_feedbacker", overall_feedbacker)

graph.add_edge(START, "language_feedbacker")
graph.add_edge(START, "analysis_feedbacker")
graph.add_edge(START, "clarity_feedbacker")
graph.add_edge("language_feedbacker", "overall_feedbacker")
graph.add_edge("analysis_feedbacker", "overall_feedbacker")
graph.add_edge("clarity_feedbacker", "overall_feedbacker")
graph.add_edge("overall_feedbacker", END)

workflow = graph.compile()

initial_state = {"essay": essay}

result = workflow.invoke(initial_state)

print("language_feedback : ",result["language_feedback"])
print("analysis_feedback : ",result["analysis_feedback"])
print("clarity_feedback : ",result["clarity_feedback"])
print("overall_feedback : ",result["overall_feedback"])
print("average_score : ",result["average_score"])






