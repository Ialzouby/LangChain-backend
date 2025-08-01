import os
from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import ChatOpenAI

from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests

openai_api_key = os.getenv("OPENAI_API_KEY")

# ---------------- FastAPI setup ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TopicRequest(BaseModel):
    topic: str

# ---------------- LangGraph Nodes ----------------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key=openai_api_key)

def search_web(topic):
    with DDGS() as ddgs:
        results = ddgs.text(topic, max_results=5)
        return [r["href"] for r in results if "href" in r]

def search_node(state):
    state["links"] = search_web(state["topic"])
    return state

def fetch_text(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        return soup.get_text()
    except Exception as e:
        return f"Failed to fetch: {e}"

def read_node(state):
    state["raw_texts"] = [fetch_text(url) for url in state["links"]]
    return state

def summarize_node(state):
    content = "\n\n".join(state["raw_texts"][:3])[:8000]
    prompt = ChatPromptTemplate.from_template("""
    You are a research assistant. Read the following web content and write a concise summary report on the topic: "{topic}"

    Content:
    {content}
    """)
    chain = prompt | llm
    summary = chain.invoke({"topic": state["topic"], "content": content})
    state["report"] = summary.content
    return state

# ---------------- LangGraph Flow ----------------
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    topic: str
    links: List[str]
    raw_texts: List[str]
    report: str

graph = StateGraph(GraphState)
graph.add_node("search", search_node)
graph.add_node("read", read_node)
graph.add_node("summarize", summarize_node)
graph.set_entry_point("search")
graph.add_edge("search", "read")
graph.add_edge("read", "summarize")
graph.add_edge("summarize", END)
research_graph = graph.compile()


# ---------------- API Route ----------------
from typing import List
from fastapi.responses import JSONResponse

class ResearchResponse(BaseModel):
    summary: str
    links: List[str]

@app.post("/api/research", response_model=ResearchResponse)
def generate_summary(request: TopicRequest):
    state = {"topic": request.topic}
    result = research_graph.invoke(state)
    return ResearchResponse(
        summary=result["report"],
        links=result["links"]
    )