from argparse import Namespace
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddingmodel = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)

store = InMemoryStore(index={"embed": embeddingmodel})

Namespace1=("user","u1")
Namespace2=("user","u2")

store.put(Namespace1,"1",{"data":"user like maggi"})
store.put(Namespace1,"2",{"data":"user like pizza"})
store.put(Namespace1,"3",{"data":"user like burgers"})
store.put(Namespace1,"4",{"data":"user like pasta"})
store.put(Namespace1,"5",{"data":"user like burgers1"})
store.put(Namespace1,"6",{"data":"user like burgers2"})


store.put(Namespace2,"1",{"data":"user like pasta"})
store.put(Namespace2,"2",{"data":"user like burgers"})
store.put(Namespace2,"3",{"data":"user like burgers2"})
store.put(Namespace2,"4",{"data":"user like burgers3"})
store.put(Namespace2,"5",{"data":"user like burgers4"})
store.put(Namespace2,"6",{"data":"user like burgers5"})

print(store.get(Namespace1,"1"))

data=store.search(Namespace1)
for d in data:
    print(f"{d.namespace[0]}:{d.namespace[1]} - {d.value['data']}")

data=store.search(Namespace2, query="burgers", limit=1)
print(data[0].value['data'])
