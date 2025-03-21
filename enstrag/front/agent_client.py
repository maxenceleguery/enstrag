import requests
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

@dataclass
class FileDocument:
    url: str | None
    local_path: str | None
    name: str
    label: str

class AgentClient():
    def __init__(self, url: str = "http://127.0.0.1", port: int = 8000):
        self.API_URL = os.getenv("API_URL", "http://rag-app:8000")

        while True:
            try:
                if (200 <= requests.get(f"{self.API_URL}/").status_code <= 299):
                    print("Starting Gradio...")
                    return
            except requests.exceptions.ConnectionError:
                print("Waiting server...")
                time.sleep(10)

    def get_themes(self):
        response = requests.get(f"{self.API_URL}/themes")
        if 400 <= response.status_code <= 599:
            raise RuntimeError(f"themes endpoint failed with status code {response.status_code}")
        body = response.json()
        return body["themes"]
    
    def get_docs(self):
        response = requests.get(f"{self.API_URL}/documents")
        if 400 <= response.status_code <= 599:
            raise RuntimeError(f"themes endpoint failed with status code {response.status_code}")
        docs_dict = response.json()
        docs = [FileDocument(**doc) for doc in docs_dict]
        return docs
    
    def add_filedoc(self, filedoc: FileDocument):
        print(f"Sending {filedoc}")
        response = requests.post(f"{self.API_URL}/doc", json=asdict(filedoc))
        if 400 <= response.status_code <= 599:
            raise RuntimeError(f"doc endpoint failed with status code {response.status_code}")

    def answer_question(self, query: str, topk_context: int = 4, verbose: bool = False):
        response = requests.get(f"{self.API_URL}/answer", params={"query": query, "topk_context": topk_context})
        
        if 400 <= response.status_code <= 599:
            raise RuntimeError(f"answer endpoint failed with status code {response.status_code}")
        
        body = response.json()
        
        return body["result"], body["retrieved_context"], body["sources"], (body["pdf_path"], body["pdf_name"], body["context_to_highlight"])

    def top_k_tokens(self, prompt: Dict[str, Any], k: int, method: str) -> List[str]:
        prompt.update({"k": k, "method": method})
        response = requests.get(f"{self.API_URL}/topk", params=prompt)

        if 400 <= response.status_code <= 599:
            raise RuntimeError(f"topk endpoint failed with status code {response.status_code}")
        
        body = response.json()
        print("Receiving : ", body)

        return body
    
if __name__ == "__main__":
    agent = AgentClient()

    print(agent.get_themes())
    #print(agent.answer_question("What is a gaussian distribution ?"))
    print(agent.get_docs())