from typing import Dict, Any, List, Tuple
import os
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import Pipeline
from ..data import VectorDB

class RagAgent:
    def __init__(self, pipe: Pipeline, db: VectorDB):
        template = (
            "You are an assistant for question-answering tasks. "
            "Use the following context and your knowledge to directly answer the following question without just repeating the context. "
            "If you need to write equations, use Markdown. Gradio supports rendering Markdown, which includes LaTeX equations when enclosed in double dollar signs ($$ for block equations) or single dollar signs ($ for inline equations). "
            "If you don't know the answer, just say simply say 'Sorry, I don't know'.\n"
            "<context>\n"
            "{context}\n"
            "</context>\n"
            "Question:\n"
            "{question}\n"
            "Answer:"
        )
        self.prompt = ChatPromptTemplate.from_template(template)

        self.hf_pipeline = HuggingFacePipeline(pipeline=pipe)
        self.llm_chain = self.prompt | self.hf_pipeline
        self.db = db

    def _pre_retrieval(self, query: str):
        return query

    def _post_retrieval(self, chunks: List[dict]):
        return chunks

    @staticmethod
    def choose_best_document(chunks: List[dict]) -> str:
        sources = list(set([chunk["name"] for chunk in chunks]))
        if len(sources) == 1:
            if os.environ.get("PERSIST_PATH") is None:
                return chunks[0]["url"], chunks[0]["name"]
            else:
                return chunks[0]["path"], chunks[0]["name"]
        
        counts = {}
        for chunk in chunks:
            if chunk["name"] not in counts.keys():
                counts[chunk["name"]] = 0
            counts[chunk["name"]] += 1 

        best_doc_name = max(counts, key=counts.get)
        for chunk in chunks:
            if chunk["name"] == best_doc_name:
                if os.environ.get("PERSIST_PATH") is None:
                    return chunk["url"], chunk["name"]
                else:
                    return chunk["path"], chunk["name"]
        return "", ""


    def prompt_llm(self, prompt: Dict[str, Any]) -> str:
        """Prompt the LLM using batchs with the list of prompts"""
        return self.llm_chain.invoke(prompt)

    def get_prompt(self, query, context) -> str:
        """Return the input of the LLM"""
        return self.prompt({"context": context, "question": query})

    def answer_question(self, query: str, verbose: bool = False) -> Tuple[str, str, str, str]:
        query = self._pre_retrieval(query)

        chunks = self.db.get_context_from_query(query)

        retrieved_context = "\n".join(chunk["text"] for chunk in chunks)
        sources = list(set([chunk["name"] for chunk in chunks]))

        chunks = self._post_retrieval(chunks)

        if verbose:
            print(f"\nContext from {sources} :\n{retrieved_context}\n")
        op = self.llm_chain.invoke({"context": retrieved_context, "question": query})
        result = op.split("Answer:")[-1].strip()

        if "\(" in result and "\)" in result:
            result = result.replace("\(", "$").replace("\)", "$")

        if "\[" in result and "\]" in result:
            result = result.replace("\[", "$$").replace("\]", "$$")

        #result = result + f"\n\nSources : {', '.join(list(sources))}"
        if verbose:
            #print(f"\nOp : {op}")
            print(f"\nYour question : {query}\n\n Predicted result: {result}")
        return result, retrieved_context, ', '.join(list(sources)), self.choose_best_document(chunks)
