from typing import Dict, Any
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

    def _post_retrieval(self, retrieved_context: str):
        return retrieved_context

    def prompt_llm(self, prompt: Dict[str, Any]) -> str:
        """Prompt the LLM using batchs with the list of prompts"""
        return self.llm_chain.invoke(prompt)

    def get_prompt(self, query, context) -> str:
        """Return the input of the LLM"""
        return self.prompt({"context": context, "question": query})

    def answer_question(self, query: str, verbose: bool = False) -> str:
        query = self._pre_retrieval(query)

        retrieved_context, sources = self.db.get_context_from_query(query)

        retrieved_context = self._post_retrieval(retrieved_context)

        if verbose:
            print(f"\nContext from {sources} :\n{retrieved_context}\n")
        op = self.llm_chain.invoke({"context": retrieved_context, "question": query})
        result = op.split("Answer:")[-1].strip()

        if "\(" in result and "\)" in result:
            result = result.replace("\(", "$").replace("\)", "$")

        if "\[" in result and "\]" in result:
            result = result.replace("\[", "$$").replace("\]", "$$")

        result = result + f"\n\nSources : {', '.join(list(sources))}"
        if verbose:
            #print(f"\nOp : {op}")
            print(f"\nYour question : {query}\n\n Predicted result: {result}")
        return result, retrieved_context
