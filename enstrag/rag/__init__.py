from langchain.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain_huggingface import HuggingFacePipeline
from transformers import Pipeline
from ..data import VectorDB

class RagAgent:
    def __init__(self, pipe: Pipeline, db: VectorDB):
        template = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context and your knowledge to answer the question without just repeating the context. If you don't know the answer, just say that you don't know.

        <context>
        The main subject is about Large Language models.
        {context}
        </context>

        Answer the following question:

        {question}<ENDofprompt>"""
        prompt = ChatPromptTemplate.from_template(template)

        self.llm_chain = LLMChain(prompt=prompt, llm=HuggingFacePipeline(pipeline=pipe))
        self.db = db

    def answer_question(self, query: str, verbose: bool = False) -> str:
        retrieved_context = self.db.get_context_from_query(query)
        if verbose:
            print(f"\nContext : {retrieved_context}")
        op = self.llm_chain.invoke({"context": retrieved_context, "question": query})
        result = op["text"].split("<ENDofprompt>")[-1]
        if verbose:
            print(f"\nOp : {op}")
            print(f"\nYour question : {query}\n\n Predicted result: {result}")
        return result, retrieved_context