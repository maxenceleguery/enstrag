from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import Pipeline
from ..data import VectorDB

class RagAgent:
    def __init__(self, pipe: Pipeline, db: VectorDB):
        template = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context and your knowledge to directly answer the question without just repeating the context. "
            "If you need to write equations, use Markdown. Gradio supports rendering Markdown, which includes LaTeX equations when enclosed in double dollar signs ($$ for block equations) or single dollar signs ($ for inline equations). "
            "If you don't know the answer, just say simply say 'Sorry, I don't know'.\n"
            "<context>\n"
            "{context}\n"
            "</context>\n"
            "Answer the following question:\n"
            "{question}\n"
            "Assistant :"
        )
        prompt = ChatPromptTemplate.from_template(template)

        self.llm_chain = prompt | HuggingFacePipeline(pipeline=pipe)
        self.db = db

    def _pre_retrieval(self, query: str):
        return query
    
    def _post_retrieval(self, retrieved_context: str):
        return retrieved_context

    def answer_question(self, query: str, verbose: bool = False) -> str:
        query = self._pre_retrieval(query)

        retrieved_context = self.db.get_context_from_query(query)

        retrieved_context = self._post_retrieval(retrieved_context)
        
        if verbose:
            print(f"\nContext :\n{retrieved_context}\n")
        op = self.llm_chain.invoke({"context": retrieved_context, "question": query})
        result = op.split("Assistant :")[-1].strip()
        if verbose:
            #print(f"\nOp : {op}")
            print(f"\nYour question : {query}\n\n Predicted result: {result}")
        return result, retrieved_context