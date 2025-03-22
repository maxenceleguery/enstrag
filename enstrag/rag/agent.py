from typing import Dict, Any, List, Tuple, Literal
import os
from torch.cuda import empty_cache, memory_allocated
import numpy as np
from numpy.linalg import norm
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import Pipeline

from ..explanation.pipeline import PerturbationPipeline, GradientPipeline
from ..explanation.generate import SimpleGenerator
from ..explanation.compare import EmbeddingComparator
from ..explanation.perturber import Perturber, LeaveNounsOutPerturber

from ..data import VectorDB, Parser, FileDocument, store_filedoc, load_filedocs

class RagAgent:
    def __init__(self, pipe: Pipeline, db: VectorDB, perturber: Perturber = LeaveNounsOutPerturber):
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
        
        # For decoder-only architecture
        pipe.tokenizer.padding_side = "left"

        self.pipeline_xrag_grad = GradientPipeline(pipe, self.db.embedding, self.prompt)
        self.pipeline_xrag_pert = PerturbationPipeline(perturber, SimpleGenerator(), EmbeddingComparator(), pipe.tokenizer, self, self.db.embedding)

    def top_k_tokens(self, prompt: Dict[str, Any], k: int, method: Literal["gradient", "perturbation"] = "perturbation") -> List[str]:
        if method == "perturbation":
            tokens = self.pipeline_xrag_pert.top_k_tokens(prompt, k)
        elif method == "gradient":
            print(memory_allocated())
            tokens = self.pipeline_xrag_grad.top_k_tokens(prompt, k)
            print(memory_allocated())
        else:
            raise ValueError(f"Wrong method for explanation. Got {method}")
        
        empty_cache()
        return tokens

    def get_themes(self):
        docs = load_filedocs()
        themes = list(set([doc.label for doc in docs]))
        themes.sort()
        return themes
    
    def add_filedoc(self, filedoc: FileDocument):
        if filedoc.local_path is None:
            filedoc.local_path = Parser.download_pdf(filedoc.url, filedoc.name)
        store_filedoc(filedoc)
        self.add_document(Parser.get_document_from_filedoc(filedoc, get_pages_num=False))

    def get_docs(self):
        docs = load_filedocs()
        return docs
    
    def add_document(self, document) -> None:
        return self.db.add_document(document)

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


    def prompt_llm(self, prompts: list[dict[str, Any]]) -> list[str]:
        """Prompt the LLM using batches with the list of prompts"""
        format_prompts = [self.prompt.format(**prompt) for prompt in prompts]
        return self.hf_pipeline.batch(format_prompts)


    def get_prompt(self, query, context) -> str:
        """Return the input of the LLM"""
        return self.prompt({"context": context, "question": query})
    
    def get_best_chunks_by_sim(self, chunks: List[dict], answer: str) -> str:
        chunks_vectors = np.array(self.db.db.embeddings.embed_documents([chunk["text"] for chunk in chunks]))
        answer_vector = np.array(self.db.db.embeddings.embed_query(answer))

        cosine_sim = np.dot(chunks_vectors, answer_vector)/(norm(chunks_vectors)*norm(answer_vector))
        print("Cosine_sim:", cosine_sim)
        best_chunk_id = np.argmax(cosine_sim)
        if os.environ.get("PERSIST_PATH") is None:
            return chunks[best_chunk_id]["url"], chunks[best_chunk_id]["name"]
        else:
            return chunks[best_chunk_id]["path"], chunks[best_chunk_id]["name"], chunks[best_chunk_id]["text"]
        
    def get_best_chunks_by_sim_for_evaluation(self, chunks: List[dict], answer: str) -> Tuple[str, str, str, List[str]]:
        chunks_vectors = np.array(self.db.db.embeddings.embed_documents([chunk["text"] for chunk in chunks]))
        answer_vector = np.array(self.db.db.embeddings.embed_query(answer))

        similarity_scores = np.dot(chunks_vectors, answer_vector) / (norm(chunks_vectors) * norm(answer_vector))
        # similarity_scores = -np.linalg.norm(chunks_vectors - answer_vector, axis=1)
        # similarity_scores = -np.sum(np.abs(chunks_vectors - answer_vector), axis=1)

        best_chunk_id = np.argmax(similarity_scores)
        if os.environ.get("PERSIST_PATH") is None:
            best_chunk = (chunks[best_chunk_id]["url"], chunks[best_chunk_id]["name"], chunks[best_chunk_id]["text"], similarity_scores)
        else:
            best_chunk = (chunks[best_chunk_id]["path"], chunks[best_chunk_id]["name"], chunks[best_chunk_id]["text"], similarity_scores)

        # Classify chunks based on cosine similarity
        classified_chunks = [chunk["text"] for _, chunk in sorted(zip(similarity_scores, chunks), reverse=True)]

        return best_chunk[0], best_chunk[1], best_chunk[2], classified_chunks

    def answer_question(self, query: str, topk_context: int = 4, verbose: bool = False) -> Tuple[str, str, str, str]:
        query = self._pre_retrieval(query)

        chunks = self.db.get_context_from_query(query, topk=topk_context)

        retrieved_context = "\n".join(chunk["text"] for chunk in chunks)
        sources = list(set([chunk["name"] for chunk in chunks]))

        chunks = self._post_retrieval(chunks)
        print("chunks", chunks)

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
        return result, retrieved_context, ', '.join(list(sources)), self.get_best_chunks_by_sim(chunks, result)

    def answer_question_for_evaluation(self, query: str, topk_context: int = 4, verbose: bool = False) -> Tuple[str, str, str, Tuple[str, str, str, np.ndarray], str]:
        query = self._pre_retrieval(query)

        chunks = self.db.get_context_from_query(query, topk=topk_context)

        retrieved_context = "\n".join(chunk["text"] for chunk in chunks)
        sources = list(set([chunk["name"] for chunk in chunks]))

        chunks = self._post_retrieval(chunks)

        if verbose:
            print(f"\nContext from {sources} :\n{retrieved_context}\n")
        op = self.llm_chain.invoke({"context": retrieved_context, "question": query})
        result = op.split("Answer:")[-1].strip()

        if r"\(" in result and r"\)" in result:
            result = result.replace(r"\(", "$").replace(r"\)", "$")

        if r"\[" in result and r"\]" in result:
            result = result.replace(r"\[", "$$").replace(r"\]", "$$")

        if verbose:
            print(f"\nYour question : {query}\n\n Predicted result: {result}")
        
        return result, retrieved_context, ', '.join(list(sources)), self.get_best_chunks_by_sim_for_evaluation(chunks, result), chunks