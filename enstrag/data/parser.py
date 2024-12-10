from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text
from PyPDF2 import PdfReader
import requests
import os
import re
from langchain.docstore.document import Document
from typing import List

class Parser:
    def __init__(self):
        pass
    
    @staticmethod
    def clean_text(text):
        # Conserve les caractères accentués et supprime les caractères non imprimables
        text = re.sub(r'[^\x20-\x7E\u00C0-\u00FF\u00B2\u00B3\u00B9\n]', '', text)
        # Remplace les espaces multiples par un espace simple, tout en conservant les retours à la ligne
        text = re.sub(r'[^\S\n]+', ' ', text).strip()
        return text

    @staticmethod
    def get_documents_from_txt(filename: str) -> List[Document]:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist")
        if not filename.endswith(".txt"):
            raise ValueError(f"Text file is expected. Got {filename}")
        
        elements = partition_text(filename=filename)
        texts = []
        for elem in elements:
            if elem.category.endswith("Text"):
                texts.append(str(elem))

        text = " ".join(texts)
        return [Document(page_content=text)]

    @staticmethod
    def get_documents_from_pdf(filename: str) -> List[Document]:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist")
        if not filename.endswith(".pdf"):
            raise ValueError(f"PDF file is expected. Got {filename}")
        
        reader = PdfReader(filename)
        texts = []
        for _, page in enumerate(reader.pages):
            raw_text = page.extract_text()
            if raw_text:
                cleaned_text = Parser.clean_text(raw_text)
                texts.append(cleaned_text+"\n\n")

        text = " ".join(texts)
        return [Document(page_content=text)]

    @staticmethod
    def get_documents_from_pdf_url(url: str) -> List[Document]:
        os.makedirs("/tmp/enstrag", exist_ok=True)
        with open('/tmp/enstrag/tmp.pdf', 'wb') as f:
            response = requests.get(url)
            f.write(response.content)

        docs = Parser.get_documents_from_pdf('/tmp/enstrag/tmp.pdf')
        os.remove('/tmp/enstrag/tmp.pdf')
        return docs

    @staticmethod
    def get_documents_from_pdf_urls(urls: List[str]) -> List[Document]:
        docs = []
        for url in urls:
            docs.extend(Parser.get_documents_from_pdf_url(url))
        return docs