from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text
from PyPDF2 import PdfReader
import requests
import os
import re
from langchain.docstore.document import Document
from typing import List
from hashlib import sha256
from dataclasses import dataclass

@dataclass
class FileDocument:
    url: str
    name: str
    label: str

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
    def get_text_from_txt(filename: str) -> str:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist")
        if not filename.endswith(".txt"):
            raise ValueError(f"Text file is expected. Got {filename}")
        
        elements = partition_text(filename=filename)
        texts = []
        for elem in elements:
            if elem.category.endswith("Text"):
                texts.append(str(elem))

        return " ".join(texts)
        return [Document(page_content=text, metadata={"hash": sha256(text.encode('utf-8')).hexdigest(), "name": filename})]

    @staticmethod
    def get_text_from_pdf(path_to_pdf: str, name: str = None) -> str:
        if not os.path.exists(path_to_pdf):
            raise FileNotFoundError(f"File {path_to_pdf} does not exist")
        if not path_to_pdf.endswith(".pdf"):
            raise ValueError(f"PDF file is expected. Got {path_to_pdf}")
        
        reader = PdfReader(path_to_pdf)
        texts = []
        for _, page in enumerate(reader.pages):
            raw_text = page.extract_text()
            if raw_text:
                cleaned_text = Parser.clean_text(raw_text)
                texts.append(cleaned_text+"\n\n")

        return " ".join(texts)

        return [Document(page_content=text, metadata={"hash": sha256(text.encode('utf-8')).hexdigest(), "name": name})]

    @staticmethod
    def get_text_from_pdf_url(url: str, name: str = None) -> str:
        if name is None:
            name = url

        if not os.path.exists("/tmp/enstrag"):
            os.makedirs("/tmp/enstrag", exist_ok=True)
            os.chmod("/tmp/enstrag", 666)
        with open('/tmp/enstrag/tmp.pdf', 'wb') as f:
            try:
                response = requests.get(url)
                f.write(response.content)
            except Exception:
                print(f"Failed to download {url}. Ignoring...")
                return ""

        text = Parser.get_text_from_pdf('/tmp/enstrag/tmp.pdf', name=name)
        os.remove('/tmp/enstrag/tmp.pdf')
        return text

    @staticmethod
    def get_document_from_filedoc(filedoc: FileDocument) -> Document:
        text =  Parser.get_text_from_pdf_url(filedoc.url, filedoc.name)
        if text == "":
            return None
        return Document(page_content=text, metadata={"hash": sha256(text.encode('utf-8')).hexdigest(), "name": filedoc.name, "label": filedoc.label})
    
    @staticmethod
    def get_documents_from_filedocs(filedocs: List[FileDocument]) -> List[Document]:
        docs = []
        for filedoc in filedocs:
            doc = Parser.get_document_from_filedoc(filedoc)
            if doc is not None:
                docs.append(doc)
        return docs