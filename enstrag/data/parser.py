from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text
from PyPDF2 import PdfReader
import pymupdf
import requests
import os
import pwd
import re
import shutil
import json
from langchain.docstore.document import Document
from typing import List, Literal
from hashlib import sha256
from dataclasses import dataclass

@dataclass
class FileDocument:
    url: str | None
    local_path: str | None
    name: str
    label: str

def store_filedoc(filedoc: FileDocument):
    if (folder := os.environ.get("PERSIST_PATH")) is None:
        return
    
    json_database = os.path.join(folder, "filedocs.json")
    if os.path.exists(json_database):
        with open(json_database, "r") as f:
            filedocs = json.load(f)
    else:
        filedocs = []
    for doc in filedocs:
        if doc["url"] == filedoc.url or doc["local_path"] == filedoc.local_path or doc["name"] == filedoc.name:
            return
        
    filedocs.append(
        {
            "url": filedoc.url,
            "local_path": filedoc.local_path,
            "name": filedoc.name,
            "label": filedoc.label
        }
    )
    with open(json_database, "w") as f:
        json.dump(filedocs, f)

def load_filedocs() -> List[FileDocument]:
    if (folder := os.environ.get("PERSIST_PATH")) is None:
        return []
    
    json_database = os.path.join(folder, "filedocs.json")
    if os.path.exists(json_database):
        with open(json_database, "r") as f:
            return [FileDocument(file["url"], file["local_path"], file["name"], file["label"]) for file in json.load(f)]
    return []
            

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

    @staticmethod
    def get_text_from_pdf(path_to_pdf: str, backend: Literal["PyPDF2", "pymupdf"] = "PyPDF2") -> List[tuple[str, int]]:
        if not os.path.exists(path_to_pdf):
            raise FileNotFoundError(f"File {path_to_pdf} does not exist")
        if not path_to_pdf.endswith(".pdf"):
            raise ValueError(f"PDF file is expected. Got {path_to_pdf}")
        
        texts = []
        if backend == "PyPDF2":
            reader = PdfReader(path_to_pdf)
            for page_number, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text()
                if raw_text:
                    cleaned_text = Parser.clean_text(raw_text)
                    texts.append((cleaned_text, page_number))

        elif backend == "pymupdf":
            doc = pymupdf.open(path_to_pdf)
            for page_number, page in enumerate(doc, start=1):
                cleaned_text = Parser.clean_text(page.get_textpage().extractText())
                texts.append((cleaned_text, page_number))

        else:
            raise ValueError(f"Wrong pdf extraction backend. Got {backend} instead of 'PyPDF2' or 'pymupdf'")

        return texts

    @staticmethod
    def download_pdf(url: str, name: str = None) -> str:
        if os.environ.get("PERSIST_PATH") is None:
            return ""
        TMP_FOLDER = os.path.join(os.environ.get("PERSIST_PATH"), "pdfs") 
        os.makedirs(TMP_FOLDER, exist_ok=True)

        if name is None:
            name = url.replace("/", "_")
        name = name.replace(" ", "_")

        pdf_path = os.path.join(TMP_FOLDER, f'{name}.pdf')
        if not os.path.exists(pdf_path):
            with open(pdf_path, 'wb') as f:
                try:
                    response = requests.get(url)
                    f.write(response.content)
                except Exception:
                    print(f"Failed to download {url}. Ignoring...")
                    return ""

        return pdf_path

    @staticmethod
    def get_text_from_pdf_url(url: str, name: str = None) -> str:
        pdf_path = Parser.download_pdf(url, name)
        text = Parser.get_text_from_pdf(pdf_path)
        return text, pdf_path

    @staticmethod
    def get_document_from_filedoc(filedoc: FileDocument) -> List[Document]:
        if filedoc.local_path is None:
            filedoc.local_path = Parser.download_pdf(filedoc.url, filedoc.name)
        text_pages = Parser.get_text_from_pdf(filedoc.local_path)
        store_filedoc(filedoc)
        if not text_pages:
            return []
        
        documents = []
        for text, page_number in text_pages:
            print('page_number', page_number)
            documents.append(Document(
                page_content=text,
                metadata={
                    "hash": sha256(text.encode('utf-8')).hexdigest(),
                    "name": str(filedoc.name),
                    "label": str(filedoc.label),
                    "url": str(filedoc.url),
                    "path": str(filedoc.local_path),
                    "page_number": str(page_number) # Add current page number to metadata
                }
            ))
        return documents

    @staticmethod
    def get_documents_from_filedocs(filedocs: List[FileDocument]) -> List[Document]:
        docs = []
        for filedoc in filedocs:
            documents = Parser.get_document_from_filedoc(filedoc)
            docs.extend(documents)
        return docs