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
    url: str
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
        if doc["url"] == filedoc.url or doc["name"] == filedoc.name:
            return
        
    filedocs.append(
        {
            "url": filedoc.url,
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
            return [FileDocument(file["url"], file["name"], file["label"]) for file in json.load(f)]
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
    def get_text_from_pdf(path_to_pdf: str, backend: Literal["PyPDF2", "pymupdf"] = "PyPDF2") -> str:
        if not os.path.exists(path_to_pdf):
            raise FileNotFoundError(f"File {path_to_pdf} does not exist")
        if not path_to_pdf.endswith(".pdf"):
            raise ValueError(f"PDF file is expected. Got {path_to_pdf}")
        
        texts = []
        if backend == "PyPDF2":
            reader = PdfReader(path_to_pdf)
            for _, page in enumerate(reader.pages):
                raw_text = page.extract_text()
                if raw_text:
                    cleaned_text = Parser.clean_text(raw_text)
                    texts.append(cleaned_text+"\n\n")

        elif backend == "pymupdf":
            doc = pymupdf.open(path_to_pdf)
            for page in doc:
                cleaned_text = Parser.clean_text(page.get_textpage().extractText())
                texts.append(cleaned_text+"\n\n")

        else:
            raise ValueError(f"Wrong pdf extraction backend. Got {backend} instead of 'PyPDF2' or 'pymupdf'")

        return " ".join(texts)

    @staticmethod
    def get_text_from_pdf_url(url: str, name: str = None) -> str:
        if os.environ.get("PERSIST_PATH") is not None:
            TMP_FOLDER = os.path.join(os.environ.get("PERSIST_PATH"), "pdfs") 
        else:
            TMP_FOLDER = "/tmp/enstrag_"+str(pwd.getpwuid(os.getuid())[0])

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

        text = Parser.get_text_from_pdf(pdf_path)

        if os.environ.get("PERSIST_PATH") is None:
            shutil.rmtree(TMP_FOLDER)
        return text, pdf_path

    @staticmethod
    def get_document_from_filedoc(filedoc: FileDocument) -> Document:
        store_filedoc(filedoc)
        text, pdf_path =  Parser.get_text_from_pdf_url(filedoc.url, filedoc.name)
        if text == "":
            return None
        return Document(page_content=text, metadata={"hash": sha256(text.encode('utf-8')).hexdigest(), "name": filedoc.name, "label": filedoc.label, "url": filedoc.url, "path": pdf_path})
    
    @staticmethod
    def get_documents_from_filedocs(filedocs: List[FileDocument]) -> List[Document]:
        docs = []
        for filedoc in filedocs:
            doc = Parser.get_document_from_filedoc(filedoc)
            if doc is not None:
                docs.append(doc)
        return docs