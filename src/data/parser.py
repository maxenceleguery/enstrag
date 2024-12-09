from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text

from langchain.docstore.document import Document
from typing import List

def get_documents_from_file(filename: str) -> List[Document]:
    # TODO Add pdf parser and detection of the extension.
    elements = partition_text(filename="attention.txt")
    texts = []
    for elem in elements:
        if elem.category.endswith("Text"):
            texts.append(str(elem))

    text = " ".join(texts)
    documents = [Document(page_content=text)]