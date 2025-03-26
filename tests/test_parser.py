import os

from enstrag.data import Parser, FileDocument

def test_download():
    os.environ["PERSIST_PATH"] = "/tmp/enstrag"
    
    doc = FileDocument("http://www.cs.man.ac.uk/~fumie/tmp/bishop.pdf", None, "ML Bishop", "Machine learning")
    try:
        docs = Parser.get_documents_from_filedocs([doc], get_pages_num=False)
    except FileNotFoundError:
        return

    assert len(docs) == 1 
    assert docs[0].page_content != "", "Got an empty document."
    assert docs[0].metadata["name"] == doc.name
    assert docs[0].metadata["label"] == doc.label

def test_text_cleaning():
    text = "   \n \t Hello  world  !"
    groud_truth = "Hello world !"
    assert Parser.clean_text(text) == groud_truth