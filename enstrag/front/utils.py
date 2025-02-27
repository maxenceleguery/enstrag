import pymupdf
import tqdm

# Function to highlight text in PDF
def highlight_text_in_pdf(pdf_file: str, highlight_text):
    page_number = 1
    doc = pymupdf.open(pdf_file)  # Open the PDF
    for page in tqdm.tqdm(doc, desc="Searching in doc..."):
        text_instances = page.search_for(highlight_text)  # Find text to highlight
        if len(text_instances) > 0:
            page_number = page.number
        for inst in text_instances:
            page.add_highlight_annot(inst)  # Highlight text

    tmp = [page_number-2, page_number-1, page_number, page_number+1, page_number+2]
    pages = []
    for page in tmp:
        if 0 <= page < len(doc):
            pages.append(page)
    print(pages)
    doc.select(pages)

    # Save the modified PDF
    new_pdf_file = pdf_file.split("/")
    new_pdf_file[-1] = "new_" + new_pdf_file[-1]
    new_pdf_file = "/".join(new_pdf_file)

    doc.save(new_pdf_file)
    return new_pdf_file, page_number + 1