import pymupdf
import tqdm
import os

# Function to highlight text in PDF
def highlight_text_in_pdf(pdf_file: str, highlight_text):
    page_number = 1
    src = pymupdf.open(pdf_file)  # Open the PDF
    for i, page in tqdm.tqdm(enumerate(src), desc="Searching in doc..."):
        text_instances = page.search_for(highlight_text)  # Find text to highlight
        if len(text_instances) > 0:
            page_number = i
            for inst in text_instances:
                page.add_highlight_annot(inst)  # Highlight text
            break

    tmp = [page_number-2, page_number-1, page_number, page_number+1, page_number+2]
    pages = []
    for page in tmp:
        if 0 <= page < len(src):
            pages.append(page)
    print(pages)

    # Save the modified PDF
    new_pdf_file = pdf_file.split("/")
    new_pdf_file[-1] = "new_" + new_pdf_file[-1]
    new_pdf_file = "/".join(new_pdf_file)

    if os.path.exists(new_pdf_file):
        os.remove(new_pdf_file)

    dst = pymupdf.open()
    dst.insert_pdf(src, from_page=pages[0], to_page=pages[-1], final=True)
    #dst.set_metadata({})
    dst.del_xml_metadata()
    dst.xref_set_key(-1, "ID", "null")
    #dst.scrub()
    dst.save(new_pdf_file, garbage=4, deflate=True, no_new_id=True)
    dst.close()
    src.close()

    return new_pdf_file, page_number + 1