import gradio as gr
#import fitz  # PyMuPDF for PDF processing
import pymupdf

# Function to highlight text in PDF
def highlight_text_in_pdf(pdf_file, highlight_text):
    doc = pymupdf.open(pdf_file.name)  # Open the PDF
    for page in doc:
        text_instances = page.search_for(highlight_text)  # Find text to highlight
        for inst in text_instances:
            page.add_highlight_annot(inst)  # Highlight text
    # Save the modified PDF
    highlighted_pdf_path = "highlighted_output.pdf"
    doc.save(highlighted_pdf_path)
    return highlighted_pdf_path

# Chatbot function
def chatbot_response(message, pdf_file):
    response = f"You said: {message}"
    highlighted_pdf = highlight_text_in_pdf(pdf_file, message)  # Highlight message in PDF
    return response, highlighted_pdf  # Return chatbot response & modified PDF

# Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.ChatInterface(fn=chatbot_response)
    pdf_input = gr.File(label="Upload PDF", type="file")
    output_pdf = gr.File(label="Highlighted PDF")

    chatbot.chatbot.value = lambda msg: chatbot_response(msg, pdf_input.value)
    chatbot.chatbot.update(fn=chatbot_response, inputs=[gr.Textbox(), pdf_input], outputs=[gr.Textbox(), output_pdf])

demo.launch(share=True)
