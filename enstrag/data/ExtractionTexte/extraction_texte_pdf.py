from PyPDF2 import PdfReader
import os
import re
from tqdm import tqdm

pdf_file = "ExtractionTexte/" + "Input/" + "attention_is_all_you_need.pdf"
output_file = "ExtractionTexte/" + "Output/" + "output.txt"


def clean_text(text):
    # Conserve les caractères accentués et supprime les caractères non imprimables
    text = re.sub(r'[^\x20-\x7E\u00C0-\u00FF\u00B2\u00B3\u00B9\n]', '', text)
    # Remplace les espaces multiples par un espace simple, tout en conservant les retours à la ligne
    text = re.sub(r'[^\S\n]+', ' ', text).strip()
    return text


if not os.path.exists(pdf_file):
    print(f"Erreur : Le fichier '{pdf_file}' est introuvable.")
else:
    try:
        reader = PdfReader(pdf_file)
        


        with open(output_file, "w", encoding="utf-8") as f:
            for page_num, page in tqdm(enumerate(reader.pages), total=len(reader.pages), desc="Lecture des pages"):
                raw_text = page.extract_text()
                if raw_text:
                    cleaned_text = clean_text(raw_text)
                    f.write(f"--- Page {page_num + 1} ---\n")
                    f.write(cleaned_text + "\n\n")
        
        print(f"Le texte a été extrait et enregistré dans '{output_file}'.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")