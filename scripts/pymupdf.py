import os
import requests
import pwd
from enstrag.front.utils import highlight_text_in_pdf

def test_highlight():
    url = "http://www.cs.man.ac.uk/~fumie/tmp/bishop.pdf"
    TMP_FOLDER = "./tests/tmp_" + str(pwd.getpwuid(os.getuid())[0])
    os.makedirs(TMP_FOLDER, exist_ok=True)

    with open(os.path.join(TMP_FOLDER, 'tmp.pdf'), 'wb') as f:
        try:
            response = requests.get(url)
            f.write(response.content)
        except Exception:
            print(f"Failed to download {url}.")            

    pdf, page_number = highlight_text_in_pdf(os.path.join(TMP_FOLDER, 'tmp.pdf'), "Aside from")
    print(pdf)
    print(page_number)

if __name__ == "__main__":
    test_highlight()