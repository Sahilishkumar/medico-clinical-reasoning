import os
from pypdf import PdfReader


BOOKS_FOLDER = "books"


for filename in os.listdir(BOOKS_FOLDER):

    if filename.lower().endswith(".pdf"):

        filepath = os.path.join(BOOKS_FOLDER, filename)

        print("\n====================================")
        print("BOOK:", filename)
        print("====================================")

        reader = PdfReader(filepath)

        print("Number of pages:", len(reader.pages))

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

        print("Characters extracted:", len(text))

        print("\nFirst 1000 characters:")
        print(text[:1000])