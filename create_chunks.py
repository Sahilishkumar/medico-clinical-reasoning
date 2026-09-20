import os

import chromadb

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


BOOKS_FOLDER = "books"


# Create Chroma database
chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)


# Create collection
collection = chroma_client.get_or_create_collection(
    name="medical_books"
)


# Text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)


# Read all books
for filename in os.listdir(BOOKS_FOLDER):

    if not filename.lower().endswith(".pdf"):
        continue


    filepath = os.path.join(
        BOOKS_FOLDER,
        filename
    )


    print("\nProcessing:", filename)


    reader = PdfReader(filepath)

    full_text = ""


    # Extract text
    for page in reader.pages:

        text = page.extract_text()

        if text:
            full_text += text + "\n"


    print(
        "Characters extracted:",
        len(full_text)
    )


    # Create chunks
    chunks = splitter.split_text(
        full_text
    )


    print(
        "Chunks created:",
        len(chunks)
    )


    # Store chunks
    ids = []

    documents = []

    metadatas = []


    for i, chunk in enumerate(chunks):

        ids.append(
            filename + "_" + str(i)
        )

        documents.append(chunk)

        metadatas.append({
            "book": filename
        })


    # Add to Chroma
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )


print("\n================================")
print("DATABASE CREATED")
print("================================")

print(
    "Total chunks:",
    collection.count()
)