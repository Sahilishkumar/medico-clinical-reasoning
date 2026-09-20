import chromadb


# Open our existing database
chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)


# Open the medical books collection
collection = chroma_client.get_collection(
    name="medical_books"
)


# Ask the user what they want to search
query = input("\nWhat do you want to search in the books?\n> ")


# Search the database
results = collection.query(
    query_texts=[query],
    n_results=5
)


print("\n====================================")
print("RELEVANT TEXTBOOK INFORMATION")
print("====================================")


# Display results
for i, document in enumerate(results["documents"][0]):

    book = results["metadatas"][0][i]["book"]

    print("\n------------------------------------")
    print("RESULT", i + 1)
    print("BOOK:", book)
    print("------------------------------------")

    print(document)