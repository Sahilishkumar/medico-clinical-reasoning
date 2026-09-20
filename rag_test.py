import os

import chromadb

from dotenv import load_dotenv
from google import genai


# Load API key
load_dotenv()


# Connect to Gemini
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)


# Open medical books collection
collection = chroma_client.get_collection(
    name="medical_books"
)


# Patient history
history = """
56-year-old male construction worker.

Productive cough for 2 months.
Fever for 6 weeks with evening rise.
Progressive breathlessness for 3 weeks.
Left-sided pleuritic chest pain for 2 weeks.
Occasional streaky hemoptysis for 10 days.
Weight loss of 5 kg over 2 months.
Decreased appetite and night sweats.

Brother was treated for tuberculosis 1 year ago.

Diabetes mellitus for 6 years with irregular control.

Smokes 15 cigarettes per day for 30 years.

Works around construction dust.

No previous history of tuberculosis, COPD or asthma.

No physical examination or investigations are available.
"""


# Search medical books
results = collection.query(
    query_texts=[history],
    n_results=5
)


# Combine retrieved textbook chunks
textbook_context = ""


for i, document in enumerate(results["documents"][0]):

    book = results["metadatas"][0][i]["book"]

    textbook_context += f"""

SOURCE {i + 1}
BOOK: {book}

{document}

"""


# Create prompt for Gemini
prompt = f"""
You are a clinical reasoning assistant for a medical student.

Use the patient's history and the retrieved textbook material to
make a DIRECT clinical prediction.

PATIENT HISTORY:
{history}

TEXTBOOK MATERIAL:
{textbook_context}

IMPORTANT:
- Give the most likely diagnosis directly.
- Keep the response SHORT.
- Do not write a long clinical report.
- Do not repeat the entire patient history.
- Do not invent examination findings or investigation results.
- Use the textbooks as the primary reference.
- Mention uncertainty when appropriate.
- This is an educational prediction, not a definitive diagnosis.

FORMAT EXACTLY LIKE THIS:

MOST LIKELY:
[one diagnosis]

WHY:
[2 to 4 short sentences explaining the strongest clues]

DIFFERENTIALS:
1. [diagnosis] — [very short reason]
2. [diagnosis] — [very short reason]
3. [diagnosis] — [very short reason]

NEXT:
[2 to 4 most important examination findings or investigations to check]

CONFIDENCE:
[High / Moderate / Low]

Do not add any other sections.
"""

# Ask Gemini
response = client.interactions.create(
    model="gemini-3.6-flash",
    input=prompt
)


# Display answer
print("\n====================================")
print("CLINICAL REASONING")
print("====================================\n")

print(response.output_text)