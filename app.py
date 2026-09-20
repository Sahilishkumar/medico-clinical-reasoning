import os
import chromadb

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai


# Load environment variables
load_dotenv()


# Flask application
app = Flask(__name__)


# Gemini
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ChromaDB
chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_collection(
    name="medical_books"
)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Health check
@app.route("/health")
def health():
    return jsonify({
        "status": "ok"
    })


# Clinical analysis
@app.route("/analyse", methods=["POST"])
def analyse():

    data = request.get_json()

    history = data.get("history", "").strip()

    if not history:
        return jsonify({
            "error": "Please enter the patient's history."
        }), 400


    print("STEP 1: History received", flush=True)


    # --------------------------------
    # SEARCH MEDICAL TEXTBOOKS
    # --------------------------------

    try:

        results = collection.query(
            query_texts=[history],
            n_results=5
        )

        print(
            "STEP 2: Chroma search completed",
            flush=True
        )

    except Exception as e:

        print(
            "CHROMA ERROR:",
            str(e),
            flush=True
        )

        return jsonify({
            "error": "Textbook search failed: " + str(e)
        }), 500


    # --------------------------------
    # PREPARE TEXTBOOK CONTEXT
    # --------------------------------

    textbook_context = ""

    for i, document in enumerate(
        results["documents"][0]
    ):

        book = results["metadatas"][0][i]["book"]

        textbook_context += f"""
SOURCE {i + 1}
BOOK: {book}

{document}
"""


    print(
        "STEP 3: Textbook context prepared",
        flush=True
    )


    # --------------------------------
    # GEMINI PROMPT
    # --------------------------------

    prompt = f"""
You are a clinical reasoning assistant for a medical student.

Use the patient's history and the retrieved textbook material to make
a DIRECT clinical prediction.

PATIENT HISTORY:

{history}


TEXTBOOK MATERIAL:

{textbook_context}


IMPORTANT:

- Give the most likely diagnosis directly.
- Keep the response SHORT.
- Maximum 120 words.
- Do not write a long clinical report.
- Do not repeat the entire patient history.
- Do not invent examination findings.
- Do not invent investigation results.
- Use the supplied textbooks to support your reasoning.
- Do not mention the textbooks or source numbers.
- Do not quote textbook passages.
- Clearly indicate uncertainty when important.
- This is an educational clinical reasoning tool, not a definitive diagnosis.


FORMAT:

MOST LIKELY:
[diagnosis]

WHY:
[maximum 2 short lines]

DIFFERENTIALS:
1. [diagnosis] — [few words]
2. [diagnosis] — [few words]
3. [diagnosis] — [few words]

NEXT:
[maximum 3 investigations or examination steps]

CONFIDENCE:
[High / Moderate / Low]

CONFIDENCE REASON:
[one short sentence]
"""


    print(
        "STEP 4: Sending request to Gemini",
        flush=True
    )


    # --------------------------------
    # GEMINI
    # --------------------------------

    try:

        response = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        print(
            "STEP 5: Gemini response received",
            flush=True
        )


        return jsonify({
            "result": response.output_text
        })


    except Exception as e:

        print(
            "GEMINI ERROR:",
            str(e),
            flush=True
        )

        return jsonify({
            "error": "Gemini request failed: " + str(e)
        }), 500


# --------------------------------
# START SERVER
# --------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )