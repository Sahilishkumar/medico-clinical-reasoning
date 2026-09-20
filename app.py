import os
import chromadb
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai

load_dotenv()

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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/analyse", methods=["POST"])
def analyse():

    data = request.get_json()

    history = data.get("history", "").strip()

    if not history:
        return jsonify({
            "error": "Please enter the patient's history."
        }), 400

    # Search medical textbooks
    results = collection.query(
        query_texts=[history],
        n_results=5
    )

    # Prepare textbook context
    textbook_context = ""

    for i, document in enumerate(results["documents"][0]):

        book = results["metadatas"][0][i]["book"]

        textbook_context += f"""
SOURCE {i + 1}
BOOK: {book}

{document}
"""

    # Clinical reasoning prompt
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
- Do not invent examination findings.
- Do not invent investigation results.
- Use the supplied textbooks as the primary reference.
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

CONFIDENCE REASON:
[one short sentence]

Do not add any other sections.
"""

    try:

        response = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        return jsonify({
            "result": response.output_text
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )