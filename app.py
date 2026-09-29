from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory
)

from pypdf import PdfReader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from google import genai
from google.genai import types

import faiss
import os
import re
import numpy as np


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# GEMINI API SETUP
# =========================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not set."
    )


client = genai.Client(
    api_key=api_key
)


# =========================================================
# GEMINI MODELS
# =========================================================

GENERATION_MODEL = "gemini-3.8-flash"

EMBEDDING_MODEL = "gemini-embedding-001"

EMBEDDING_DIMENSION = 768


# =========================================================
# UPLOAD FOLDER
# =========================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================================================
# GLOBAL PDF DATA
# =========================================================

pdf_chunks = []

pdf_sources = []

pdf_pages = []

faiss_index = None


# =========================================================
# CHECK TAMIL TEXT
# =========================================================

def is_tamil(text):

    return bool(
        re.search(
            r"[\u0B80-\u0BFF]",
            text or ""
        )
    )


# =========================================================
# CREATE GEMINI EMBEDDINGS
# =========================================================

def create_embeddings(
    texts,
    task_type
):

    all_vectors = []

    batch_size = 50


    for start in range(
        0,
        len(texts),
        batch_size
    ):

        batch = texts[
            start:start + batch_size
        ]


        response = client.models.embed_content(

            model=EMBEDDING_MODEL,

            contents=batch,

            config=types.EmbedContentConfig(

                task_type=task_type,

                output_dimensionality=
                    EMBEDDING_DIMENSION

            )

        )


        vectors = [
            item.values
            for item in response.embeddings
        ]


        all_vectors.extend(
            vectors
        )


    return np.asarray(
        all_vectors,
        dtype="float32"
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# SERVE UPLOADED PDF FILES
# =========================================================
# This route is required so that when the user clicks
# the uploaded PDF filename, the PDF opens in the browser.
# =========================================================

@app.route(
    "/uploads/<path:filename>"
)
def uploaded_file(filename):

    return send_from_directory(

        app.config[
            "UPLOAD_FOLDER"
        ],

        filename

    )


# =========================================================
# UPLOAD PDFs
# =========================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_pdfs():

    global pdf_chunks
    global pdf_sources
    global pdf_pages
    global faiss_index


    # -----------------------------------------------------
    # GET FILES
    # -----------------------------------------------------

    files = request.files.getlist(
        "pdf"
    )


    if not files:

        return jsonify({

            "success": False,

            "message":
                "Please select at least one PDF."

        })


    # -----------------------------------------------------
    # RESET OLD PDF DATA
    # -----------------------------------------------------

    pdf_chunks = []

    pdf_sources = []

    pdf_pages = []

    faiss_index = None


    # -----------------------------------------------------
    # VARIABLES
    # -----------------------------------------------------

    all_text = ""

    total_pages = 0

    uploaded_files = []


    try:

        # -------------------------------------------------
        # TEXT SPLITTER
        # -------------------------------------------------

        splitter = (
            RecursiveCharacterTextSplitter(

                chunk_size=500,

                chunk_overlap=50

            )
        )


        # -------------------------------------------------
        # PROCESS EACH PDF
        # -------------------------------------------------

        for file in files:


            if not file.filename:

                continue


            # ---------------------------------------------
            # CHECK PDF
            # ---------------------------------------------

            if not file.filename.lower().endswith(
                ".pdf"
            ):

                continue


            # ---------------------------------------------
            # SAVE PDF
            # ---------------------------------------------

            filepath = os.path.join(

                app.config[
                    "UPLOAD_FOLDER"
                ],

                file.filename

            )


            file.save(
                filepath
            )


            # ---------------------------------------------
            # READ PDF
            # ---------------------------------------------

            reader = PdfReader(
                filepath
            )


            total_pages += len(
                reader.pages
            )


            uploaded_files.append(
                file.filename
            )


            # ---------------------------------------------
            # PROCESS EACH PAGE
            # ---------------------------------------------

            for page_number, page in enumerate(

                reader.pages,

                start=1

            ):


                page_text = (
                    page.extract_text()
                    or ""
                )


                if not page_text.strip():

                    continue


                # -----------------------------------------
                # STORE TEXT
                # -----------------------------------------

                all_text += (
                    page_text
                    + "\n"
                )


                # -----------------------------------------
                # SPLIT PAGE INTO CHUNKS
                # -----------------------------------------

                page_chunks = (
                    splitter.split_text(
                        page_text
                    )
                )


                # -----------------------------------------
                # STORE CHUNKS
                # -----------------------------------------

                for chunk in page_chunks:


                    pdf_chunks.append(
                        chunk
                    )


                    pdf_sources.append(
                        file.filename
                    )


                    pdf_pages.append(
                        page_number
                    )


        # -------------------------------------------------
        # CHECK EXTRACTED TEXT
        # -------------------------------------------------

        if not pdf_chunks:

            return jsonify({

                "success": False,

                "message":
                    "Could not extract text from the PDFs."

            })


        # -------------------------------------------------
        # CREATE DOCUMENT EMBEDDINGS
        # -------------------------------------------------

        document_embeddings = (
            create_embeddings(

                pdf_chunks,

                "RETRIEVAL_DOCUMENT"

            )
        )


        # -------------------------------------------------
        # NORMALIZE EMBEDDINGS
        # -------------------------------------------------

        faiss.normalize_L2(
            document_embeddings
        )


        # -------------------------------------------------
        # CREATE FAISS INDEX
        # -------------------------------------------------

        faiss_index = (
            faiss.IndexFlatIP(
                document_embeddings.shape[1]
            )
        )


        # -------------------------------------------------
        # ADD EMBEDDINGS TO FAISS
        # -------------------------------------------------

        faiss_index.add(
            document_embeddings
        )


        # -------------------------------------------------
        # RETURN SUCCESS
        # -------------------------------------------------

        return jsonify({

            "success": True,

            "files":
                uploaded_files,

            "file_count":
                len(uploaded_files),

            "pages":
                total_pages,

            "text_length":
                len(all_text),

            "chunks":
                len(pdf_chunks),

            "message":
                "Multiple PDFs uploaded successfully."

        })


    except Exception as e:


        return jsonify({

            "success": False,

            "message":
                f"Error processing PDFs: {str(e)}"

        })


# =========================================================
# ASK QUESTION
# =========================================================

@app.route(
    "/ask",
    methods=["POST"]
)
def ask_question():

    global pdf_chunks
    global pdf_sources
    global pdf_pages
    global faiss_index


    try:

        # -------------------------------------------------
        # GET REQUEST DATA
        # -------------------------------------------------

        data = request.get_json(
            silent=True
        ) or {}


        question = str(
            data.get(
                "question",
                ""
            )
        ).strip()


        language = data.get(
            "language",
            "en-IN"
        )


        # -------------------------------------------------
        # CHECK LANGUAGE
        # -------------------------------------------------

        if language not in (
            "en-IN",
            "ta-IN"
        ):

            language = "en-IN"


        # -------------------------------------------------
        # CHECK QUESTION
        # -------------------------------------------------

        if not question:

            return jsonify({

                "success": False,

                "message":
                    "Please enter a question."

            })


        # -------------------------------------------------
        # CHECK PDF UPLOAD
        # -------------------------------------------------

        if (
            not pdf_chunks
            or faiss_index is None
        ):

            return jsonify({

                "success": False,

                "message":
                    "Please upload PDF files first."

            })


        # -------------------------------------------------
        # CREATE QUESTION EMBEDDING
        # -------------------------------------------------

        question_embedding = (
            create_embeddings(

                [question],

                "RETRIEVAL_QUERY"

            )
        )


        # -------------------------------------------------
        # NORMALIZE QUESTION EMBEDDING
        # -------------------------------------------------

        faiss.normalize_L2(
            question_embedding
        )


        # -------------------------------------------------
        # SEARCH FAISS
        # -------------------------------------------------

        k = min(
            5,
            len(pdf_chunks)
        )


        scores, indices = (
            faiss_index.search(

                question_embedding,

                k

            )
        )


        # -------------------------------------------------
        # SIMILARITY THRESHOLD
        # -------------------------------------------------

        similarity_threshold = 0.35


        relevant_chunks = []

        source_names = []

        source_pages = []

        valid_scores = []


        # -------------------------------------------------
        # GET RELEVANT CHUNKS
        # -------------------------------------------------

        for i in range(k):


            index = int(
                indices[0][i]
            )


            score = float(
                scores[0][i]
            )


            if index < 0:

                continue


            if score < similarity_threshold:

                continue


            relevant_chunks.append(
                pdf_chunks[index]
            )


            source_names.append(
                pdf_sources[index]
            )


            source_pages.append(
                pdf_pages[index]
            )


            valid_scores.append(
                score
            )


        # -------------------------------------------------
        # ANSWER NOT FOUND
        # -------------------------------------------------

        if not relevant_chunks:


            if language == "ta-IN":

                not_found = (
                    "மன்னிக்கவும், பதிவேற்றிய PDF-களில் "
                    "இந்த கேள்விக்கான பதிலை "
                    "கண்டுபிடிக்க முடியவில்லை."
                )

            else:

                not_found = (
                    "Sorry, I could not find "
                    "the answer in the uploaded PDFs."
                )


            return jsonify({

                "success": True,

                "question":
                    question,

                "answer":
                    not_found,

                "sources":
                    [],

                "similarity":
                    0

            })


        # =================================================
        # CREATE CONTEXT
        # =================================================

        context_parts = []


        for i in range(
            len(relevant_chunks)
        ):


            context_parts.append(

                f"Source: {source_names[i]}\n"

                f"Page: {source_pages[i]}\n"

                f"Content: {relevant_chunks[i]}"

            )


        context = (
            "\n\n".join(
                context_parts
            )
        )


        # =================================================
        # LANGUAGE INSTRUCTION
        # =================================================

        if language == "ta-IN":


            language_instruction = """

Answer ONLY in natural Tamil.

The answer must be a clear Tamil
sentence or short paragraph.

Do not translate the question
into English.

Do not include English unless
an English technical term is necessary.

"""


        else:


            language_instruction = """

Answer ONLY in clear English.

The answer must be a clear English
sentence or short paragraph.

"""


        # =================================================
        # GEMINI PROMPT
        # =================================================

        prompt = f"""

You are DocuMind AI,
an intelligent PDF question answering assistant.

Use ONLY the information provided
in the PDF content below.

PDF CONTENT:

{context}


USER QUESTION:

{question}


RESPONSE LANGUAGE:

{language_instruction}


STRICT RULES:

1. Give ONLY the final answer.

2. Do NOT repeat the user's question.

3. Do NOT mention FAISS.

4. Do NOT mention Gemini.

5. Do NOT mention retrieval.

6. Do NOT mention embeddings.

7. Do NOT mention internal processing.

8. Do NOT mention the PDF context.

9. Do NOT invent information.

10. Answer only from the relevant
PDF content.

11. Keep the answer simple and natural.

12. Return a sentence or short paragraph.

13. Do NOT return unrelated questions
or answers.

14. Do NOT give multiple unrelated
answers.

15. If the answer is not available
in the supplied PDF content, use:

Tamil:
"மன்னிக்கவும், பதிவேற்றிய PDF-களில்
இந்த கேள்விக்கான பதிலை கண்டுபிடிக்க முடியவில்லை."

English:
"Sorry, I could not find the answer
in the uploaded PDFs."

"""


        # =================================================
        # GEMINI GENERATION
        # =================================================

        response = client.models.generate_content(

            model=GENERATION_MODEL,

            contents=prompt

        )


        answer = (
            response.text
            or ""
        ).strip()


        # =================================================
        # CREATE SOURCE DETAILS
        # =================================================

        source_details = []


        for i in range(
            len(source_names)
        ):


            detail = {

                "file":
                    source_names[i],

                "page":
                    source_pages[i]

            }


            if detail not in source_details:

                source_details.append(
                    detail
                )


        # =================================================
        # BEST SIMILARITY
        # =================================================

        best_similarity = (
            max(valid_scores)
            if valid_scores
            else 0
        )


        # =================================================
        # RETURN ANSWER
        # =================================================

        return jsonify({

            "success": True,

            "question":
                question,

            "answer":
                answer,

            "sources":
                source_details,

            "similarity":
                round(
                    best_similarity,
                    3
                )

        })


    except Exception as e:


        return jsonify({

            "success": False,

            "message":
                f"Server error: {str(e)}"

        })


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )