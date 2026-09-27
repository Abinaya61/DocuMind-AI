from flask import Flask, render_template, request, jsonify
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from google import genai

import faiss
import os


app = Flask(__name__)


# =====================================================
# 1. GEMINI SETUP
# =====================================================

api_key = os.environ.get("GEMINI_API_KEY")

client = genai.Client(
    api_key=api_key
)


# =====================================================
# 2. SENTENCE TRANSFORMER
# =====================================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =====================================================
# 3. UPLOAD FOLDER
# =====================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =====================================================
# 4. GLOBAL VARIABLES
# =====================================================

pdf_chunks = []

pdf_sources = []

pdf_pages = []

pdf_embeddings = None

faiss_index = None


# =====================================================
# 5. HOME
# =====================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =====================================================
# 6. MULTIPLE PDF UPLOAD
# =====================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_pdfs():

    global pdf_chunks
    global pdf_sources
    global pdf_pages
    global pdf_embeddings
    global faiss_index


    # ---------------------------------------------
    # Get uploaded files
    # ---------------------------------------------

    files = request.files.getlist("pdf")


    if not files:

        return jsonify({

            "success": False,

            "message":
                "Please select at least one PDF"

        })


    # ---------------------------------------------
    # Reset previous data
    # ---------------------------------------------

    pdf_chunks = []

    pdf_sources = []

    pdf_pages = []

    pdf_embeddings = None

    faiss_index = None

    all_text = ""

    total_pages = 0

    uploaded_files = []


    try:

        # =========================================
        # PROCESS EACH PDF
        # =========================================

        for file in files:


            if file.filename == "":
                continue


            # -------------------------------------
            # Check PDF
            # -------------------------------------

            if not file.filename.lower().endswith(".pdf"):

                continue


            # -------------------------------------
            # Save PDF
            # -------------------------------------

            filepath = os.path.join(

                app.config["UPLOAD_FOLDER"],

                file.filename

            )

            file.save(filepath)


            # -------------------------------------
            # Read PDF
            # -------------------------------------

            reader = PdfReader(
                filepath
            )


            total_pages += len(
                reader.pages
            )


            uploaded_files.append(
                file.filename
            )


            pdf_text = ""


            # =====================================
            # PROCESS EACH PAGE
            # =====================================

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):


                # ---------------------------------
                # Extract page text
                # ---------------------------------

                page_text = page.extract_text()


                if not page_text:

                    continue


                # ---------------------------------
                # Store complete text
                # ---------------------------------

                pdf_text += (
                    page_text +
                    "\n"
                )


                # =================================
                # CHUNK THIS PAGE
                # =================================

                text_splitter = RecursiveCharacterTextSplitter(

                    chunk_size=500,

                    chunk_overlap=50

                )


                page_chunks = text_splitter.split_text(
                    page_text
                )


                # =================================
                # STORE CHUNKS + SOURCE + PAGE
                # =================================

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


            # -------------------------------------
            # Add PDF text
            # -------------------------------------

            all_text += (
                pdf_text +
                "\n"
            )


        # =========================================
        # CHECK CONTENT
        # =========================================

        if not pdf_chunks:

            return jsonify({

                "success": False,

                "message":
                    "Could not extract text from the PDFs"

            })


        # =========================================
        # CREATE EMBEDDINGS
        # =========================================

        pdf_embeddings = model.encode(

            pdf_chunks,

            convert_to_numpy=True

        )


        # =========================================
        # FLOAT32
        # =========================================

        pdf_embeddings = pdf_embeddings.astype(
            "float32"
        )


        # =========================================
        # NORMALIZE
        # =========================================

        faiss.normalize_L2(
            pdf_embeddings
        )


        # =========================================
        # CREATE FAISS INDEX
        # =========================================

        dimension = pdf_embeddings.shape[1]


        faiss_index = faiss.IndexFlatIP(
            dimension
        )


        # =========================================
        # ADD EMBEDDINGS
        # =========================================

        faiss_index.add(
            pdf_embeddings
        )


        # =========================================
        # SUCCESS RESPONSE
        # =========================================

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
                "Multiple PDFs uploaded successfully"

        })


    except Exception as e:

        return jsonify({

            "success": False,

            "message":
                f"Error processing PDFs: {str(e)}"

        })


# =====================================================
# 7. ASK QUESTION
# =====================================================

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

        # =========================================
        # GET REQUEST
        # =========================================

        data = request.get_json()


        if not data:

            return jsonify({

                "success": False,

                "message":
                    "Invalid request"

            })


        question = data.get(
            "question",
            ""
        ).strip()


        # =========================================
        # CHECK QUESTION
        # =========================================

        if not question:

            return jsonify({

                "success": False,

                "message":
                    "Please enter a question"

            })


        # =========================================
        # CHECK PDF
        # =========================================

        if not pdf_chunks or faiss_index is None:

            return jsonify({

                "success": False,

                "message":
                    "Please upload PDF files first"

            })


        # =========================================
        # QUESTION EMBEDDING
        # =========================================

        question_embedding = model.encode(

            [question],

            convert_to_numpy=True

        ).astype("float32")


        # =========================================
        # NORMALIZE
        # =========================================

        faiss.normalize_L2(
            question_embedding
        )


        # =========================================
        # FAISS SEARCH
        # =========================================

        k = min(

            5,

            len(pdf_chunks)

        )


        scores, indices = faiss_index.search(

            question_embedding,

            k

        )


        # =========================================
        # SIMILARITY THRESHOLD
        # =========================================

        SIMILARITY_THRESHOLD = 0.35


        # =========================================
        # RELEVANT DATA
        # =========================================

        relevant_chunks = []

        source_names = []

        source_pages = []


        # =========================================
        # FILTER SEARCH RESULTS
        # =========================================

        for i in range(k):

            index = indices[0][i]

            score = float(
                scores[0][i]
            )


            # -------------------------------------
            # Invalid index
            # -------------------------------------

            if index < 0:

                continue


            # -------------------------------------
            # Ignore low similarity
            # -------------------------------------

            if score < SIMILARITY_THRESHOLD:

                continue


            # -------------------------------------
            # Store relevant chunk
            # -------------------------------------

            relevant_chunks.append(
                pdf_chunks[index]
            )


            source_names.append(
                pdf_sources[index]
            )


            source_pages.append(
                pdf_pages[index]
            )


        # =========================================
        # NO RELEVANT RESULT
        # =========================================

        if not relevant_chunks:

            return jsonify({

                "success": True,

                "question":
                    question,

                "answer":
                    "Sorry, I could not find the answer in the uploaded PDFs.",

                "sources": [],

                "pages": [],

                "similarity": 0

            })


        # =========================================
        # CREATE CONTEXT
        # =========================================

        context_parts = []


        for i in range(
            len(relevant_chunks)
        ):

            context_parts.append(

                f"Source: {source_names[i]}\n"
                f"Page: {source_pages[i]}\n"
                f"Content: {relevant_chunks[i]}"

            )


        context = "\n\n".join(
            context_parts
        )


        # =========================================
        # GEMINI PROMPT
        # =========================================

        prompt = f"""
You are DocuMind AI, an intelligent PDF question answering assistant.

Use ONLY the information provided in the PDF content below.

PDF CONTENT:

{context}

USER QUESTION:

{question}

RULES:

1. Give only the final answer.
2. Do not repeat the question.
3. Do not mention FAISS.
4. Do not mention Gemini.
5. Do not mention the PDF context.
6. Do not invent information.
7. Keep the answer clear and simple.
8. Answer only using relevant information.
9. If the answer is not available in the provided content, say:

"Sorry, I could not find the answer in the uploaded PDFs."
"""


        # =========================================
        # GEMINI
        # =========================================

        response = client.models.generate_content(

            model="gemini-3.8-flash",

            contents=prompt

        )


        # =========================================
        # ANSWER
        # =========================================

        answer = response.text.strip()


        # =========================================
        # UNIQUE SOURCE + PAGE
        # =========================================

        source_details = []


        for i in range(
            len(source_names)
        ):

            source = source_names[i]

            page = source_pages[i]


            detail = {

                "file":
                    source,

                "page":
                    page

            }


            if detail not in source_details:

                source_details.append(
                    detail
                )


        # =========================================
        # BEST SIMILARITY
        # =========================================

        valid_scores = []


        for i in range(k):

            index = indices[0][i]

            score = float(
                scores[0][i]
            )


            if index >= 0 and score >= SIMILARITY_THRESHOLD:

                valid_scores.append(
                    score
                )


        best_similarity = max(
            valid_scores
        ) if valid_scores else 0


        # =========================================
        # RETURN RESULT
        # =========================================

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


# =====================================================
# 8. RUN FLASK
# =====================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )