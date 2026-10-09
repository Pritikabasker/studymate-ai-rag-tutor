
import streamlit as st
import fitz
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 StudyMate AI")
st.caption("Upload notes • Learn topics • Clear doubts • Take quizzes")

# ---------------- GEMINI ----------------
try:
    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )
except Exception:
    st.error("API key problem. Check .streamlit/secrets.toml")
    st.stop()

MODEL = "gemini-3.8-flash"

# ---------------- SESSION STATE ----------------
defaults = {
    "pages": [],
    "chunks": [],
    "topics": [],
    "selected_topic": "",
    "lesson": "",
    "doubts": [],
    "quiz": [],
    "quiz_topic": "",
    "quiz_submitted": False,
    "quiz_score": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------- HELPERS ----------------
def ask_ai(prompt):
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    return response.text or ""


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def make_chunks(pages):
    chunks = []

    for page_no, text in enumerate(pages, start=1):
        words = text.split()
        size = 180
        overlap = 30

        start = 0
        while start < len(words):
            part = words[start:start + size]
            if part:
                chunks.append({
                    "text": " ".join(part),
                    "page": page_no
                })
            start += size - overlap

    return chunks


def get_context(query, limit=5):
    chunks = st.session_state.chunks

    if not chunks:
        return ""

    docs = [c["text"] for c in chunks]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english"
        )
        matrix = vectorizer.fit_transform(docs)
        q_vector = vectorizer.transform([query])
        scores = cosine_similarity(
            q_vector, matrix
        ).flatten()

        best = scores.argsort()[-limit:][::-1]
        results = [
            chunks[i] for i in best if scores[i] > 0
        ]

    except ValueError:
        results = []

    return "\n\n".join(
        f"[Page {c['page']}]\n{c['text']}"
        for c in results
    )


def reset_topic():
    st.session_state.lesson = ""
    st.session_state.doubts = []
    st.session_state.quiz = []
    st.session_state.quiz_topic = ""
    st.session_state.quiz_submitted = False
    st.session_state.quiz_score = None


# ---------------- SIDEBAR: UPLOAD ----------------
with st.sidebar:
    st.header("📄 Your Study Material")

    uploaded = st.file_uploader(
        "Upload your notes (PDF)",
        type=["pdf"]
    )

    if uploaded and st.button(
        "📖 Read PDF and Find Topics",
        type="primary",
        use_container_width=True
    ):
        with st.spinner("Reading your PDF..."):
            try:
                pdf = fitz.open(
                    stream=uploaded.getvalue(),
                    filetype="pdf"
                )

                pages = [
                    page.get_text()
                    for page in pdf
                ]

                st.session_state.pages = pages
                st.session_state.chunks = make_chunks(pages)
                st.session_state.topics = []
                st.session_state.selected_topic = ""
                reset_topic()

            except Exception as e:
                st.error(f"Could not read PDF: {e}")

    if st.session_state.pages:
        st.success(
            f"PDF loaded: {len(st.session_state.pages)} pages"
        )

        if st.button("🔄 Classify Topics", use_container_width=True):
            all_text = "\n".join(
                f"Page {i+1}: {text}"
                for i, text in enumerate(
                    st.session_state.pages
                )
            )

            all_text = all_text[:35000]

            prompt = f"""
You are an expert study material organizer.

Read the following PDF text and identify its main
study topics and subtopics.

Return ONLY valid JSON in this format:
{{
  "topics": [
    {{
      "name": "Topic name",
      "description": "One sentence about the topic",
      "keywords": ["word1", "word2"],
      "pages": [1, 2]
    }}
  ]
}}

Rules:
- Identify around 4 to 15 useful topics if possible.
- Do not invent topics not present in the notes.
- Use the page numbers shown in the text.
- Keep topic names short and clear.
- Include subtopics when they are meaningful.
- If page numbers are unclear, use an empty list.

PDF TEXT:
{all_text}
"""

            with st.spinner("AI is organizing your topics..."):
                try:
                    result = ask_ai(prompt)
                    data = parse_json(result)
                    topics = data.get("topics", [])

                    st.session_state.topics = topics

                    if topics:
                        st.success(
                            f"Found {len(topics)} topics!"
                        )
                    else:
                        st.warning("No topics found.")

                except Exception as e:
                    st.error(
                        "Topic classification failed. "
                        "Try again. Details: " + str(e)
                    )


# ---------------- MAIN ----------------
if not st.session_state.pages:
    st.info(
        "👈 Upload your PDF in the sidebar to begin."
    )
    st.markdown("""
    ### Your learning journey

    1. Upload your class notes.
    2. Let AI organize the topics.
    3. Select one topic and learn it.
    4. Ask questions whenever you have doubts.
    5. Take a quiz after studying.
    """)
    st.stop()


if not st.session_state.topics:
    st.warning(
        "Your PDF is loaded. Click **Classify Topics** "
        "in the sidebar to organize your notes."
    )
    st.stop()


# ---------------- TOPIC SELECTION ----------------
st.subheader("🗂️ Topics found in your PDF")

topic_names = [
    t.get("name", "Untitled topic")
    for t in st.session_state.topics
]

selected = st.selectbox(
    "Choose what you want to learn",
    topic_names,
    key="topic_dropdown"
)

if selected != st.session_state.selected_topic:
    st.session_state.selected_topic = selected
    reset_topic()

topic_data = next(
    (
        t for t in st.session_state.topics
        if t.get("name") == selected
    ),
    {}
)

description = topic_data.get("description", "")
pages = topic_data.get("pages", [])

st.markdown(f"### 📘 {selected}")
if description:
    st.write(description)

if pages:
    st.caption(
        "Topic appears around PDF page(s): "
        + ", ".join(map(str, pages))
    )


# ---------------- TOPIC CONTEXT ----------------
keywords = topic_data.get("keywords", [])
search_query = selected + " " + " ".join(keywords)
context = get_context(search_query, limit=7)

if not context:
    context = "\n\n".join(
        f"[Page {i+1}]\n{text}"
        for i, text in enumerate(st.session_state.pages)
        if text.strip()
    )[:14000]


# ---------------- THREE LEARNING SECTIONS ----------------
learn_tab, doubt_tab, quiz_tab = st.tabs([
    "👩‍🏫 Teaching",
    "💬 Doubt Clearance",
    "📝 Quiz"
])


# ---------------- TEACHING ----------------
with learn_tab:
    st.subheader("👩‍🏫 Learn this topic")

    st.write(
        "AI will explain the selected topic using "
        "your uploaded notes."
    )

    if st.button(
        "✨ Teach me this topic",
        type="primary",
        key="teach_button"
    ):
        prompt = f"""
You are a friendly college tutor.

Teach the student the topic: {selected}

Use ONLY the supplied study material.
If something is missing, say that it is not covered
in the uploaded notes.

Teach in this order:
1. Simple introduction
2. Explanation in beginner-friendly language
3. Important concepts, step by step
4. A simple example or analogy
5. Important points to remember
6. A short recap
7. Ask the student one practice question

Use clear headings and simple language.
Mention PDF page numbers when possible.
Do not pretend to know information absent from the notes.

STUDY MATERIAL:
{context}
"""

        with st.spinner("Preparing your lesson..."):
            try:
                st.session_state.lesson = ask_ai(prompt)
            except Exception as e:
                st.error(
                    "Could not prepare lesson: " + str(e)
                )

    if st.session_state.lesson:
        st.markdown(st.session_state.lesson)


# ---------------- DOUBT CLEARANCE ----------------
with doubt_tab:
    st.subheader("💬 Ask your doubt")

    st.caption(
        f"Questions about: {selected}"
    )

    for msg in st.session_state.doubts:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    doubt = st.chat_input(
        "Type your question here...",
        key="doubt_input"
    )

    if doubt:
        st.session_state.doubts.append({
            "role": "user",
            "content": doubt
        })

        with st.chat_message("user"):
            st.markdown(doubt)

        prompt = f"""
You are a patient tutor helping a student understand:
{selected}

Answer the student's doubt using the study material.
Explain step by step in simple language.
Give an example if useful.
If the PDF does not contain enough information,
say so clearly rather than inventing an answer.
Mention relevant page numbers when possible.

STUDY MATERIAL:
{get_context(selected + " " + doubt, limit=5)}

STUDENT'S DOUBT:
{doubt}
"""

        with st.chat_message("assistant"):
            with st.spinner("Clearing your doubt..."):
                try:
                    answer = ask_ai(prompt)
                    st.markdown(answer)
                except Exception as e:
                    answer = (
                        "AI is temporarily unavailable. "
                        "Please try again. " + str(e)
                    )
                    st.error(answer)

        st.session_state.doubts.append({
            "role": "assistant",
            "content": answer
        })


# ---------------- QUIZ ----------------
with quiz_tab:
    st.subheader("📝 Test what you learned")

    st.write(
        "Take this quiz after studying the topic."
    )

    if st.button(
        "🎯 Generate 5-question quiz",
        type="primary",
        key="quiz_button"
    ):
        prompt = f"""
Create exactly 5 multiple-choice questions
about the topic: {selected}

Use ONLY the study material below.
Questions should test understanding, not just memory.
Each question must have 4 options.
Exactly one option must be correct.

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "question": "Question text",
      "options": [
        "Option A",
        "Option B",
        "Option C",
        "Option D"
      ],
      "answer": 0,
      "explanation": "Why this answer is correct"
    }}
  ]
}}

Rules:
- answer is the zero-based index (0, 1, 2, or 3).
- Keep the questions clear and fair.
- Do not use facts absent from the material.

STUDY MATERIAL:
{context}
"""

        with st.spinner("Creating your quiz..."):
            try:
                result = ask_ai(prompt)
                data = parse_json(result)
                questions = data.get("questions", [])

                if not questions:
                    st.error("No questions were generated.")
                else:
                    st.session_state.quiz = questions
                    st.session_state.quiz_topic = selected
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score = None

            except Exception as e:
                st.error(
                    "Quiz generation failed: " + str(e)
                )

    if st.session_state.quiz:
        if st.session_state.quiz_topic != selected:
            st.info(
                "Generate a new quiz for this topic."
            )
        elif not st.session_state.quiz_submitted:
            with st.form("quiz_form"):
                answers = []

                for i, q in enumerate(
                    st.session_state.quiz
                ):
                    st.markdown(
                        f"**Q{i+1}. {q['question']}**"
                    )

                    options = q.get("options", [])

                    choice = st.radio(
                        "Choose one:",
                        options,
                        key=f"quiz_{i}",
                        index=None
                    )

                    answers.append(choice)
                    st.divider()

                submitted = st.form_submit_button(
                    "Submit my answers",
                    type="primary"
                )

            if submitted:
                score = 0
                user_answers = []

                for i, q in enumerate(
                    st.session_state.quiz
                ):
                    options = q.get("options", [])
                    chosen = answers[i]
                    correct_index = q.get("answer", 0)

                    correct_option = (
                        options[correct_index]
                        if isinstance(correct_index, int)
                        and 0 <= correct_index < len(options)
                        else ""
                    )

                    if chosen == correct_option:
                        score += 1

                    user_answers.append({
                        "chosen": chosen,
                        "correct": correct_option
                    })

                st.session_state.quiz_score = score
                st.session_state.quiz_answers = user_answers
                st.session_state.quiz_submitted = True
                st.rerun()

        else:
            score = st.session_state.quiz_score
            total = len(st.session_state.quiz)

            st.success(
                f"Your score: {score}/{total}"
            )

            if total and score == total:
                st.balloons()
                st.write("Excellent! You got all answers right.")
            elif total and score >= total * 0.6:
                st.write("Good work! Review the explanations below.")
            else:
                st.write(
                    "Keep practising! Read the lesson again "
                    "and retry the quiz."
                )

            for i, q in enumerate(
                st.session_state.quiz
            ):
                user_answer = (
                    st.session_state.quiz_answers[i]
                )

                st.markdown(
                    f"**Q{i+1}. {q['question']}**"
                )

                if user_answer["chosen"] == user_answer["correct"]:
                    st.success(
                        "Correct: " + str(user_answer["correct"])
                    )
                else:
                    st.error(
                        "Your answer: "
                        + str(user_answer["chosen"])
                    )
                    st.write(
                        "Correct answer: "
                        + str(user_answer["correct"])
                    )

                st.caption(
                    "Explanation: "
                    + q.get("explanation", "")
                )
                st.divider()

            if st.button("🔁 Retake quiz"):
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score = None
                st.rerun()