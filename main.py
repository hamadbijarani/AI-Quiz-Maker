import os
import json
import shutil
import random
import asyncio
import streamlit as st
from pydantic import SecretStr
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_float import float_init, float_css_helper
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# 1. Loading environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-1.5-flash"


@st.cache_data(show_spinner=False)
def get_pdf_text(pdf_files):
    """Extracts text from a list of uploaded PDF files."""
    full_text = ""
    for pdf_file in pdf_files:
        pdf = PdfReader(pdf_file)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text
    return full_text

@st.cache_data(show_spinner=False)
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=2000)
    return splitter.split_text(text)

@st.cache_resource
def get_embeddings():
    """Returns a cached instance of the Google Generative AI Embeddings."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=SecretStr(GOOGLE_API_KEY))

@st.cache_resource
def get_llm():
    """Returns a cached instance of the ChatGoogleGenerativeAI model."""
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, google_api_key=SecretStr(GOOGLE_API_KEY))

def create_and_save_vector_store(text_chunks):
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)

    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)

@st.cache_resource
def load_vector_store():
    """Loads the FAISS vector store from the local path."""
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

def create_question(question_data: dict, idx: int):
    score_increment, question_increment, quiz_end = 0, 0, False
    question_container = st.container(border=True)
    with question_container:
        st.markdown(f"### **Question {idx+1}:** {question_data['question']}")
        user_answer = st.radio(
            "Select your answer:",
            question_data['options'],
            index=None,
            key=f"answer_{idx}"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit Answer", key=f"submit_{idx}", disabled=(user_answer is None)):
                st.session_state.options_chosen.append(user_answer)
                question_increment = 1

                if user_answer == question_data['correct_option']:
                    score_increment = 1
                    st.success("Correct!", icon="✅")
                else:
                    st.error(f"Incorrect! The correct answer is: **{question_data['correct_option']}**", icon="❌")
        with col2:
            if st.button("End Quiz", key=f"end_{idx}"):
                quiz_end = True
                score_increment = -1
                question_increment = -1

    return score_increment, question_increment, quiz_end

def generate_quiz_from_faiss(vector_db, num_questions: int = 5):
    if vector_db is None:
        st.error("Vector database not found. Please upload a PDF first.")
        return []
    
    retrieval_queries = [
        "Generate quiz questions about the key concepts and main ideas in the document.",
        "Create a quiz based on the important details and factual information presented.",
        "Formulate questions that test understanding of the document's primary topics.",
        "What are some potential multiple-choice questions from this text?",
        "Generate a quiz that covers the essential information from the document."
    ]

    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 20})
    random_query = random.choice(retrieval_queries)
    retrieved_docs = retriever.get_relevant_documents(random_query)

    if len(retrieved_docs) > 15:
        docs_for_context = random.sample(retrieved_docs, 15)
    else:
        docs_for_context = retrieved_docs

    context = " ".join([doc.page_content for doc in docs_for_context])

    quiz_prompt = PromptTemplate(
        input_variables=["context", "num_questions"],
        template="""
        You are an expert quiz generator. Using the provided context, generate {num_questions} multiple-choice questions.
        Introduce variety and randomness in the phrasing of your questions to ensure they are not repetitive.

        Each question must have exactly 4 answer options, and you must clearly specify which option is correct.

         IMPORTANT:
        - Vary the style, structure, and order of your questions so they are not repetitive.
        - Randomize phrasing and avoid predictable patterns.
        - Each question must have exactly 4 answer options.
        - Clearly specify the correct option.
        - Return ONLY valid JSON (no markdown, no extra text).

        Example output format:
        [
          {{
            "question": "Sample question?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_option": "Option A"
          }}
        ]

        Context:
        {context}
        """
    )

    llm = get_llm()
    chain = quiz_prompt | llm
    response = chain.invoke({"context": context, "num_questions": num_questions})

    try:
        cleaned_response = response.content.strip().replace("```json", "").replace("```", "")
        quiz_data = json.loads(cleaned_response)

        if isinstance(quiz_data, list):
            random.shuffle(quiz_data)
            for question in quiz_data:
                if 'options' in question and isinstance(question['options'], list):
                    random.shuffle(question['options'])
            return quiz_data
        else:
            return []
            
    except Exception as e:
        st.error(f"Failed to create the quiz from the model's response. Please try again. Error: {e}")
        return []


def process_pdfs(pdf_docs, alerts_placeholder):
    """Coordinates the processing of uploaded PDF files."""
    alerts_placeholder.info("Processing uploaded documents...")
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    create_and_save_vector_store(text_chunks)
    st.session_state.pdf_uploaded = True
    
    st.cache_resource.clear()
    alerts_placeholder.success("Processing complete. You can now start the quiz!")


def main():
    st.set_page_config(page_title="AI Quiz Maker", page_icon=":book:", layout="centered")
    st.title("AI Quiz Maker")
    float_init()

    st.session_state.setdefault("quiz_ongoing", False)
    st.session_state.setdefault("score", 0)
    st.session_state.setdefault("question_number", 0)
    st.session_state.setdefault("question_bank", [])
    st.session_state.setdefault("pdf_uploaded", False)
    st.session_state.setdefault("review_mode", False)
    st.session_state.setdefault("options_chosen", [])
    st.session_state.setdefault('last_file_count', 0)
    st.session_state.setdefault('total_questions', 5)
    st.session_state.setdefault('ballons_showed', False)
    st.session_state.setdefault("quiz_ended", True)
    

    with st.sidebar:
        st.header("Upload your PDF")
        pdf_docs = st.file_uploader(
            "Upload one or more PDFs to generate a quiz", 
            accept_multiple_files=True, 
            type=["pdf"], 
            key="pdf_uploader"
        )

        if len(pdf_docs) != st.session_state.last_file_count:
            st.session_state.last_file_count = len(pdf_docs)
            if pdf_docs:
                with st.spinner("Processing PDFs... This may take a moment."):
                    process_pdfs(pdf_docs, st.sidebar)
                st.rerun()
            else:
                if os.path.exists(FAISS_INDEX_PATH):
                    shutil.rmtree(FAISS_INDEX_PATH)
                st.session_state.pdf_uploaded = False
                st.cache_resource.clear()
                st.rerun()

        if st.button("Reset Quiz"):
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")
            st.session_state.quiz_ongoing = False
            st.rerun()



    if not st.session_state.quiz_ongoing:
        if not st.session_state.pdf_uploaded:
            st.info("Please upload and process a PDF file to start the quiz.")
        else: 
            st.success("Document processed! 🎉 Ready to start the quiz.")
            st.radio("How many questions would you like in the quiz?", [5, 10, 20], key="total_questions", horizontal=True)

            if st.button("Begin Quiz"):
                with st.spinner("Generating your quiz..."):
                    vector_db = load_vector_store()
                    st.session_state.question_bank = generate_quiz_from_faiss(
                        vector_db,
                        num_questions=st.session_state.total_questions
                    )
                if st.session_state.question_bank:
                    st.session_state.quiz_ongoing = True
                    st.session_state.score = 0
                    st.session_state.question_number = 0
                    st.session_state.options_chosen = []
                    st.session_state.review_mode = False
                    st.session_state.ballons_showed = False
                    st.session_state.quiz_ended = False
                    st.rerun()
                else:
                    st.error("Could not generate a quiz. Please check your document or try again.")
    else:
        current_q_num = st.session_state.question_number
        total_questions = len(st.session_state.question_bank)

        if current_q_num < total_questions and not st.session_state.quiz_ended:
            question_data = st.session_state.question_bank[current_q_num]
            o1, o2, o3 = create_question(question_data, current_q_num)
            if o2 > 0:
                st.session_state.score += o1
                st.session_state.question_number += o2
                st.rerun()
            if o3:
                st.session_state.quiz_ended = True
                st.session_state.question_number -= 1
                st.rerun()
        else:
            if st.session_state.ballons_showed == False:
                if st.session_state.score > 0:
                    st.balloons()
                st.session_state.ballons_showed = True
        
            st.markdown(f"## Quiz Complete! Your Score: {st.session_state.score} / {st.session_state.question_bank.__len__()}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("End Quiz", use_container_width=True):
                    st.session_state.quiz_ongoing = False
                    st.rerun()
            with col2:
                if st.button("Review Answers", use_container_width=True):
                    st.session_state.review_mode = True
                    st.rerun()
            
            if st.session_state.review_mode:
                css = float_css_helper(justify="center", align="center", width="40%", height="50%", overflow_y="scroll")
                review_container = st.container(height=400, border=True)
                with review_container:
                    st.button("Close Review", on_click=lambda: st.session_state.update({"review_mode": False}))
                    st.markdown("### 🧐 Review Your Answers")
                    for idx, q_data in enumerate(st.session_state.question_bank):
                        with st.container(border=True):
                            st.markdown(f"**Question {idx+1}:** {q_data['question']}")
                            user_choice = st.session_state.options_chosen[idx] if idx < len(st.session_state.options_chosen) else None
                            correct_choice = q_data['correct_option']

                            if user_choice == correct_choice:
                                st.success(f"Your answer: {user_choice} (Correct ✅)")
                            else:
                                st.error(f"Your answer: {user_choice} (Incorrect ❌)")
                                st.info(f"Correct answer: {correct_choice}")
    
    
if __name__=="__main__":
    main()
