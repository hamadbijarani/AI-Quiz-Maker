# 🧠 AI Quiz Maker

AI-powered quiz generator that creates interactive quizzes from **PDF documents** using **LangChain, Google Generative AI, and Streamlit**. Perfect for teachers, trainers, and students who want to turn learning materials into quizzes instantly.

---

## 🚀 Features

* 📄 **Upload PDFs** → Extracts text and processes it.
* 🤖 **AI-Powered** → Generates quiz questions with multiple-choice answers.
* 📝 **Interactive Quiz** → Take the quiz inside the Streamlit app.
* ✅ **Instant Feedback** → Know if your answer is correct on submission.
* 💾 **FAISS Vector Store** → Efficient semantic search for context retrieval.

---

## 🛠️ Tech Stack

* [Python 3.10+](https://www.python.org/)
* [Streamlit](https://streamlit.io/) – Web app framework
* [LangChain](https://www.langchain.com/) – Prompt orchestration
* [Google Generative AI](https://ai.google.dev/) – LLM backend
* [FAISS](https://github.com/facebookresearch/faiss) – Vector search
* [PyPDF2](https://pypi.org/project/PyPDF2/) – PDF parsing
* [dotenv](https://pypi.org/project/python-dotenv/) – Environment variable management

---

## 📂 Project Structure

```
AI-Quiz-Maker/
│── Pics
  │── Interface 1.png
  │── Interface 2.png
│── main.py              # Streamlit app entry point
│── requirements.txt     # Dependencies
│── LICENSE              # Apache License 2.0
│── README.md            # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repo**

```bash
git clone https://github.com/hamadbijarani/AI-Quiz-Maker.git
cd AI-Quiz-Maker
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
   Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_api_key_here
```

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run main.py
```

1. Upload a **PDF document**.
2. The AI generates **quiz questions**.
3. Start the quiz and test your knowledge interactively!

---

## 📸 Demo (Optional)

1. 
!["Home Screen"](https://github.com/hamadbijarani/AI-Quiz-Maker/blob/71b64c28d0a24b9baead634693eb4bdc6217b393/Pics/Interface1.png)

2.
!["During A Quiz"](https://github.com/hamadbijarani/AI-Quiz-Maker/blob/71b64c28d0a24b9baead634693eb4bdc6217b393/Pics/Interface%202.png)

---

## 🤝 Contributing

Contributions are welcome!

* Fork the repo
* Create a new branch (`feature/your-feature`)
* Commit changes
* Open a pull request

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use and modify.

