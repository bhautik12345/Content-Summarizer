# 🦜 LangChain Content Summarizer

This is a Streamlit-based web application that summarizes content from YouTube videos or web pages using **LangChain** and **LLM models** (like LLaMA via Groq API). It also generates a quiz based on the summarized content.

## 🚀 Features

- 🔗 Accepts both YouTube and general website URLs.
- 🧠 Uses `LangChain` and `ChatGroq` to process and summarize text.
- ✂️ Intelligent chunking of large documents.
- 📝 Provides a structured summary with numbered key points.
- ❓ Auto-generates quiz questions and answers (min. 5).
- 🛡️ API key protected (Groq).

## 🛠️ Technologies

- Python
- Streamlit
- LangChain
- ChatGroq (LLaMA model)
- YoutubeLoader & UnstructuredURLLoader

## 📦 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/bhautik12345/Content-Summarizer.git
   cd Content-Summarizer
   ```

