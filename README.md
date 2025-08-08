# INTELLIRAG - RAG-Based Cognitive Q&A AI Chatbot

INTELLIRAG is a multi-modal, persistent, Retrieval-Augmented Generation (RAG) chatbot built with Gemini and LangChain, featuring secure Firebase authentication and a Streamlit UI for interactive research and analysis.

---

## Features

- **Knowledge Ingestion:** PDFs, text, images (Gemini vision), audio (Whisper), tabular data
- **Secure Authentication:** Firebase Auth
- **Session Persistence:** User chat history and knowledge base
- **Advanced Q&A:** Combines FAISS semantic and BM25 lexical search with Gemini
- **Autonomous Data Science Agent:** Pandas-powered analytics
- **Automated Reports:** Research planning and summary synthesis
- **Animated Streamlit Interface**

---

## Setup Procedure

### 1. **Clone the Repository**


### 2. **Python Environment**

- Use Python **3.10, 3.11, or 3.12** for full compatibility.


### 3. **Install Dependencies**


#### Install FFmpeg on your system (required for Whisper audio transcription):

- **Windows:**  
  Use [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/)  

### 4. **Environment Variables & Secrets**

Create a file called `.env` in your repo root with the following (no quotes, matching your API/key info):


Download your **Firebase service account JSON** from Google Firebase Console and place it in your repo root.  
**Do NOT commit this file to Git.**  
Add the following to `.gitignore`:


### 5. **Run the App**

Visit [http://localhost:8501](http://localhost:8501).

---

## Usage

1. **Sign Up/Login** using Firebase Auth.
2. **Dashboard:** Access or create your knowledge base.
3. **Upload:** Add PDFs, text, images, audio, or paste text.
4. **Chat:** Ask questions interactively and view document sources.
5. **Autonomous Analysis:** Enter a research topic to get multi-step research plans and full report synthesis.

---

## Security Note

- Never commit `.env` or your Firebase key file.
- Rotate and remove secrets if accidentally pushed (see GitHub’s [secret-scanning docs](https://docs.github.com/code-security/secret-scanning/working-with-secret-scanning-and-push-protection/working-with-push-protection-from-the-command-line#resolving-a-blocked-push)).

---

## Troubleshooting

- **tokenizers/build errors:** Use Python 3.10–3.12 or install Rust/Cargo.
- **Firebase errors:** Check `.env` and key file paths.
- **Whisper/FFmpeg errors:** Confirm ffmpeg is installed and on PATH.

---

## License

MIT

---

## Contributing

1. Fork the repo
2. Create a feature branch
3. Commit changes
4. Push and create a Pull Request

