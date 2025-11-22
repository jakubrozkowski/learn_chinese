# Learn Chinese - AI Language Learning Tool

An interactive Chinese learning application that helps with translation, text improvement, pronunciation, and tracks learning history using semantic search.

**Live Demo:** [https://learn-chinese.streamlit.app/](https://learn-chinese.streamlit.app/)

## Features

- **Translation (EN/PL/ES→CN):** Translate text in your language to natural Chinese with grammar explanations
- **Text Improvement:** Polish your Chinese writing with AI corrections and explanations  
- **Text-to-Speech:** Listen to correct pronunciation
- **Semantic History Search:** Find past translations using vector embeddings
- **Multi-language UI:** English, Polish, Spanish, Chinese interface

## Tech Stack

- **Python** + **Streamlit** for web interface
- **OpenAI API:** GPT-4o-mini (translation/improvement), Whisper TTS, Embeddings
- **QDrant** Vector Database for semantic search
- **Deployed** on Streamlit Cloud

## Why I Built This

As a China Studies student learning Mandarin, I wanted a personalized tool combining AI with practical language learning. This project demonstrates my ability to:
- Build full-stack AI applications
- Integrate multiple AI models (LLM, TTS, Embeddings)
- Implement vector databases for semantic search
- Deploy production-ready applications

## Setup

1. Clone the repository
```bash
git clone https://github.com/jakubrozkowski/learn_chinese.git
cd learn_chinese
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables
```bash
# Create .env file with:
OPENAI_API_KEY=your_key_here
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
```

4. Run locally
```bash
streamlit run app.py
```

## What I Learned

- Managing complex state in Streamlit applications
- Integrating multiple OpenAI models in a single app
- Implementing vector search for language learning
- Building multilingual user interfaces
- Deploying AI applications to production

## Connect

Built by Jakub Rozkowski | [LinkedIn](https://www.linkedin.com/in/jakub-r%C3%B3%C5%BCkowski-934102346/) | [GitHub](https://github.com/jakubrozkowski)

Part of my journey learning AI/ML while studying Chinese at Jagiellonian University.

---

**⭐ If you find this helpful, give it a star! ⭐**
