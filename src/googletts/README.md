# FastAPI & Gradio Voice Chatbot

This project implements a modular, production-ready voice chatbot with FastAPI backend and Gradio frontend. It uses Google Cloud TTS for speech synthesis (all languages, including Hebrew), and a Nebius LLM for generating chatbot responses in English (with optional Google LLM). An optional F5-TTS engine can be enabled for English-only synthesis. The system automatically handles language switching based on user selection.

## Setup

1. **Clone the repository** and navigate into it:

   ```bash
   git clone <repo-url>
   cd <repo-folder>
