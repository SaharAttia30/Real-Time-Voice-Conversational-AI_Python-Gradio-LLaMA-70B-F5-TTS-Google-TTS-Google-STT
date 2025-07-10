# services/llm_service.py

import openai
from app.config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def get_response(text, history):
    # LLM integration here
    return "LLM response"