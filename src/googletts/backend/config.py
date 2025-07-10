import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_APPLICATION_CREDENTIALS = ""
os.environ["OPENAI_API_KEY"] = ""
# Nebius API configuration
NEBIUS_API_KEY = ""
NEBIUS_API_URL = "https://api.studio.nebius.com/v1/"

# Flags to switch models
USE_GOOGLE_LLM = os.getenv("USE_GOOGLE_LLM", "false").lower() == "true"
USE_F5_TTS = os.getenv("USE_F5_TTS", "true").lower() == "true"
