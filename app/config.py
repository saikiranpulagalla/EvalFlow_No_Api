import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
INPUT_TOKEN_PRICE = float(os.getenv("INPUT_TOKEN_PRICE", 0.00015)) / 1000  # Per token
OUTPUT_TOKEN_PRICE = float(os.getenv("OUTPUT_TOKEN_PRICE", 0.0006)) / 1000  # Per token
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")