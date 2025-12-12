import json
import os
import time
from typing import Dict, List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer, util
from .config import OPENAI_API_KEY, GOOGLE_API_KEY, MODEL_NAME, INPUT_TOKEN_PRICE, OUTPUT_TOKEN_PRICE, EMBEDDING_MODEL
from .json_cleaner import clean_json

# Load embedding model
embedder = SentenceTransformer(EMBEDDING_MODEL)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_response(prompt: str, model_type: str = "openai", model_name: str = None) -> Tuple[str, Dict]:
    """Generate LLM response with fallback."""
    try:
        # Use provided model name or fall back to default from config
        if model_name is None:
            model_name = MODEL_NAME
        
        # Read API keys from environment (allows Streamlit UI to override)
        openai_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
        google_key = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY)
        
        if model_type == "openai":
            llm = ChatOpenAI(model=model_name, api_key=openai_key)
        else:
            llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=google_key)
        
        start_time = time.perf_counter()
        response = await llm.ainvoke(prompt)
        latency = (time.perf_counter() - start_time) * 1000  # ms
        
        # Token usage from response metadata (LangChain standard)
        usage = response.response_metadata.get('token_usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        cost = (prompt_tokens * INPUT_TOKEN_PRICE) + (completion_tokens * OUTPUT_TOKEN_PRICE)
        
        return response.content, {"latency": latency, "cost": cost, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    
    except Exception as e:
        if model_type == "openai":
            return await generate_response(prompt, model_type="gemini")  # Fallback
        raise e

def parse_jsons(conversation_path: str, context_path: str) -> Tuple[str, str, List[str]]:
    """Parse JSONs from file paths and extract query, history, contexts."""
    with open(conversation_path, 'r') as f:
        conv = json.loads(clean_json(f.read()))
    with open(context_path, 'r') as f:
        ctx = json.loads(clean_json(f.read()))
    
    return _extract_from_jsons(conv, ctx)

def parse_jsons_from_objects(conversation: Dict, context_vectors) -> Tuple[str, str, List[str], List[Dict]]:
    """Parse JSON objects (already parsed) and extract query, history, contexts, and context objects."""
    # Handle both cases: direct array or wrapped in response structure
    if isinstance(context_vectors, dict) and 'data' in context_vectors:
        # It's wrapped in response structure (from API response)
        ctx_obj = context_vectors
    elif isinstance(context_vectors, list):
        # It's a direct array
        ctx_obj = {'data': {'vector_data': context_vectors}}
    else:
        # Assume it's a dict with vector_data directly
        ctx_obj = {'data': {'vector_data': context_vectors.get('vector_data', []) if isinstance(context_vectors, dict) else []}}
    
    return _extract_from_jsons(conversation, ctx_obj)

def _extract_from_jsons(conv: Dict, ctx: Dict) -> Tuple[str, str, List[str], List[Dict]]:
    """Helper function to extract query, history, contexts, and context objects from already-parsed JSON objects."""
    # Assume last user message is the specific one (as per samples)
    messages = conv.get('conversation_turns', []) or conv.get('messages', [])
    user_messages = [msg for msg in messages if msg.get('role') == 'User' or msg.get('sender_id') != 1]
    query = user_messages[-1]['message'] if user_messages else ""
    
    # History: all prior messages
    history = "\n".join([f"{msg['role']}: {msg['message']}" for msg in messages[:-1]])
    
    # Contexts: concatenate relevant texts and keep context objects
    vectors = ctx.get('data', {}).get('vector_data', []) or ctx.get('contexts', [])
    contexts = [v['text'] for v in vectors if 'text' in v]
    context_objects = [v for v in vectors if 'text' in v]
    
    return query, history, contexts, context_objects

def build_prompt(query: str, history: str, contexts: List[str]) -> str:
    """Build generation prompt."""
    template = ChatPromptTemplate.from_template(
        "You are a helpful AI nurse for Malpani Infertility Clinic. Answer based ONLY on the provided context.\n"
        "Context: {context}\n"
        "History: {history}\n"
        "User Query: {query}\n"
        "Response:"
    )
    return template.format(context="\n".join(contexts), history=history, query=query)