import asyncio
from typing import Dict, List, Tuple
import httpx
import re
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers.util import cos_sim
from .utils import generate_response, embedder

def _parse_score(score_str: str) -> int:
    """
    Parse score from various formats:
    - "10" -> 10
    - "10/10" -> 10
    - "Score: 10" -> 10
    Returns int between 1-10, defaults to 5 if unparseable
    """
    score_str = score_str.strip()
    
    # Try to extract just the number part
    match = re.search(r'(\d+)', score_str)
    if match:
        score = int(match.group(1))
        # Clamp to 1-10 range
        return max(1, min(10, score))
    
    # Default to 5 if can't parse
    return 5

async def evaluate_relevance_completeness(response: str, query: str, model_type: str = "openai", model_name: str = None) -> Tuple[int, int, str]:
    """LLM-as-judge for relevance and completeness."""
    prompt = ChatPromptTemplate.from_template(
        "Rate the response's relevance to the query (1-10, 10=perfect). "
        "Also rate completeness (does it fully address the query?). "
        "Query: {query}. Response: {response}. "
        "Output format: Relevance: X\nCompleteness: Y\nExplanation: Z"
    ).format(query=query, response=response)
    
    judge_response, _ = await generate_response(prompt, model_type=model_type, model_name=model_name)
    lines = judge_response.split("\n")
    relevance = _parse_score(lines[0].split(": ")[1] if ": " in lines[0] else lines[0])
    completeness = _parse_score(lines[1].split(": ")[1] if ": " in lines[1] else lines[1])
    explanation = lines[2].split(": ")[1] if len(lines) > 2 and ": " in lines[2] else "No explanation provided"
    return relevance, completeness, explanation

async def evaluate_hallucination(response: str, contexts: List[str], model_type: str = "openai", model_name: str = None) -> Tuple[int, List[str], str]:
    """Hallucination check: LLM grounding + embedding similarity."""
    # Embedding similarity as initial filter
    response_emb = embedder.encode(response)
    context_embs = embedder.encode(contexts)
    similarities = [cos_sim(response_emb, emb).item() for emb in context_embs]
    avg_sim = sum(similarities) / len(similarities) if similarities else 0
    
    # LLM judge for detailed check
    prompt = ChatPromptTemplate.from_template(
        "Verify if the response is fully supported by the context. "
        "List any unsupported claims (hallucinations). Rate accuracy 1-10. "
        "Context: {context}. Response: {response}. "
        "Output: Accuracy: X\nHallucinations: [list]\nExplanation: Z"
    ).format(context="\n".join(contexts), response=response)
    
    judge_response, _ = await generate_response(prompt, model_type=model_type, model_name=model_name)
    lines = judge_response.split("\n")
    
    accuracy = _parse_score(lines[0].split(": ")[1] if ": " in lines[0] else lines[0])
    
    # Try to parse hallucinations list
    hallucinations = []
    if len(lines) > 1:
        try:
            hal_str = lines[1].split(": ")[1] if ": " in lines[1] else lines[1]
            # Try to evaluate as Python list
            hallucinations = eval(hal_str)
            if not isinstance(hallucinations, list):
                hallucinations = [hal_str]
        except:
            # If can't parse as list, just use the raw string
            hallucinations = [lines[1]]
    
    explanation = lines[2].split(": ")[1] if len(lines) > 2 and ": " in lines[2] else "No explanation provided"
    
    # Adjust score if similarity low
    if avg_sim < 0.7:
        accuracy = max(1, accuracy - 2)
        hallucinations.append("Low semantic match to context")
    
    return accuracy, hallucinations, explanation