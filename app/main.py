from fastapi import FastAPI, HTTPException
from .models import InputData, EvaluationReport, ContextWithScore
from .utils import parse_jsons_from_objects, build_prompt, generate_response, embedder
from .evaluators import evaluate_relevance_completeness, evaluate_hallucination
import asyncio
import traceback
import os
from .config import OPENAI_API_KEY, GOOGLE_API_KEY

app = FastAPI(title="BeyondChats LLM Evaluation Pipeline")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/evaluate", response_model=EvaluationReport)
async def evaluate_llm(input_data: InputData):
    try:
        # Determine which API keys to use: UI keys take priority, fall back to .env
        openai_key = input_data.openai_api_key or OPENAI_API_KEY
        google_key = input_data.google_api_key or GOOGLE_API_KEY
        
        # Set API keys in environment for LangChain to use
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        if google_key:
            os.environ["GOOGLE_API_KEY"] = google_key
        
        # Validate that at least one API key is available for the selected model
        if input_data.model_type == "openai" and not openai_key:
            raise ValueError("OpenAI API key not found. Please provide it in .env file or through UI.")
        elif input_data.model_type == "gemini" and not google_key:
            raise ValueError("Google API key not found. Please provide it in .env file or through UI.")
        
        query, history, contexts, context_objects = parse_jsons_from_objects(input_data.conversation, input_data.context_vectors)
        if not query:
            raise ValueError("No user query found in conversation JSON")
        
        # Compute similarity scores for retrieved context
        query_emb = embedder.encode(query)
        retrieved_context = []
        for ctx_obj in context_objects:
            ctx_text = ctx_obj.get('text', '')
            ctx_emb = embedder.encode(ctx_text)
            similarity = float((query_emb @ ctx_emb.T).item())  # Cosine similarity
            # Normalize to 0-1 range (cosine similarity is -1 to 1, but typically 0 to 1)
            similarity = max(0, min(1, similarity))
            
            retrieved_context.append(ContextWithScore(
                text=ctx_text,
                source_url=ctx_obj.get('source_url'),
                similarity_score=similarity
            ))
        
        # Sort by similarity descending and keep only top 3
        retrieved_context.sort(key=lambda x: x.similarity_score, reverse=True)
        retrieved_context = retrieved_context[:3]  # Limit to top 3
        
        # Extract just the text for building the prompt (contexts is still a list of strings)
        top_context_texts = [ctx.text for ctx in retrieved_context]
        
        prompt = build_prompt(query, history, top_context_texts)
        
        # Use custom model_name if provided, otherwise use default from config
        model_name = input_data.model_name
        generated_response, metrics = await generate_response(prompt, model_type=input_data.model_type, model_name=model_name)
        
        # Parallel evaluations
        (relevance, completeness, rel_exp), (accuracy, hallucinations, acc_exp) = await asyncio.gather(
            evaluate_relevance_completeness(generated_response, query),
            evaluate_hallucination(generated_response, top_context_texts)
        )
        
        return EvaluationReport(
            generated_response=generated_response,
            relevance_score=relevance,
            completeness_score=completeness,
            accuracy_score=accuracy,
            hallucinations=hallucinations,
            latency_ms=metrics['latency'],
            cost_usd=metrics['cost'],
            retrieved_context=retrieved_context,
            prompt_used=prompt,
            explanations={"relevance_completeness": rel_exp, "accuracy_hallucination": acc_exp}
        )
    
    except Exception as e:
        print(f"Error in /evaluate endpoint: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))