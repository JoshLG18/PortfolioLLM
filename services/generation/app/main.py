# ------------------------------------------------------------------------------------------------------
# Script Name: main.py
# Description: This script defines the FastAPI application for the generation service.
# It includes endpoints for health check, generating answers, and checking the daily quota.
# ------------------------------------------------------------------------------------------------------
# Configuration

from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta, timezone

from .quota import remaining_quota, increment_quota, DAILY_QUOTA
from .gemini_client import generate as gemini_generate, GEMINI_MODEL_NAME
from .prompts import build_answer_prompt

app = FastAPI(title="generation-service")

# ------------------------------------------------------------------------------------------------------
# Endpoints

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(payload: dict):
    
    if remaining_quota() <= 0: # check if the remaining quota is less than or equal to 0, indicating that the daily limit has been reached
        raise HTTPException(
            status_code=429,
            detail={"error": "quota_exceeded", "remaining": 0, "limit": DAILY_QUOTA},
        )

    prompt = build_answer_prompt(payload["question"], payload["sources"]) # build the prompt using the question and sources from the payload
    answer = gemini_generate(prompt, payload.get("temperature", 0.3)) # generate the answer using the gemini_generate function with the prompt and optional temperature from the payload
    increment_quota() # increment the quota count after generating the answer

    return { # return a dictionary containing the generated answer, the model name, and the current quota information
        "answer": answer,
        "model": GEMINI_MODEL_NAME,
        "quota": {"remaining": remaining_quota(), "limit": DAILY_QUOTA},
    }


@app.get("/quota")
def quota():
    tomorrow = datetime.now(timezone.utc).date() + timedelta(days=1) # get tomorrow's date in UTC
    resets_at = datetime.combine(tomorrow, datetime.min.time(), tzinfo=timezone.utc).isoformat() # get the reset time in ISO format (YYYY-MM-DDTHH:MM:SS+00:00) for tomorrow at 00:00 UTC
    
    return { # return the remaining quota, limit, and reset time in a dictionary
        "remaining": remaining_quota(),
        "limit": DAILY_QUOTA,
        "resets_at": resets_at,
    }
