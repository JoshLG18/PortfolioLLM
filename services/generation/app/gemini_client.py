# ------------------------------------------------------------------------------------------------------
# Script Name: gemini_client.py
# Description: This script provides a client for interacting with the Gemini generative model.
# It includes functions to get a cached instance of the model and generate content based on a prompt
# ------------------------------------------------------------------------------------------------------
# get_gemini_client() - Returns a cached instance of the Gemini generative model.
# generate(prompt: str, temperature: float = 0.3) - Generates content based on the provided prompt and temperature using the Gemini model.
# ------------------------------------------------------------------------------------------------------

# Configuration
import os
import google.generativeai as genai

GEMINI_MODEL_NAME = os.environ.get("GENERATION_MODEL", "gemma-4-26b-a4b-it")

_gemini = None

# ------------------------------------------------------------------------------------------------------
# Functions

def get_gemini_client():
    global _gemini # declare _gemini as a global variable to access it within the function

    if _gemini is None:
        api_key = os.environ.get("GEMINI_API_KEY") # fetch the api key from environment variable
        genai.configure(api_key=api_key) # configure the genai library with the api key
        _gemini = genai.GenerativeModel(GEMINI_MODEL_NAME) # create a new instance of the GenerativeModel with the specified model name

    return _gemini # return the cached instance of the model

def generate(prompt: str, temperature: float = 0.3) -> str:
    model = get_gemini_client() # get the cached instance of the model
    result = model.generate_content( # get the model to generate content based on the provided prompt and temperature
        prompt,
        generation_config={"temperature": temperature, "top_p": 0.95},
    )

    return (result.text or "").strip() # return the generated text, stripping any leading or trailing whitespace