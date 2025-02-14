import os
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import JSONResponse
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Set it in environment variables or .env file.")

genai.configure(api_key=API_KEY)

# Model Configuration
generation_config = {
    "temperature": 0.4,  # Lower temperature for more accuracy
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 10,  # Limit output tokens significantly
}

# Assuming gemini-pro-vision is suitable, check the documentation for the accurate model to use.
model = genai.GenerativeModel(
    model_name="gemini-pro-vision",  # Replace with the correct model name
    generation_config=generation_config,
    system_instruction="Identify the digit or alphabet in the image.  Return only the single identified character (digit or letter). If the image is not a digit or letter, return a question mark '?'.",  # Refined prompt
)

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to EMNIST prediction API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)) #.convert("L")  # Keep color, experiment

        # Convert image to Base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Construct the Gemini API request payload (check Gemini API documentation for correct structure)
        contents = [
            {
                "mime_type": "image/png",
                "data": base64_image
            }
        ]

        try:
            response = model.generate_content(contents)
            response.resolve()  # Force eager evaluation and raise exceptions if needed
            result = response.text
        except Exception as gemini_error:
            logging.error(f"Gemini API error: {gemini_error}")
            raise HTTPException(status_code=500, detail=f"Gemini API error: {gemini_error}") #Improved Error Handling
        
        if not result:
            logging.warning("Empty response from Gemini API.")
            result = "?"  # Default if nothing is returned.
        
        logging.info(f"Prediction result: {result}") # Add logging

        return JSONResponse(content={"result": result})

    except Exception as e:
        logging.exception("Error processing image.")
        raise HTTPException(status_code=500, detail=str(e)) # Improved Error Handling