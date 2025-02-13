import os
import io
from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Set it in environment variables or .env file.")

genai.configure(api_key=API_KEY)

# Model Configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    system_instruction="Identify the digit and alphabet which is being uploaded in the image. Just show the result without any extra text. If any other input image is given, return a random digit or letter.",
)

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to EMNIST prediction API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Send image to Gemini API
        response = model.generate_content([image])

        return JSONResponse(content={"result": response.text})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


