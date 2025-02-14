import os
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import JSONResponse
from PIL import Image, ImageOps
from dotenv import load_dotenv
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables (replace with your actual API key management)
API_KEY = "AIzaSyDGKeZb9uB2gjJ9ubFnnYHVvAeoR00xNMc"  # Replace with your actual API key
# load_dotenv() # Not needed anymore since the API KEY is directly defined

genai.configure(api_key=API_KEY)

# Model Configuration
generation_config = {
    "temperature": 0.4,  # Lower temperature for more accuracy
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 10,  # Limit output tokens significantly, adjust as needed
}

# Choose the appropriate model.  Crucially, change this to a supported model.
#  `gemini-pro-vision` is ALSO deprecated.  Use `gemini-1.5-flash` or `gemini-1.5-pro`.
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",  #  <----  **CHANGED THIS LINE**  Use 'gemini-1.5-flash' or 'gemini-1.5-pro'
    generation_config=generation_config
)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to EMNIST prediction API"}


def preprocess_image(image: Image.Image) -> Image.Image:
    # Resize to a standard size
    image = image.resize((256, 256))

    # Convert to grayscale
    image = image.convert("L")  # 'L' mode is grayscale

    # Invert the image
    image = ImageOps.invert(image)
    # Threshold to create a binary image (black and white)
    threshold = 128  # Adjust this value as needed
    image = ImageOps.grayscale(image)

    return image


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        preprocessed_image = preprocess_image(image)  # Preprocess the image

        # Refined prompt to Gemini, specify the expected output
        prompt = "Identify the single digit or alphabet character in the image. Return only the single identified character (digit or letter). If the image does not contain a digit or letter, return a question mark '?'. Do not include any explanations or additional text."

        try:
             # Pass the PIL Image object DIRECTLY to the model.
            response = model.generate_content([prompt, preprocessed_image])
            response.resolve()
            result = response.text
        except Exception as gemini_error:
            logging.error(f"Gemini API error: {gemini_error}")
            raise HTTPException(status_code=500, detail=f"Gemini API error: {gemini_error}")

        if not result:
            logging.warning("Empty response from Gemini API.")
            result = "?"  # Default if nothing is returned.

        result = result.strip()

        logging.info(f"Prediction result: {result}")
        return JSONResponse(content={"result": result})


    except Exception as e:
        logging.exception("Error processing image.")
        raise HTTPException(status_code=500, detail=str(e))