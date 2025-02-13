import os
import io
from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse
from PIL import Image, ImageOps
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
API_KEY = "AIzaSyA3KSVWjkRo7Je1oCnLy8LHMony-paUyrk"


genai.configure(api_key=API_KEY)

# Model Configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

model = genai.GenerativeModel(model_name="gemini-2.0-flash", generation_config=generation_config)

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
    image = image.point(lambda x: 0 if x < threshold else 255, '1') #'1' is bilevel image

    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Preprocess the image
        processed_image = preprocess_image(image)

        #Ask it to show the letter with high confidence and give it a few example
        prompt = "Identify the letter in this image. I want you to identify the letter and only provide that letter. Nothing else. For examples: Image with 'A', you response 'A'. Image with 'B', you response 'B'. etc"
        # Send the image and the prompt to Gemini
        response = model.generate_content([prompt, processed_image])

        return JSONResponse(content={"result": response.text})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

