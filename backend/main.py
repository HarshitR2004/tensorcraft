from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
import torch
import torch.nn as nn
from PIL import Image
import io

app = FastAPI(title="EMNIST Classification API", description="A FastAPI-based image classification API using VGG16 for EMNIST ByClass dataset.")

# Load pretrained VGG16 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize VGG16 with pretrained weights
weights = VGG16_Weights.IMAGENET1K_V1
model = vgg16(weights=weights)  # Load the model with pretrained weights

# Freeze all the layers so gradients are not calculated during training.
# This is very important for transfer learning and speed
for param in model.parameters():
    param.requires_grad = False

# Modify the classifier for EMNIST (62 classes)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features=num_features, out_features=62)

# Load fine-tuned model weights (if available)
MODEL_PATH = "backend/vggnet.pth"
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # strict=True by default
    print("Loaded fine-tuned model weights.")
except Exception as e:
    print("Using default pretrained VGG16 weights. Error:", e)

model.to(device)
model.eval()

# Define preprocessing for VGG16 (ImageNet normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to VGG16 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

def preprocess_image(image_bytes):
    """
    Convert uploaded image bytes to a tensor.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Convert to RGB
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    return image

@app.get("/", response_class=HTMLResponse)
async def welcome_page():
    """
    Welcome page with API instructions.
    """
    return """
    <html>
        <head>
            <title>EMNIST Classification API</title>
        </head>
        <body style="font-family: Arial, sans-serif; text-align: center;">
            <h1>Welcome to the EMNIST Classification API</h1>
            <p>This API classifies handwritten characters from the <b>EMNIST ByClass</b> dataset using a pretrained VGG16 model.</p>
            <h3>How to Use:</h3>
            <ul style="display: inline-block; text-align: left;">
                <li>Send a POST request to <code>/predict/</code> with an image file.</li>
                <li>Supported formats: <b>PNG, JPG, JPEG</b>.</li>
                <li>The API returns the predicted class (0-61).</li>
            </ul>
            <h3>Example Request:</h3>
            <pre>curl -X 'POST' -F 'file=@image.png' 'http://127.0.0.1:8000/predict/'</pre>
            <p><b>Happy Coding! 🚀</b></p>
        </body>
    </html>
    """

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # Read image file and preprocess it
    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    return {"prediction": int(predicted.item())}