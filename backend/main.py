from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from torchvision.models import vgg16
import torch
import torch.nn as nn
from PIL import Image
import io

# Initialize FastAPI
app = FastAPI()

# Load the trained model
MODEL_PATH = "backend/vggnet.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained VGG16 model with modified classifier
model = vgg16(weights=None)  # No default pretrained weights
model.classifier = nn.Sequential(
    nn.Linear(in_features=25088, out_features=512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(in_features=512, out_features=256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(in_features=256, out_features=62)  # Output layer for 62 classes
)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),                # Resize to VGG16 input size
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize for VGG16
])

def preprocess_image(image_bytes):
    """
    Convert uploaded image bytes to a tensor.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    return image

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

# Run the API with: uvicorn main:app --host 0.0.0.0 --port 8000
