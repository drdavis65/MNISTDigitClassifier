from flask import Flask, request, jsonify
import torch
from mnist_classifier import SimpleCNN
from torchvision.transforms import transforms
from PIL import Image
import io

app = Flask(__name__)

# Load your model
model = SimpleCNN()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Updated transformation pipeline
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.Grayscale(num_output_channels=1),  # Ensure it's grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/')
def index():
    return "Welcome to the digit classification API! Use /predict to POST an image."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400

    file = request.files['file']
    # Open the uploaded image and convert it to grayscale in one step
    image = Image.open(io.BytesIO(file.read())).convert('L')

    # Preprocess the image and predict
    image_tensor = transform(image)
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0))  # add batch dimension
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
