from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from cnn_model import ComplexCNN

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load your trained PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ComplexCNN(num_classes=3).to(device)
state_dict_path = os.path.join(os.getcwd(),'state_dict.pth')
# state_dict_path = r"C:\work_of_atri\de 5th sem project\de_sem_5\model.pth"
state_dict = torch.load(state_dict_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Image preprocessing function
def prepare_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Adjust size based on your model's input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to match training
    ])
    image = Image.open(image_path).convert('RGB')  # Ensure 3 channels (RGB)
    return transform(image).unsqueeze(0)  # Add batch dimension

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Prepare image for prediction
        image = prepare_image(file_path).to(device)

        # Predict the class
        with torch.no_grad():
            outputs = model(image)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        classes = ['Unripe', 'Ripe', 'Rotten']  # Replace with your model's class names
        result = classes[predicted_class]
        
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
