import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import mss
import time
import torchvision.models as models
import matplotlib.pyplot as plt 

device = 'cuda'
# Define the transformations
mean = [0.485, 0.456, 0.406]  # Replace with your model's mean if different
std = [0.229, 0.224, 0.225]  # Replace with your model's std if different

transform1 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)])

transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

# Load the model
model_path = "liminal_space_classifier.pth"  # Replace with the path to your .pth file
model = torch.load(model_path)
model.eval()
# Function to capture the screen
def capture_screen():
    with mss.mss() as sct:
        # Capture the entire screen
        screen = sct.grab(sct.monitors[0])
        # Convert to a PIL image
        img = Image.frombytes("RGB", screen.size, screen.rgb).convert('RGB')
    return img

# Function to predict whether an image contains a liminal space
def predict(image):
    # Apply the transformations
    input_tensor = transform2(image).reshape(1,3,224,224).to(device)  # Add batch dimension
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
    # Assuming the model outputs probabilities, adjust thresholding logic as needed
    predicted = torch.argmax(output)
    return "Liminal Space" if predicted.item() == 0 else "Not Liminal"

def visualize(image, result):
    plt.clf()  # Clear the current figure
    plt.imshow(image)
    plt.title(f"Prediction: {result}")
    plt.axis('off')
    plt.pause(0.01)  # Pause to update the plot

# Main loop to continuously check the screen
if __name__ == "__main__":
    print("Starting screen detection... Press Ctrl+C to stop.")
    plt.figure(figsize=(8, 6))  # Set the figure size
    try:
        while True:
            screen_image = capture_screen()
            imageseen = transform1(screen_image)
            result = predict(imageseen)
            visualize(imageseen, result)
            time.sleep(1)  # Adjust the delay as needed
    except KeyboardInterrupt:
        print("Stopping detection.")
        plt.close()
