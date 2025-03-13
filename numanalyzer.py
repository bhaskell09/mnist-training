import tkinter as tk
from tkinter import filedialog

import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from net import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = Net().to(device)
network.load_state_dict(torch.load('mnist_model_Swish.pth', weights_only=True))
network.eval()

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image = np.array(image, dtype=np.float32)
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)
    image = image / 255.0
    return image.to(device)

def predict_image():
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename()
    if image_path:
        image_tensor = preprocess_image(image_path)
        with torch.no_grad():
            output = network(image_tensor)
            predicted_label = torch.argmax(output, dim=1).item()
        print(f'Predicted label: {predicted_label}')
    else:
        print("No file selected")

predict_image()

while True:
    again = input("Do you want to analyze another image? (y/n): ").strip().lower()
    if again == 'y':
        predict_image()
    elif again == 'n':
        print("Exiting...")
        break
    else:
        print("Invalid input. Please enter 'y' or 'n'.")