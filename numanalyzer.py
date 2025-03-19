import select
import sys
import tkinter as tk
from tkinter import filedialog

import numpy as np
import torch
import torchvision
from PIL import Image
from sty import fg

from net import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = Net().to(device)
network.load_state_dict(torch.load('mnist_model_Conv.pth', weights_only=True))
network.eval()

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,), (1,))
])

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
            prediction = torch.argmax(output, dim=1).item()
        confidence = 100 * output[0, prediction].to("cpu").item()
        if confidence >= 90:
            print(f'Prediction: {prediction}')
            print(fg.green + f'Confidence: {confidence:.3f}%' + fg.rs)
        elif confidence < 90 and confidence > 50: 
            print(f'Prediction: {prediction}' + fg.rs)
            print(fg.yellow  + f'Confidence: {confidence:.3f}%' + fg.rs)
        elif confidence < 50:
            print(f'Prediction: {prediction}')
            print(fg.red + f'Confidence: {confidence:.3f}%' + fg.rs)
    else:
        print("No file selected.\nExiting...")
        exit()

predict_image()

def input_with_timeout(prompt, timeout):
    print(prompt, end=': ', flush=True)
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.readline().strip().lower()
    else:
        return 'n'

while True:
    again = input_with_timeout("Do you want to analyze another image? (y/n)", 10)
    if again == 'y':
        predict_image()
    elif again == 'n':
        print("\nExiting...")
        break
    else:
        print("Invalid input. Please enter 'y' or 'n'.")