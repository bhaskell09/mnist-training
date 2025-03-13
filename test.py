import torch
import torchvision
from tqdm import tqdm

from net import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = Net().to(device)
network.load_state_dict(torch.load('model_weights.pth', weights_only=True))
network.eval()
