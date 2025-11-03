# Modified MosaiQ: single-circuit generator for 20x20 images

import os
from PIL import Image
import math
import logging
import numpy as np
from scipy.linalg import sqrtm
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import pennylane as qml
import argparse

# Configure a simple logger so debug statements are consistent and optional
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(' ')

parser = argparse.ArgumentParser()
parser.add_argument('--mode', dest='mode', type=str, default='train')
parser.add_argument('--ds', dest='dataset', type=str, default='MNIST')
parser.add_argument('--ds_class', dest='ds_class', type=int, default=5)
# parser.add_argument('--env', dest='env', type=str, default='Sim')
parser.add_argument('--num_iter', dest='num_iter', type=int, default=1000)
# New: allow adjustable image size (default 20)
parser.add_argument('--image_size', dest='image_size', type=int, default=20)
parser.add_argument('--log-level', dest='log_level', type=str, default='INFO',
                    help='Logging level: DEBUG, INFO, WARNING, ERROR')
parser.add_argument('--explain-args', dest='explain_args', action='store_true',
                    help='Print detailed explanations for each CLI argument and exit')
args = parser.parse_args()

numeric_level = getattr(logging, args.log_level.upper(), None)
if not isinstance(numeric_level, int):
    logger.warning(f"Invalid log level '{args.log_level}', defaulting to INFO")
    numeric_level = logging.INFO
logger.setLevel(numeric_level)

# If requested, print a helpful explanation for each CLI argument and exit
if args.explain_args:
        help_text = f"""
Usage explanation for mosaiq.py arguments:

    --mode: Operation mode for the script. 'train' runs GAN training. 'test' runs saved-model evaluation.
    --tr1: Label/name of the first trained generator model to use in test mode (e.g. '5' corresponds to 'generator_5').
    --tr2: Label/name of the second trained generator model to use in test mode.
    --ds: Dataset to use. Supported values: 'MNIST' (default) or 'Fashion'.
    --ds_class: Integer label of the digit/class to train on (default: 5). Only images of this class are used.
    # --env: Execution environment for the quantum circuit. 'simulation' (default) uses PennyLane lightning.qubit. Use 'Real' to swap in a function stub for a QPU backend.
    --num-iter: Number of training iterations to run when in 'train' mode (default: 20).
    --image_size: Size of the square images to process (default: 20). Images will be resized to image_size x image_size.
    --log-level: Logging verbosity. One of DEBUG, INFO (default), WARNING, ERROR.
    --explain-args: Shows this help text and exits.

Notes:
    - When running with --mode train the script will save models as 'generator_<ds_class>' and 'disc_<ds_class>'.
    - When running with --mode test, supply --tr1 and --tr2 with the labels of saved models to compare them.
    - The script expects MNIST data to be available or will download it to './mnist' by default.
"""
        print(help_text)
        import sys
        sys.exit(0)

# Function to scale numpy data to [min,max]
def scale_data(data, scale=None, dtype=np.float32):
    if scale is None:
        scale = [-1, 1]
    min_data, max_data = float(np.min(data)), float(np.max(data))
    min_scale, max_scale = float(scale[0]), float(scale[1])
    data = ((max_scale - min_scale) * (data - min_data) / (max_data - min_data)) + min_scale
    return data.astype(dtype)

# Load and preprocess dataset, flattening to image_size x image_size
if args.dataset == 'MNIST':
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(torch.flatten)
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist', download=True, train=True, transform=transform),
        batch_size=10000, shuffle=True
    )
elif args.dataset == 'Fashion':
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(torch.flatten)
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./fashion', download=True, train=True, transform=transform),
        batch_size=10000, shuffle=True
    )
# Collect only the selected class
train_data = []
selected_class = args.ds_class
max_samples = 100  # limit dataset size for faster testing
for (data, labels) in train_loader:
    for x, y in zip(data, labels):
        if int(y) == selected_class:
            train_data.append(x.numpy())
            if len(train_data) >= max_samples:
                break
    if len(train_data) >= max_samples:
        break

train_data = scale_data(np.array(train_data), [0, 1])  # scale pixels to [0,1]

# Set model dimensions
image_size = args.image_size        # e.g. 20
orig_dims = image_size * image_size  # total pixels (400)
batch_size = 8
pca_dims = image_size   # compress 400->20 features
n_qubits = pca_dims     # number of qubits equals number of PCA features
q_depth = 6

# Compute PCA on training images (each flattened vector has length orig_dims)
from sklearn.decomposition import PCA
pca = PCA(n_components=pca_dims)
pca_data_full = pca.fit_transform(train_data)  # shape: (num_samples, pca_dims)

# DataLoader for PCA-transformed data (features scaled to [-1,1] for training)
scaled_pca_data = scale_data(pca_data_full, [0, 1])
dataloader = torch.utils.data.DataLoader(
    scaled_pca_data, batch_size=batch_size, shuffle=True, drop_last=True
)

# Discriminator (unchanged, input dimension = pca_dims)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(pca_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# Set up PennyLane device with n_qubits wires
dev = qml.device("lightning.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_circuit(noise, weights):
    # weights shape: (q_depth * n_qubits,) flattened before reshape
    weights = weights.reshape(q_depth, n_qubits)
    # Encode noise into single-qubit rotations
    for i in range(n_qubits):
        qml.RY(noise[i], wires=i)
        qml.RX(noise[i], wires=i)
    # Variational layers with CZ entanglement
    for layer in range(q_depth):
        for w in range(n_qubits):
            qml.RY(weights[layer][w], wires=w)
        for w in range(n_qubits - 1):
            qml.CZ(wires=[w, w+1])
    # Return expectation of PauliX on each qubit (one feature per qubit)
    return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

def get_noise_upper_bound(errG, errD, original_ratio, current_noise,
                          min_noise=1e-4, max_noise=math.pi/2,
                          smoothing=0.9, aggressiveness=0.5):
    """
    Adaptively adjust the upper bound of the noise amplitude used to sample
    rotation angles for the quantum generator.

    Parameters
    ----------
    errG : torch.Tensor or float
        Current generator loss (scalar).
    errD : torch.Tensor or float
        Current discriminator loss (scalar).
    original_ratio : float
        Reference ratio errD/errG measured at an early point in training.
        If None, function returns `current_noise` unchanged (safe fallback).
    current_noise : float
        Current value of the noise upper bound (used for smoothing).
    min_noise : float
        Minimum allowed upper bound (prevents collapse to zero).
    max_noise : float
        Maximum allowed upper bound for safety (keeps angles reasonable).
    smoothing : float in (0,1)
        Exponential smoothing factor: result = smoothing*current + (1-smoothing)*proposed.
        Use large smoothing to avoid oscillations.
    aggressiveness : float >=0
        How strongly the proposed change reacts to ratio shifts. Higher -> bigger changes.

    Returns
    -------
    float
        New noise upper bound (clamped between min_noise and max_noise).
    """
    # defensively extract scalars if torch tensors were passed
    try:
        g = float(errG.detach().cpu().item())
        d = float(errD.detach().cpu().item())
    except Exception:
        g = float(errG)
        d = float(errD)

    # avoid division by zero
    eps = 1e-9
    if original_ratio is None or original_ratio <= 0 or g <= 0:
        # no reliable reference yet — keep current value
        return float(np.clip(current_noise, min_noise, max_noise))

    # compute current ratio: how discriminator compares to generator now
    ratio = (d + eps) / (g + eps)

    # relative change compared to original_ratio
    rel_change = (ratio / original_ratio) - 1.0

    # propose a multiplicative change: 1 + aggressiveness * tanh(rel_change)
    # tanh keeps proposal bounded and smooth for very large deviations
    multiplier = 1.0 + aggressiveness * math.tanh(rel_change)
    proposed = current_noise * multiplier

    # clamp proposal to sensible range
    proposed = float(np.clip(proposed, min_noise, max_noise))

    # smooth the update to avoid jitter
    new_noise = float(smoothing * current_noise + (1.0 - smoothing) * proposed)

    # final clamp for safety
    new_noise = float(np.clip(new_noise, min_noise, max_noise))
    return new_noise

# (Optional) QNode for real hardware, not used in simulation
# def quantum_circuit_real_machine(noise, weights):
#     weights = weights.reshape(q_depth, n_qubits)
#     for i in range(n_qubits):
#         qml.RY(noise[i], wires=i)
#         qml.RX(noise[i], wires=i)
#     for i in range(q_depth):
#         for y in range(n_qubits):
#             qml.RY(weights[i][y], wires=y)
#         for y in range(n_qubits - 1):
#             qml.CZ(wires=[y, y+1])
#     return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

# Quantum Generator (no patching): one parameter tensor of shape (q_depth, n_qubits)
class QuantumGenerator(nn.Module):
    def __init__(self, q_delta=1.0):
        super().__init__()
        # Single parameter tensor instead of a list
        self.q_params = nn.Parameter(q_delta * torch.rand(q_depth, n_qubits))
    def forward(self, noise_batch):
        outputs = []
        for noise in noise_batch:
            f = quantum_circuit(noise, self.q_params)
            # if args.env == 'Real':
                # f = quantum_circuit_real_machine(noise, self.q_params)
            # f is a tensor of length n_qubits; convert to float and collect
            outputs.append(f.float())
        # Stack into (batch_size, pca_dims) tensor
        return torch.stack(outputs)

# Initialize models and optimizers
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
discriminator = Discriminator().to(device)
generator = QuantumGenerator().to(device)
criterion = nn.BCELoss()
optD = optim.SGD(discriminator.parameters(), lr=0.05)
optG = optim.SGD(generator.parameters(), lr=0.3)

real_labels = torch.ones(batch_size, dtype=torch.float, device=device)
fake_labels = torch.zeros(batch_size, dtype=torch.float, device=device)

# Training loop (simplified)
noise_upper_bound = math.pi / 8
original_ratio = None

os.makedirs("gen_images_dist", exist_ok=True)

print(f"Starting training... Epochs: {args.num_iter}, Batches: {len(dataloader)}")

for epoch in range(args.num_iter):
    for pca_batch in dataloader:

        # Prepare real PCA batch
        real_data = pca_batch.to(device).float()  # shape (batch_size, pca_dims)

        # Generate fake PCA vectors via quantum generator
        noise = torch.rand(batch_size, n_qubits, device=device) * noise_upper_bound
        fake_data = generator(noise)  # shape (batch_size, pca_dims)

        # Update Discriminator
        discriminator.zero_grad()
        out_real = discriminator(real_data).view(-1)
        out_fake = discriminator(fake_data.detach()).view(-1)
        errD = criterion(out_real, real_labels) + criterion(out_fake, fake_labels)
        errD.backward()
        optD.step()

        # Update Generator
        generator.zero_grad()
        out_fake_forG = discriminator(fake_data).view(-1)
        errG = criterion(out_fake_forG, real_labels)
        errG.backward()
        optG.step()

        # Optionally adjust noise schedule (as in original)
        if original_ratio is None:
            original_ratio = errD.detach().item() / errG.detach().item()
        noise_upper_bound = get_noise_upper_bound(errG, errD, original_ratio, noise_upper_bound)

    # End of epoch: (optional diagnostics, not shown here)
    print(f'Epoch: {epoch}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}')

    generator.eval()
    with torch.no_grad():
        # generate noise for a few samples
        noise = torch.rand(batch_size, n_qubits, device=device) * noise_upper_bound
        gen_pca = generator(noise).detach().cpu().numpy()  # (batch_size, pca_dims)

        # inverse PCA to reconstruct full image (20x20)
        gen_images = pca.inverse_transform(gen_pca)

        # Scale to 0–255 and save each image
        for i in range(min(batch_size, 4)):  # save a few images
            img = gen_images[i].reshape(image_size, image_size)
            # optional normalization: clip to [0,1]
            img = np.clip(img, 0, 1)
            img_uint8 = (img * 255).astype(np.uint8)

            im = Image.fromarray(img_uint8)
            save_path = f"gen_images_dist/epoch_{epoch:04d}_sample_{i:02d}.png"
            im.save(save_path)
            print(f"Saved: {save_path}")

    generator.train()

# After training: generate and invert PCA to images for visualization
noise = torch.rand(batch_size, n_qubits, device=device) * noise_upper_bound
gen_pca = generator(noise).detach().cpu().numpy()  # (batch_size, pca_dims)
gen_images = pca.inverse_transform(gen_pca)      # (batch_size, orig_dims)
# Reshape and save the first image (thresholded)
im_array = gen_images[0].reshape(image_size, image_size)
binary_im = (im_array <= 0.5).astype(np.uint8) * 255  # simple binarization
from PIL import Image
Image.fromarray(binary_im).save(f"gen_images_dist/{selected_class}_{epoch}.png")

