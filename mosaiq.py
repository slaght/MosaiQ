# Library imports
# Note: Much of the code is based on the structure of PatchGan implemented in this pennylane tutorial https://pennylane.ai/qml/demos/tutorial_quantum_gans.html
import math
import os
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pennylane as qml
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from matplotlib import cm
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
import argparse

# Configure a simple logger so debug statements are consistent and optional
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(' ')

parser = argparse.ArgumentParser()
parser.add_argument('--mode', dest='mode', type=str, default='train')
parser.add_argument('--tr1', dest='trained_model1', type=str, default='None')
parser.add_argument('--tr2', dest='trained_model2', type=str, default='None')
parser.add_argument('--ds', dest='dataset', type=str, default='MNIST')
parser.add_argument('--ds_class', dest='ds_class', type=int, default=5)
parser.add_argument('--env', dest='env', type=str, default='simulation')
parser.add_argument('--log-level', dest='log_level', type=str, default='INFO',
                    help='Logging level: DEBUG, INFO, WARNING, ERROR')
parser.add_argument('--explain-args', dest='explain_args', action='store_true',
                    help='Print detailed explanations for each CLI argument and exit')
parser.add_argument('--num-iter', dest='num_iter', type=int, default=20,
                    help='Number of training iterations (default: 20)')


args = parser.parse_args()
# Apply requested log level
# Apply requested log level
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
    --env: Execution environment for the quantum circuit. 'simulation' (default) uses PennyLane lightning.qubit. Use 'Real' to swap in a function stub for a QPU backend.
    --num-iter: Number of training iterations to run when in 'train' mode (default: 20).
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

def scale_data(data, scale=None, dtype=np.float32):
    """Linearly scale numpy data to a target range.

    Args:
        data: np.ndarray of input values.
        scale: tuple/list [min, max] target range (default [-1,1]).
        dtype: output numpy dtype.

    Returns:
        Scaled data as numpy array with dtype.
    """
    if scale is None:
        scale = [-1, 1]
    min_data, max_data = [float(np.min(data)), float(np.max(data))]
    min_scale, max_scale = [float(scale[0]), float(scale[1])]
    data = ((max_scale - min_scale) * (data - min_data) / (max_data - min_data)) + min_scale
    return data.astype(dtype)

if args.dataset == 'MNIST':
    # Load full MNIST dataset (flattened) into a single large batch
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist', download=True, train=True,
                       transform=transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           transforms.Lambda(torch.flatten),
                       ])),
        batch_size=10000,
        shuffle=True,
    )
elif args.dataset == 'Fashion':
    # Alternative: FashionMNIST
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./fashion', download=True, train=True,
                              transform=transforms.Compose([
                                  torchvision.transforms.ToTensor(),
                                  transforms.Lambda(torch.flatten),
                              ])),
        batch_size=10000,
        shuffle=True,
    )
train_data = []
# Select only the requested digit/class from the dataset
selected_class = args.ds_class
selected_class_name = str(selected_class)
for (data, labels) in train_loader:
        for x, y in zip(data, labels):
                if y == selected_class:
                        train_data.append(x.numpy())

# Function from https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
def calculate_fid(act1, act2):
    """Compute Frechet Inception Distance between two sets of activations.

    act1, act2: arrays of shape (n_samples, n_features)
    """
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
train_data = scale_data(np.array(train_data), [0, 1])

# Model / data hyperparameters
image_size = 5
batch_size = 8
pca_dims = image_size * batch_size
n_qubits = image_size
q_depth = 6
n_generators = batch_size

# Fit PCA on flattened images and transform
pca = PCA(n_components=pca_dims)
pca_data_full = pca.fit_transform(train_data)

# Ordering used to rearrange PCA components into patches for the quantum generator
ordering = []
for i in range(8):
    k = 4 * i
    l = [i, 39 - k, 38 - k, 37 - k, 36 - k]
    ordering.append(l)

pca_min, pca_max = np.min(pca_data_full), np.max(pca_data_full)

# Pair transformed pca vectors with original images if needed later
full_train_data = [(i, j) for i, j in zip(scale_data(pca_data_full), train_data)]

transform = transforms.Compose([transforms.ToTensor()])
dataloader = torch.utils.data.DataLoader(
    scale_data(pca_data_full), batch_size=batch_size, shuffle=True, drop_last=True
)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(pca_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


dev = qml.device("lightning.qubit", wires=n_qubits)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_circuit(noise, weights):
    """Quantum circuit used by the quantum generator.

    noise: vector of length n_qubits used as rotation angles for initial layer
    weights: flattened parameters reshaped to (q_depth, n_qubits)
    """
    weights = weights.reshape(q_depth, n_qubits)
    # Encode classical noise into single-qubit rotations
    for qubit_idx in range(n_qubits):
        qml.RY(noise[qubit_idx], wires=qubit_idx)
        qml.RX(noise[qubit_idx], wires=qubit_idx)

    # Apply a sequence of parameterized rotation layers and CZ entangling gates
    for layer in range(q_depth):
        for wire in range(n_qubits):
            qml.RY(weights[layer][wire], wires=wire)
        for wire in range(n_qubits - 1):
            qml.CZ(wires=[wire, wire + 1])

    # Return expectation values of PauliX for each qubit as the quantum feature vector
    return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

# Uncomment this line for running on real machines
#@qml.qnode(qml.device(name='qiskit.ibmq', wires=5, backend='ibmq_jakarta', ibmqx_token="ibm_token_here"))
def quantum_circuit_real_machine(noise, weights):

    # This function mirrors `quantum_circuit` but is intended to be
    # swapped in if using a real QPU backend. Keeping it separate so
    # minor backend-specific adaptations can be made here.
    weights = weights.reshape(q_depth, n_qubits)

    for i in range(n_qubits):
        qml.RY(noise[i], wires=i)
        qml.RX(noise[i], wires=i)

    # Repeated layer
    for i in range(q_depth):
        for y in range(n_qubits):
            qml.RY(weights[i][y], wires=y)
        for y in range(n_qubits - 1):
            qml.CZ(wires=[y, y + 1])
    return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

class QuantumGenerator(nn.Module):
    """Quantum generator that uses multiple small quantum patches to build a feature vector."""

    def __init__(self, n_generators, q_delta=1):
        super().__init__()

        # Each generator has a separate parameter tensor of shape (q_depth, n_qubits)
        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(q_depth, n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )

        self.n_generators = n_generators

    def forward(self, noise_batch):
        """Forward pass: run quantum circuit for each noise vector and each patch parameter set.

        Returns a tensor of shape (batch_size, n_generators * image_size).
        """
        images = torch.Tensor(noise_batch.size(0), 0).to(device)
        patch_size = image_size
        # Iterate over each set of quantum patch parameters
        for params in self.q_params:
            patches = torch.Tensor(0, patch_size).to(device)
            for single_noise in noise_batch:
                f = quantum_circuit(single_noise, params)
                if args.env == 'Real':
                    f = quantum_circuit_real_machine(single_noise, params)
                f = f.clone().detach()
                q_out = f.float().unsqueeze(0)
                patches = torch.cat((patches, q_out))

            # Reorder PCA components according to predefined ordering
            flattened_order = [j for sub in ordering for j in sub]
            patches = torch.flatten(patches)
            patches = patches[flattened_order]  # Rearrange order of pca components
            patches = patches.reshape(batch_size, patch_size)
            images = torch.cat((images, patches), 1)
        return images




lrG = 0.3
lrD = 0.05
# Number of training iterations (can be overridden via --num-iter)
num_iter = args.num_iter

gen_losses = []
disc_losses = []
discriminator = Discriminator().to(device)
generator = QuantumGenerator(n_generators).to(device)
criterion = nn.BCELoss()
optD = optim.SGD(discriminator.parameters(), lr=lrD)
optG = optim.SGD(generator.parameters(), lr=lrG)
real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)
counter = 0

# Noise scaling upper bound used when sampling rotation angles for the quantum circuit
noise_upper_bound = math.pi / 8

def relu(x):
    return x * (x > 0)


def get_noise_upper_bound(gen_loss, disc_loss, original_ratio):
    """Adjust the noise upper bound dynamically based on discriminator/generator loss ratio.

    This function mirrors the original heuristic: start near pi/8 and increase up to pi/2
    as the ratio deviates from the original ratio.
    """
    R = disc_loss.detach().numpy() / gen_loss.detach().numpy()
    return math.pi / 8 + (5 * math.pi / 8) * relu(np.tanh((R - (original_ratio))))

original_ratio = None
upper_bounds = [math.pi / 8]
results = []
generated_images = []

if args.mode == 'train':
    logger.info('Starting training loop')
    for epoch in tqdm(range(num_iter)):
        for batch_idx, pca_batch in enumerate(dataloader):
            # Prepare real data batch in PCA space
            pca_data = pca_batch
            data = pca_data.reshape(batch_size, pca_dims)
            real_data = data.to(device).to(torch.float32)

            # Sample noise and generate fake PCA vectors via the quantum generator
            noise = torch.rand(batch_size, n_qubits, device=device) * noise_upper_bound
            fake_data = generator(noise)

            # -------------------- Update Discriminator --------------------
            discriminator.zero_grad()
            outD_real = discriminator(real_data).view(-1)
            outD_fake = discriminator(fake_data.detach()).view(-1)
            errD_real = criterion(outD_real, real_labels)
            errD_fake = criterion(outD_fake, fake_labels)
            errD_real.backward()
            errD_fake.backward()
            errG = criterion(outD_fake, real_labels)
            errD = errD_real + errD_fake
            gen_losses.append(errG.detach().numpy())
            disc_losses.append(errD.detach().numpy())
            optD.step()

            # -------------------- Update Generator --------------------
            generator.zero_grad()
            outD_fake = discriminator(fake_data).view(-1)
            errG = criterion(outD_fake, real_labels)
            errG.backward()
            optG.step()

            # Keep track of a baseline ratio between losses to stabilize the noise schedule
            if original_ratio is None:
                original_ratio = errD.detach().numpy() / errG.detach().numpy()
            noise_upper_bound = get_noise_upper_bound(errG, errD, original_ratio)
            upper_bounds.append(noise_upper_bound)
            np.save(f'upper_bounds_{selected_class_name}', upper_bounds)
            counter += 1

            test_images = generator(noise).detach().numpy()
            test_images = pca.inverse_transform(test_images)
            
            # fid = calculate_fid(test_images.reshape([batch_size, 784]), train_data)
            # logger.info(f"FID at checkpoint {counter}: {fid}")

            # Periodically save diagnostics and generated images
            if counter % 100 == 0:
                print(f'Iteration: {counter}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}')
                # logger.info(f"Checkpoint: epoch={epoch} batch={batch_idx} counter={counter}")
                test_images = scale_data(test_images, [0, 1])
                np.save(f'gen_loss_{selected_class_name}', gen_losses)
                np.save(f'disc_loss_{selected_class_name}', disc_losses)
                from PIL import Image
                im = np.reshape(test_images[0], [28, 28])
                new_im = np.zeros([28, 28])
                for i in range(28):
                    for j in range(28):
                        # Binary threshold for visualization
                        new_im[i][j] = 0.0 if im[i][j] > 0.5 else 1.0
                im = Image.fromarray(np.uint8(255 - (new_im * 255)))
                im = im.save(os.path.join("gen_images_dist", f"{selected_class_name}_{counter}.png"))
                torch.save(generator.state_dict(), f"generator_{selected_class_name}")
                torch.save(discriminator.state_dict(), f"disc_{selected_class_name}")

# Evaluate variance on different trained models (tr1 and tr2)
if args.mode =='test':
    logger.info('Entering test mode: evaluating variance of two trained models')
    cdf_pairs = []
    model_labels = [args.trained_model1, args.trained_model2]
    for model_label in model_labels:
        if model_label is None or model_label == 'None':
            logger.warning(f"Skipping empty model label: {model_label}")
            continue
        logger.info(f"Loading generator weights for: {model_label}")
        generator.load_state_dict(torch.load(f"generator_{model_label}"))
        # Collect multiple generated samples to compute per-sample variance
        outputs = np.zeros([25, batch_size, 784])
        noise_upper_bounds = np.load(f'upper_bounds_{model_label}.npy')
        # Use the stored maximum noise from the file (scalar or array). Fall back to current bound if missing.
        noise_scale = float(noise_upper_bounds[-1]) if hasattr(noise_upper_bounds, '__len__') else float(noise_upper_bounds)
        for i in range(25):
            sampled_noise = torch.rand(batch_size, n_qubits, device=device) * noise_scale
            output = pca.inverse_transform(generator(sampled_noise).detach().numpy())
            outputs[i] = output

        outputs = outputs.reshape([25 * batch_size, 784])
        mean_vec = np.mean(outputs, axis=0)
        per_sample_variances = []
        for i in range(outputs.shape[0]):
            per_sample_variances.append(np.sum(np.square(outputs[i] - mean_vec)))

        x = np.sort(per_sample_variances)
        y = np.arange(outputs.shape[0]) / float(outputs.shape[0])
        cdf_pairs.append((x, y))

    # Plot comparison if we have two results
    if len(cdf_pairs) >= 2:
        plt.plot(cdf_pairs[0][0], cdf_pairs[0][1], marker='o', label=str(model_labels[0]))
        plt.plot(cdf_pairs[1][0], cdf_pairs[1][1], marker='o', label=str(model_labels[1]))
        np.save('vars', cdf_pairs)
        plt.legend()
        plt.show()
    else:
        logger.info('Not enough model results to plot CDFs; finished test mode.')
