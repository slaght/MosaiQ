import os
from PIL import Image
import math
import logging
import matplotlib.pyplot as plt
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


# -------------------------------------------------------------------------
# SCRIPT SETUP
# -------------------------------------------------------------------------

# Argument parsing --------------------------------------------------------
parser = argparse.ArgumentParser()
# Dataset options
parser.add_argument('--mode', dest='mode', type=str, default='train')
parser.add_argument('--ds', dest='dataset', type=str, default='MNIST')
parser.add_argument('--ds_class', dest='ds_class', type=str, default='5')
parser.add_argument('--num_iter', dest='num_iter', type=int, default=1000)
parser.add_argument('--image_size', dest='image_size', type=int, default=20)

# Training stabilization options
parser.add_argument('--opt', dest='opt', type=str, choices=['sgd','adam'], default='adam', help='Optimizer: sgd or adam (default: adam)')
parser.add_argument('--lrG', dest='lrG', type=float, default=2e-4, help='Generator learning rate (default: 2e-4)')
parser.add_argument('--lrD', dest='lrD', type=float, default=2e-4, help='Discriminator learning rate (default: 2e-4)')
parser.add_argument('--betas', dest='betas', type=str, default='0.5,0.999', help='Adam betas as comma-separated pair, e.g. 0.5,0.999')
parser.add_argument('--label-smooth', dest='label_smooth', type=float, default=0.1, help='One-sided label smoothing for real labels, e.g. 0.1 -> real=0.9')
parser.add_argument('--instance-noise', dest='instance_noise', type=float, default=0.02, help='Stddev of Gaussian instance noise added to D inputs (0 to disable)')
parser.add_argument('--d-steps', dest='d_steps', type=int, default=1, help='Number of discriminator updates per generator update')
parser.add_argument('--pca-dims', dest='pca_dims_arg', type=int, default=None, help='Override PCA feature count (also sets n_qubits). Use carefully for speed.')
parser.add_argument('--minibatch-std', dest='minibatch_std', type=int, default=1, help='Enable minibatch standard deviation trick in D (1 on, 0 off).')
parser.add_argument('--fm-weight', dest='fm_weight', type=float, default=5.0, help='Feature matching loss weight added to G loss (default: 5.0).')

# Patching generator options
parser.add_argument('--patch-size', dest='patch_size', type=int, default=5, help='Number of qubits/features per patch generator (default: 5).')
parser.add_argument('--n-generators', dest='n_generators_arg', type=int, default=None, help='Number of patch generators. If not set, computed from PCA dims / patch size.')

# Utility options
parser.add_argument('--log-level', dest='log_level', type=str, default='INFO', help='Logging level: DEBUG, INFO, WARNING, ERROR')
parser.add_argument('--explain-args', dest='explain_args', action='store_true', help='Print detailed explanations for each CLI argument and exit')

args = parser.parse_args()

# If requested, print a helpful explanation for each CLI argument and exit
if args.explain_args:
        help_text = f"""
Usage explanation for mosaiq.py arguments:

    --mode: Operation mode for the script. 'train' runs GAN training. 'test' runs saved-model evaluation.
    --tr1: Label/name of the first trained generator model to use in test mode (e.g. '5' corresponds to 'generator_5').
    --tr2: Label/name of the second trained generator model to use in test mode.
    --ds: Dataset to use. Supported values: 'MNIST' (default), CIFAR (color) or 'Fashion'.
    --ds_class: Integer label of the digit/class to train on (default: 5). Only images of this class are used.
    # --env: Execution environment for the quantum circuit. 'simulation' (default) uses PennyLane lightning.qubit. Use 'Real' to swap in a function stub for a QPU backend.
    --num-iter: Number of training iterations to run when in 'train' mode (default: 20).
    --image_size: Size of the square images to process (default: 20). Images will be resized to image_size x image_size.
    --patch-size: Number of qubits/features produced by each patch sub-generator (default: 5).
    --n-generators: Number of patch sub-generators to chain. If not provided, it's inferred so total PCA dims = patch-size * n-generators.
    --pca-dims: Override total PCA feature count. When using patches, the actual PCA dims may be floored to a multiple of patch-size.
    --log-level: Logging verbosity. One of DEBUG, INFO (default), WARNING, ERROR.
    --explain-args: Shows this help text and exits.

Notes:
    - When running with --mode train the script will save models as 'generator_<ds_class>' and 'disc_<ds_class>'.
    - When running with --mode test, supply --tr1 and --tr2 with the labels of saved models to compare them.
    - The script expects data to be available or will download it into './datasets/<DatasetName>' by default.
"""
        print(help_text)
        import sys
        sys.exit(0)

# Logger setup --------------------------------------------------------
# Configure a simple logger so debug statements are consistent and optional
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(' ')

numeric_level = getattr(logging, args.log_level.upper(), None)
if not isinstance(numeric_level, int):
    logger.warning(f"Invalid log level '{args.log_level}', defaulting to INFO")
    numeric_level = logging.INFO
logger.setLevel(numeric_level)


# -------------------------------------------------------------------------
# DATA LOADING AND PROCESSING
# -------------------------------------------------------------------------

# Dataset loading ---------------------------------------------------------
# Load and preprocess dataset, flattening to image_size x image_size
transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Lambda(torch.flatten),
])

if args.dataset == 'MNIST':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./datasets', download=True, train=True, transform=transform),
        batch_size=10000, shuffle=True
    )
elif args.dataset == 'Fashion':
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./datasets', download=True, train=True, transform=transform),
        batch_size=10000, shuffle=True
    )
elif args.dataset in ('CIFAR', 'CIFAR10'):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./datasets', download=True, train=True, transform=transform),
        batch_size=10000, shuffle=True
    )
else:
    raise ValueError(f"Unsupported dataset '{args.dataset}'. Supported: MNIST, Fashion, CIFAR")

# Build training data from downloaded dataset -----------------------------
# Collect only the selected class. Accept integer labels or (for CIFAR) class names.
train_data = []
selected_class_arg = args.ds_class

# Resolve selected class index depending on dataset
selected_class_idx = None
if args.dataset in ('CIFAR', 'CIFAR10'):
    cifar_labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    # Try interpret as integer index first, else as a name
    try:
        selected_class_idx = int(selected_class_arg)
    except Exception:
        sc = str(selected_class_arg).strip().lower()
        if sc in cifar_labels:
            selected_class_idx = cifar_labels.index(sc)
        else:
            raise ValueError(
                f"Unknown CIFAR class '{selected_class_arg}'. Valid names: {cifar_labels} or integer 0-9"
            )
else:
    # MNIST/Fashion expect an integer class label
    try:
        selected_class_idx = int(selected_class_arg)
    except Exception:
        raise ValueError("For MNIST/Fashion, --ds_class must be an integer label")

for (data, labels) in train_loader:
    for x, y in zip(data, labels):
        if (selected_class_idx is None) or (int(y) == selected_class_idx):
            train_data.append(x.numpy())

# Function to scale numpy data to [min,max]
def scale_data(data, scale=None, dtype=np.float32):
    if scale is None:
        scale = [-1, 1]
    min_data, max_data = float(np.min(data)), float(np.max(data))
    min_scale, max_scale = float(scale[0]), float(scale[1])
    data = ((max_scale - min_scale) * (data - min_data) / (max_data - min_data)) + min_scale
    return data.astype(dtype)

train_data = scale_data(np.array(train_data), [0, 1]) # scale pixels to [0,1]





# CONTINUE TO COMMENT AFTER HERE:

# Prepare output directories based on dataset and class
images_dir = os.path.join('sample_output', str(args.dataset).lower(), selected_class_arg)
checkpoints_dir = os.path.join('checkpoints', str(args.dataset).lower(), selected_class_arg)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)

# Set model dimensions
image_size = args.image_size        # e.g. 20
batch_size = 8

# Derive original flattened dimension from loaded data (handles RGB vs grayscale)
if train_data.ndim == 2:
    orig_dims = train_data.shape[1]
else:
    # fallback: assume square grayscale
    orig_dims = image_size * image_size

# Base PCA dims target: use image_size by default (as in newer version), limited by orig dims
base_pca_dims = min(image_size, orig_dims)
if args.pca_dims_arg is not None:
    base_pca_dims = int(max(1, min(args.pca_dims_arg, orig_dims)))

# Patching configuration
patch_size = max(1, int(args.patch_size))
if args.n_generators_arg is None:
    # Ensure total PCA dims is a multiple of patch_size by flooring
    pca_dims = max(patch_size, (base_pca_dims // patch_size) * patch_size)
    if pca_dims != base_pca_dims:
        logger.info(f"Adjusting PCA dims from {base_pca_dims} to {pca_dims} to align with patch-size {patch_size}.")
    n_generators = pca_dims // patch_size
else:
    n_generators = max(1, int(args.n_generators_arg))
    pca_dims = n_generators * patch_size
    if pca_dims > orig_dims:
        logger.warning(f"Requested PCA dims ({pca_dims}) exceed available features ({orig_dims}). Reducing to {orig_dims} by adjusting n_generators.")
        n_generators = max(1, orig_dims // patch_size)
        pca_dims = n_generators * patch_size

# Number of qubits equals patch size (each sub-generator outputs one patch)
n_qubits = patch_size
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

# Discriminator with optional minibatch standard deviation and feature output
class Discriminator(nn.Module):
    def __init__(self, use_minibatch_std: bool = True):
        super().__init__()
        self.use_mbstd = bool(use_minibatch_std)
        in_dim = pca_dims + (1 if self.use_mbstd else 0)
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        self.act = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, x, return_features: bool = False):
        # Minibatch standard deviation trick (append one scalar feature)
        if self.use_mbstd and x.shape[0] > 1:
            # per-feature std over batch, then average to single scalar
            eps = 1e-8
            std_per_feat = torch.sqrt(torch.var(x, dim=0, unbiased=False) + eps)
            mbstd_scalar = std_per_feat.mean().view(1, 1).expand(x.shape[0], 1)
            x = torch.cat([x, mbstd_scalar.to(x.device)], dim=1)
        elif self.use_mbstd:
            # batch size 1: append zeros
            x = torch.cat([x, torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)], dim=1)

        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        logits = self.out_act(self.fc3(h))
        if return_features:
            return logits, h
        return logits

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

# Simple Frechet distance (FID-like) between two sets of flattened images
def calculate_fid(act1: np.ndarray, act2: np.ndarray) -> float:
    """Compute Frechet distance between two multivariate Gaussians.

    Parameters
    ----------
    act1 : array, shape (n1, d)
        First set of activations (here: flattened images).
    act2 : array, shape (n2, d)
        Second set of activations.

    Returns
    -------
    float
        Frechet distance value (lower is better).
    """
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)

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

# Quantum Generator (no patching): one parameter tensor of shape (q_depth, n_qubits)
class QuantumGenerator(nn.Module):
    """Patch-based quantum generator composed of multiple sub-generators.

    Each sub-generator produces `patch_size` features (equal to `n_qubits`).
    Concatenating `n_generators` patches yields a vector of length `pca_dims`.
    """
    def __init__(self, n_generators: int, q_delta: float = 1.0):
        super().__init__()
        self.n_generators = int(n_generators)
        # One parameter tensor per patch sub-generator
        self.q_params = nn.ParameterList([
            nn.Parameter(q_delta * torch.rand(q_depth, n_qubits))
            for _ in range(self.n_generators)
        ])

    def forward(self, noise_batch: torch.Tensor) -> torch.Tensor:
        # noise_batch: (B, n_qubits)
        B = noise_batch.shape[0]
        device_in = noise_batch.device
        patch_mats = []
        # Generate each patch with its own parameters
        for params in self.q_params:
            patch_outputs = []
            for noise in noise_batch:
                f = quantum_circuit(noise, params)
                patch_outputs.append(f.float())
            patch_mat = torch.stack(patch_outputs)  # (B, n_qubits)
            patch_mats.append(patch_mat)
        # Concatenate all patches to form (B, pca_dims)
        images = torch.cat(patch_mats, dim=1)
        return images.to(device_in)

# Initialize models and optimizers
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
discriminator = Discriminator(use_minibatch_std=bool(args.minibatch_std)).to(device)
generator = QuantumGenerator(n_generators=n_generators).to(device)
criterion = nn.BCELoss()
# Optimizer selection
if args.opt == 'adam':
    try:
        beta_vals = tuple(float(x) for x in args.betas.split(','))
        if len(beta_vals) != 2:
            raise ValueError
    except Exception:
        beta_vals = (0.5, 0.999)
    optD = optim.Adam(discriminator.parameters(), lr=args.lrD, betas=beta_vals)
    optG = optim.Adam(generator.parameters(), lr=args.lrG, betas=beta_vals)
else:
    optD = optim.SGD(discriminator.parameters(), lr=args.lrD, momentum=0.5)
    optG = optim.SGD(generator.parameters(), lr=args.lrG, momentum=0.5)

# Labels with optional one-sided smoothing for real labels
real_label_val = 1.0 - max(0.0, float(args.label_smooth))
real_labels = torch.full((batch_size,), real_label_val, dtype=torch.float, device=device)
fake_labels = torch.zeros(batch_size, dtype=torch.float, device=device)

instance_noise_std = float(args.instance_noise)

def add_instance_noise(x, std):
    if std <= 0:
        return x
    return x + torch.randn_like(x) * std

# Training loop (simplified)
noise_upper_bound = math.pi / 8
original_ratio = None

# Output a diagram of the quantum generator circuit and save a PNG before training starts
try:
    sample_noise = torch.zeros(n_qubits, dtype=torch.float32)
    sample_weights = torch.zeros(q_depth, n_qubits, dtype=torch.float32)
    ascii_circuit = qml.draw(quantum_circuit)(sample_noise, sample_weights)
    print("\nQuantum generator circuit:\n" + ascii_circuit)
    fig, ax = qml.draw_mpl(quantum_circuit)(sample_noise, sample_weights)
    fig.savefig('quantum_generator.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved circuit diagram to quantum_generator.png")
except Exception as e:
    print(f"Warning: could not render circuit diagram: {e}")

# Output directories created above per dataset/class

print(f"Starting training... Epochs: {args.num_iter}, Batches: {len(dataloader)}")

# Track history for plots
epoch_gen_losses = []
epoch_disc_losses = []
epoch_fids = []

for epoch in range(args.num_iter):
    for pca_batch in dataloader:

        # Prepare real PCA batch
        real_data = pca_batch.to(device).float()  # shape (batch_size, pca_dims)

        # Generate fake PCA vectors via quantum generator
        noise = torch.rand(batch_size, n_qubits, device=device) * noise_upper_bound
        fake_data = generator(noise)  # shape (batch_size, pca_dims)

        # Update Discriminator (possibly multiple steps)
        errD = None
        for _ in range(max(1, int(args.d_steps))):
            discriminator.zero_grad()
            real_in = add_instance_noise(real_data, instance_noise_std)
            fake_in = add_instance_noise(fake_data.detach(), instance_noise_std)
            out_real = discriminator(real_in).view(-1)
            out_fake = discriminator(fake_in).view(-1)
            errD_step = criterion(out_real, real_labels) + criterion(out_fake, fake_labels)
            errD_step.backward()
            optD.step()
            errD = errD_step

    # Update Generator (with feature matching)
    generator.zero_grad()
    fake_forG_in = add_instance_noise(fake_data, instance_noise_std)
    # Get discriminator outputs and intermediate features
    out_fake_forG, fake_feats = discriminator(fake_forG_in, return_features=True)
    out_fake_forG = out_fake_forG.view(-1)
    # Compute real features (detach to avoid gradients into D)
    real_for_feats = add_instance_noise(real_data, instance_noise_std)
    _, real_feats = discriminator(real_for_feats, return_features=True)
    real_feats = real_feats.detach()
    # Feature matching: match mean feature activations
    fm_weight = float(args.fm_weight)
    mean_fake = torch.mean(fake_feats, dim=0)
    mean_real = torch.mean(real_feats, dim=0)
    fm_loss = torch.mean((mean_fake - mean_real) ** 2)
    # Standard GAN loss + feature matching term
    errG = criterion(out_fake_forG, real_labels) + fm_weight * fm_loss
    errG.backward()
    optG.step()

    # Optionally adjust noise schedule (as in original)
    if original_ratio is None:
        original_ratio = errD.detach().item() / errG.detach().item()
    noise_upper_bound = get_noise_upper_bound(errG, errD, original_ratio, noise_upper_bound)

    # End of epoch: diagnostics and artifact saving handled below

    generator.eval()
    with torch.no_grad():
        # generate noise for a few samples
        noise = torch.rand(batch_size, n_qubits, device=device) * noise_upper_bound
        gen_pca = generator(noise).detach().cpu().numpy()  # (batch_size, pca_dims)

        # inverse PCA to reconstruct full image (20x20)
        gen_images = pca.inverse_transform(gen_pca)

        # Save only one image per epoch (first sample)
        i = 0
        channels = int(orig_dims // (image_size * image_size)) if image_size > 0 else 1
        if channels == 1:
            img = gen_images[i].reshape(image_size, image_size)
            img = np.clip(img, 0, 1)
            img_uint8 = (img * 255).astype(np.uint8)
            im = Image.fromarray(img_uint8)
        else:
            try:
                img_chw = gen_images[i].reshape(channels, image_size, image_size)
            except Exception:
                img_hwc = gen_images[i].reshape(image_size, image_size, channels)
                img = np.clip(img_hwc, 0, 1)
                img_uint8 = (img * 255).astype(np.uint8)
                im = Image.fromarray(img_uint8)
            else:
                img_hwc = np.moveaxis(img_chw, 0, -1)
                img = np.clip(img_hwc, 0, 1)
                img_uint8 = (img * 255).astype(np.uint8)
                im = Image.fromarray(img_uint8)

        save_path = os.path.join(images_dir, f"epoch_{epoch:04d}.png")
        im.save(save_path)

    # Compute FID for this epoch using a subset of real data
    try:
        real_pool = train_data
        # Use up to 1000 real samples for speed
        max_real = 1000
        if real_pool.shape[0] > max_real:
            idx = np.random.choice(real_pool.shape[0], size=max_real, replace=False)
            real_sample = real_pool[idx]
        else:
            real_sample = real_pool
        # Flatten generated images (already (batch, orig_dims))
        gen_flat = gen_images.reshape(gen_images.shape[0], -1)
        fid_value = calculate_fid(gen_flat, real_sample)
    except Exception as e:
        fid_value = float('nan')
        print(f"Warning: FID computation failed at epoch {epoch}: {e}")

    # Update histories and save plots
    epoch_disc_losses.append(float(errD.detach().item()))
    epoch_gen_losses.append(float(errG.detach().item()))
    epoch_fids.append(float(fid_value))

    # Plot and save losses
    try:
        plt.figure(figsize=(6,4))
        plt.plot(epoch_disc_losses, label='D loss')
        plt.plot(epoch_gen_losses, label='G loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'loss.png'))
        plt.close()
    except Exception as e:
        print(f"Warning: could not save loss plot: {e}")

    # Plot and save FID
    try:
        plt.figure(figsize=(6,4))
        plt.plot(epoch_fids, label='FID')
        plt.xlabel('Epoch')
        plt.ylabel('FID (lower is better)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'fid.png'))
        plt.close()
    except Exception as e:
        print(f"Warning: could not save FID plot: {e}")

    # Log epoch summary with FID
    print(f"Epoch: {epoch}, D Loss: {epoch_disc_losses[-1]:0.4f}, G Loss: {epoch_gen_losses[-1]:0.4f}, FID: {epoch_fids[-1]:0.2f}")

    # Save models for this dataset/class after each epoch
    torch.save(generator.state_dict(), os.path.join(checkpoints_dir, f"G_epoch_{epoch:04d}.pt"))
    torch.save(discriminator.state_dict(), os.path.join(checkpoints_dir, f"D_epoch_{epoch:04d}.pt"))
    # Also keep latest symlinks/files for convenience
    torch.save(generator.state_dict(), os.path.join(checkpoints_dir, "G_latest.pt"))
    torch.save(discriminator.state_dict(), os.path.join(checkpoints_dir, "D_latest.pt"))

    generator.train()