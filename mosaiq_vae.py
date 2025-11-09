import math, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pennylane as qml
from tqdm.auto import tqdm

# -----------------------
# Hyperparameters and setup
# -----------------------
class Args:
    dataset = "MNIST"          # "MNIST" or "FashionMNIST"
    image_size = 28            # Image width/height (28 for full resolution, or smaller e.g. 14 for speed)
    patch_size = 2             # Patch size for quantum generator (each patch is patch_size x patch_size)
    n_qubits = None            # Number of qubits per quantum sub-generator (if None, defaults to patch_size**2)
    n_patches = None           # Number of sub-generators (if None, will use (image_size/patch_size)**2 patches to tile the image)
    latent_dim = None          # Latent dimension (if None, will default to n_qubits, i.e. one latent value per qubit)
    # Training hyperparams
    epochs = 10
    batch_size = 64
    lr = 1e-3                  # Learning rate for all networks (can tune per network if desired)
    n_critic = 5               # Number of critic updates per generator update (WGAN-GP)
    lambda_gp = 10.0           # Gradient penalty coefficient
    gp_every = 1               # Apply gradient penalty every this many critic steps (1 = every step)
    gamma = 1.0                # Weight for reconstruction loss in generator update (balance content vs style)
    save_model = True          # Whether to save model checkpoints
    output_dir = "vae_qwgan_outputs"  # Directory for outputs (models and sample images)
    seed = 0

args = Args()

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Determine derived architecture parameters
if args.n_qubits is None:
    args.n_qubits = args.patch_size ** 2
if args.n_patches is None:
    # Ensure image_size is divisible by patch_size
    assert args.image_size % args.patch_size == 0, "Image size must be an integer multiple of patch size."
    patches_per_side = args.image_size // args.patch_size
    args.n_patches = patches_per_side ** 2
if args.latent_dim is None:
    # Use latent dimension equal to number of qubits (each sub-generator uses entire latent vector)
    args.latent_dim = args.n_qubits

# Create output directory
if args.save_model or True:
    os.makedirs(args.output_dir, exist_ok=True)

print(f"Using {args.dataset} dataset with image size {args.image_size}x{args.image_size}.")
print(f"Patch-based quantum generator: {args.n_patches} patches of size {args.patch_size}x{args.patch_size}, {args.n_qubits} qubits each.")
print(f"Latent dim: {args.latent_dim}, Critic updates per batch: {args.n_critic}, λ_GP: {args.lambda_gp}, γ: {args.gamma}")

# -----------------------
# Data loading (MNIST or FashionMNIST)
# -----------------------
transform_list = []
# If using a smaller image size than 28, resize the images
if args.image_size != 28:
    transform_list.append(transforms.Resize(args.image_size))
transform_list += [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  # Normalize to [-1, 1]
transform = transforms.Compose(transform_list)

dataset_class = datasets.MNIST if args.dataset.upper() == "MNIST" else datasets.FashionMNIST
train_dataset = dataset_class(root="./data", train=True, download=True, transform=transform)
test_dataset  = dataset_class(root="./data", train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# -----------------------
# Model Definitions
# -----------------------
# Encoder: maps input image -> latent distribution (mu, logvar)
class Encoder(nn.Module):
    def __init__(self, latent_dim=4, img_size=28):
        super(Encoder, self).__init__()
        # Convolutional layers to downsample image
        # Use small architecture for speed
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)  # out: 16 x (img_size/2) x (img_size/2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1) # out: 32 x (img_size/4) x (img_size/4)
        # Compute output size after conv layers to define the linear layer input
        # (We can derive it since each conv halved the spatial dimensions)
        conv_out_size = img_size
        conv_out_size = (conv_out_size + 2*1 - 4) // 2 + 1  # after conv1
        conv_out_size = (conv_out_size + 2*1 - 4) // 2 + 1  # after conv2
        conv_feat_dim = 32 * conv_out_size * conv_out_size
        self.fc = nn.Linear(conv_feat_dim, 128)  # fully connected layer
        self.mu_layer = nn.Linear(128, latent_dim)
        self.logvar_layer = nn.Linear(128, latent_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        # Input x: [B, 1, H, W]
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = x.view(x.size(0), -1)        # flatten
        x = self.act(self.fc(x))
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

# Quantum sub-generator (a parametric quantum circuit producing a patch)
class QuantumSubgenerator(nn.Module):
    def __init__(self, n_qubits, latent_dim):
        super(QuantumSubgenerator, self).__init__()
        self.n_qubits = n_qubits
        self.latent_dim = latent_dim
        # Initialize trainable quantum circuit parameters (two layers of RY rotations per qubit)
        weight_shape = (2, n_qubits)  # 2 layers of rotations for each qubit
        # Use small random initialization for angles
        init_weights = 0.1 * torch.randn(weight_shape)
        self.weights = nn.Parameter(init_weights)  # trainable parameters for the quantum circuit
        # Set up PennyLane quantum node
        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        # Define the quantum circuit (QNode) with data embedding and parameterized rotations
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(weights, inputs):
            # Encode input latent vector into qubit rotations (angle embedding)
            # If latent_dim > n_qubits, take first n_qubits components (or pad if smaller)
            # Here we assume latent_dim >= n_qubits for embedding; otherwise, could repeat or pad.
            for i in range(self.n_qubits):
                # If latent vector shorter, wrap around index
                angle = inputs[i] if i < len(inputs) else 0.0
                qml.RY(angle, wires=i)
            # Layer 1: trainable rotations RY for each qubit
            for i in range(self.n_qubits):
                qml.RY(weights[0, i], wires=i)
            # Entangling layer (CNOT chain connecting all qubits in a ring)
            if self.n_qubits > 1:
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            # Layer 2: another trainable rotation on each qubit
            for i in range(self.n_qubits):
                qml.RY(weights[1, i], wires=i)
            # (Optionally could add another entanglement here for more expressivity)
            # Measure expectation <Z> on each qubit to get real-valued outputs
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        # Create QNode for the circuit
        self.qnode = qml.QNode(circuit, self.dev, interface="torch", diff_method="parameter-shift")

    def forward(self, latent):
        # latent: tensor of shape [batch, latent_dim]
        # We will call the QNode for each sample in the batch (sequentially, as PennyLane by default processes one set of inputs at a time).
        # For efficiency, consider using Pennylane's batch execution or parallelizing this loop in the future.
        outputs = []
        for i in range(latent.shape[0]):
            # Get single sample latent vector
            z = latent[i]
            # Ensure z is 1D tensor (detach not needed, we want grad to flow into z)
            # Run quantum circuit: returns expectation values for each qubit (length = n_qubits)
            out = self.qnode(self.weights, z)
            # out is a torch tensor of shape (n_qubits,)
            outputs.append(out)
        # Stack outputs for the batch
        outputs = torch.stack(outputs, dim=0)  # shape [batch, n_qubits]
        return outputs  # Note: outputs are in [-1,1] range due to expval of PauliZ

# Full Quantum Generator: holds multiple sub-generators to produce the full image
class QuantumGenerator(nn.Module):
    def __init__(self, n_patches, patch_size, n_qubits, latent_dim):
        super(QuantumGenerator, self).__init__()
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.n_qubits = n_qubits
        self.latent_dim = latent_dim
        # Create a list of sub-generators (one per patch)
        self.subgens = nn.ModuleList([
            QuantumSubgenerator(n_qubits, latent_dim) for _ in range(n_patches)
        ])
        # Determine how many patches per side to tile the image
        self.patches_per_side = int(math.sqrt(n_patches))
        assert self.patches_per_side ** 2 == n_patches, "n_patches should be a perfect square."
        # Each patch outputs patch_size*patch_size pixels (we expect n_qubits == patch_size**2 for a square patch output)
        assert self.n_qubits == self.patch_size * self.patch_size, "For square patches, n_qubits should equal patch_size^2."

    def forward(self, z):
        # z: latent tensor [batch, latent_dim]
        batch_size = z.size(0)
        # Each sub-generator will use the *same* latent input (global latent) to produce its patch
        # (Alternatively, one could segment z for each patch, but here we feed the full latent to all patches for global coherence.)
        patch_outputs = []  # to collect patch images from each sub-generator
        for subgen in self.subgens:
            # Compute patch output for the batch (shape [batch, n_qubits])
            patch_out = subgen(z)  # forward pass through quantum sub-generator
            # Reshape patch output to image patch dimensions [batch, 1, patch_size, patch_size]
            patch_img = patch_out.view(batch_size, 1, self.patch_size, self.patch_size)
            patch_outputs.append(patch_img)
        # Now stitch patches together into full images
        # patch_outputs list length = n_patches, each element shape [B, 1, patch_size, patch_size]
        # We arrange them row by row
        full_images = torch.zeros(batch_size, 1, self.patches_per_side * self.patch_size, self.patches_per_side * self.patch_size)
        # Place each patch in the appropriate location in the full image
        patch_index = 0
        for i in range(self.patches_per_side):
            for j in range(self.patches_per_side):
                patch_img = patch_outputs[patch_index]  # [B,1,patch_size,patch_size]
                y0 = i * self.patch_size
                x0 = j * self.patch_size
                full_images[:, :, y0:y0+self.patch_size, x0:x0+self.patch_size] = patch_img
                patch_index += 1
        # full_images shape: [B, 1, image_size, image_size]
        return full_images

# Critic (WGAN discriminator): outputs Wasserstein score for input images
class Critic(nn.Module):
    def __init__(self, img_size=28):
        super(Critic, self).__init__()
        # Simple CNN for discriminator
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)  # out: 32 x (img_size/2) x (img_size/2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # out: 64 x (img_size/4) x (img_size/4)
        # Compute conv output size similar to encoder
        conv_out = img_size
        conv_out = (conv_out + 2*1 - 4) // 2 + 1
        conv_out = (conv_out + 2*1 - 4) // 2 + 1
        conv_feat_dim = 64 * conv_out * conv_out
        self.fc = nn.Linear(conv_feat_dim, 64)
        self.out_layer = nn.Linear(64, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        # x: [B, 1, H, W]
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.act(self.fc(x))
        out = self.out_layer(x)
        return out  # no activation, can output negative or positive (WGAN score)

# Instantiate models
encoder = Encoder(latent_dim=args.latent_dim, img_size=args.image_size)
generator = QuantumGenerator(n_patches=args.n_patches, patch_size=args.patch_size, 
                             n_qubits=args.n_qubits, latent_dim=args.latent_dim)
critic = Critic(img_size=args.image_size)

print(f"Backend available gpu: {torch.cuda.is_available()}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
encoder.to(device)
generator.to(device)
critic.to(device)

# Optimizers for encoder, generator, and critic
opt_enc   = optim.Adam(encoder.parameters(), lr=args.lr, betas=(0.5, 0.9))
opt_gen   = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.9))
opt_critic= optim.Adam(critic.parameters(), lr=args.lr, betas=(0.5, 0.9))

# -----------------------
# Training loop
# -----------------------
def reparameterize(mu, logvar):
    """Sample latent vector z from Gaussian(mu, sigma) via reparameterization."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# For tracking training losses
history = {"recon_loss": [], "kl_loss": [], "gen_adv_loss": [], "critic_loss": []}

print("Starting training...")
start_time = time.time()
for epoch in range(1, args.epochs+1):
    encoder.train(); generator.train(); critic.train()
    epoch_recon = 0.0
    epoch_kl = 0.0
    epoch_gen_adv = 0.0
    epoch_disc = 0.0
    num_batches = 0
    # Progress bar for batches within the current epoch
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
    for (real_imgs, _) in pbar:
        print(f"Processing batch {num_batches+1}...")
        num_batches += 1
        batch_size = real_imgs.size(0)
        # Move real images to [-1,1] range (already normalized by transform)
        real_imgs = real_imgs.to(device) # put the input tensors on the correct device
        # ---------------------
        # Encode and generate
        # ---------------------
        mu, logvar = encoder(real_imgs)
        z = reparameterize(mu, logvar)   # sample latent vector from encoder distribution
        fake_imgs = generator(z.detach())  # generate fake images; detach z to prevent encoder gradients from adversarial loss
        fake_imgs = fake_imgs.to(device)    # put the generated images on the correct device
        # (We detach z to dissociate QGAN gradients from encoder:contentReference[oaicite:12]{index=12}, so encoder only gets VAE loss)
        # ---------------------
        # Critic updates (n_critic iterations)
        # ---------------------
        # Critic inner-loop progress bar (nested). Shows progress of n_critic updates for this batch.
        # critic_pbar = tqdm(range(args.n_critic), desc="Critic updates", leave=False)
        print(f"Starting critic updates...{args.n_critic}")
        for t in range(args.n_critic):
            print(f"Starting critic update {t+1}/{args.n_critic}...")
            # Sample real and fake scores
            real_scores = critic(real_imgs)        # D(x)
            fake_scores = critic(fake_imgs.detach())  # D(G(z)), detach G to not update generator in critic loop
            # WGAN-GP loss components
            wasserstein_loss = fake_scores.mean() - real_scores.mean()  # (D(fake) - D(real))
            # Gradient penalty (compute only at specified intervals for speed)
            gp = 0.0
            if (t % args.gp_every == 0):
                # Interpolate between real and fake for GP
                alpha = torch.rand(batch_size, 1, 1, 1).to(device)
                interp = alpha * real_imgs + (1 - alpha) * fake_imgs.detach()
                interp.requires_grad_(True)
                interp_scores = critic(interp)
                # Compute gradient w.rt. interpolated images
                grad = torch.autograd.grad(outputs=interp_scores, inputs=interp,
                                            grad_outputs=torch.ones_like(interp_scores),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
                # Compute L2 norm of gradients
                grad = grad.view(batch_size, -1)
                grad_norm = torch.sqrt(torch.sum(grad**2, dim=1) + 1e-8)
                gp = ((grad_norm - 1) ** 2).mean()
            # Critic total loss
            critic_loss = wasserstein_loss + args.lambda_gp * gp
            # Optimize critic
            opt_critic.zero_grad()
            critic_loss.backward()
            opt_critic.step()
            # critic_pbar.set_postfix({"loss": f"{critic_loss.item():.4f}", "t": t})
        # After n_critic updates, update encoder and generator once
        # ---------------------
        # Encoder update (VAE loss only)
        # ---------------------
        # Compute VAE losses: reconstruction and KL divergence
        print("1...")
        recon_imgs_enc = generator(z)  # generator with *non-detached* z for encoder's reconstruction gradient
        print("2...")
        recon_loss = F.mse_loss(recon_imgs_enc, real_imgs, reduction='mean')  # L2 reconstruction loss
        print("3...")
        # KL divergence between q(z|x) and p(z) ~ N(0,1) for each sample (closed form)
        kl_loss = 0.5 * torch.sum(mu**2 + torch.exp(logvar) - logvar - 1) / batch_size
        print("4...")
        enc_loss = recon_loss + kl_loss
        print("5...")
        opt_enc.zero_grad()
        print("6...")
        enc_loss.backward()  # retain graph as generator graph reused for its update
        print("7...")
        opt_enc.step()
        print("8...")
        # ---------------------
        # Generator update (weighted recon + adversarial loss)
        # ---------------------
        # Compute generator adversarial loss: maximize D(fake) => minimize -D(fake)
        recon_imgs_gen = generator(z.detach())  # no grad into encoder
        print("9...")
        gen_adv_loss = -critic(recon_imgs_gen).mean()
        print("10...")
        gen_loss = args.gamma * F.mse_loss(recon_imgs_gen, real_imgs, reduction='mean') + gen_adv_loss
        print("11...")
        opt_gen.zero_grad()
        print("12...")
        gen_loss.backward()
        print("13...")
        opt_gen.step()
        print("14...")
        # ---------------------
        # Accumulate losses for reporting
        # ---------------------
        epoch_recon   += recon_loss.item()
        print("15...")
        epoch_kl      += kl_loss.item()
        print("16...")
        epoch_gen_adv += gen_adv_loss.item()
        print("17...")
        epoch_disc    += critic_loss.item()
        print("18...")
        # Update progress bar with running averages
        pbar.set_postfix({
            "recon": f"{epoch_recon/num_batches:.4f}",
            "kl": f"{epoch_kl/num_batches:.4f}",
            "gen_adv": f"{epoch_gen_adv/num_batches:.4f}",
            "critic": f"{epoch_disc/num_batches:.4f}"
        })
    # End of epoch
    epoch_recon   /= num_batches
    epoch_kl      /= num_batches
    epoch_gen_adv /= num_batches
    epoch_disc    /= num_batches
    history["recon_loss"].append(epoch_recon)
    history["kl_loss"].append(epoch_kl)
    history["gen_adv_loss"].append(epoch_gen_adv)
    history["critic_loss"].append(epoch_disc)
    elapsed = time.time() - start_time
    print(f"Epoch {epoch}/{args.epochs} - Recon: {epoch_recon:.4f}, KL: {epoch_kl:.4f}, Gen Adv: {epoch_gen_adv:.4f}, Critic: {epoch_disc:.4f} - time {elapsed:.1f}s")

# Save model checkpoints
if args.save_model:
    torch.save({
        "encoder": encoder.state_dict(),
        "generator": generator.state_dict(),
        "critic": critic.state_dict()
    }, os.path.join(args.output_dir, "vae_qwgan_final.pth"))
    print(f"Model checkpoints saved to {args.output_dir}/vae_qwgan_final.pth")

# -----------------------
# Inference and Evaluation
# -----------------------
encoder.eval(); generator.eval(); critic.eval()

# Function to reconstruct a batch of images using encoder & generator
def reconstruct_images(input_images):
    """Encode input images to latent space and reconstruct them via the generator."""
    with torch.no_grad():
        mu, logvar = encoder(input_images)
        z = mu  # use mean for reconstruction (deterministic recon)
        recon_imgs = generator(z)
    return recon_imgs

# Fit a Gaussian Mixture Model (GMM) on training set latents for prior
print("Fitting GMM to latent space...")
all_latents = []
with torch.no_grad():
    for imgs, _ in train_loader:
        mu, logvar = encoder(imgs)
        z = mu  # take mean as representative latent (could sample as well)
        all_latents.append(z.numpy())
all_latents = np.concatenate(all_latents, axis=0)
n_components = 10  # number of mixture components (e.g., 10 for MNIST digits)
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=args.seed)
gmm.fit(all_latents)
print(f"GMM fitted with {n_components} components on {all_latents.shape[0]} latent vectors.")

def sample_from_gmm(num_samples):
    """Generate new images by sampling latent vectors from the GMM prior and mapping through generator."""
    z_samples = gmm.sample(num_samples)[0]  # shape (num_samples, latent_dim)
    z_samples = torch.tensor(z_samples, dtype=torch.float)
    with torch.no_grad():
        gen_images = generator(z_samples)
    return gen_images

# Generate some example reconstructions from test set
test_iter = iter(test_loader)
test_imgs, _ = next(test_iter)
recon_imgs = reconstruct_images(test_imgs)
# The test and reconstructed images are in [-1,1] range. Convert to [0,1] for visualization:
orig_vis = (test_imgs * 0.5 + 0.5).clamp(0,1)      # de-normalize to [0,1]
recon_vis = (recon_imgs * 0.5 + 0.5).clamp(0,1)    # de-normalize to [0,1]

# Plot and save reconstruction results
fig, axes = plt.subplots(2, min(8, args.batch_size), figsize=(min(8,args.batch_size)*1.5, 3))
for i in range(min(8, args.batch_size)):
    axes[0, i].imshow(orig_vis[i].squeeze().cpu().numpy(), cmap="gray")
    axes[0, i].axis('off')
    axes[1, i].imshow(recon_vis[i].squeeze().cpu().numpy(), cmap="gray")
    axes[1, i].axis('off')
axes[0,0].set_title("Original")
axes[1,0].set_title("Reconstructed")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "reconstructions.png"))
plt.close()
print(f"Saved example reconstructions to {args.output_dir}/reconstructions.png")

# Generate new samples from GMM prior
num_samples = 16
gen_images = sample_from_gmm(num_samples)
gen_vis = (gen_images * 0.5 + 0.5).clamp(0,1)
# Plot and save generated samples
cols = int(math.sqrt(num_samples))
rows = int(math.ceil(num_samples / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
for idx in range(num_samples):
    i, j = divmod(idx, cols)
    axes[i, j].imshow(gen_vis[idx].squeeze().cpu().numpy(), cmap="gray")
    axes[i, j].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "generated_samples.png"))
plt.close()
print(f"Saved generated samples to {args.output_dir}/generated_samples.png")
