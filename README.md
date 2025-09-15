# Stable Diffusion (DDPM) from Scratch with Convolutional Variational Autoencoder latent compression and attention-enhanced U-Net architecture.

In this notebook, I built a Denoising Diffusion Probabilistic Model (DDPM) completely from the ground-up using PyTorch, focusing on unconditional image generation.
My goal was to deeply understand the mathematics and architecture behind Stable Diffusion, including its latent-space compression via a Convolutional Variational Autoencoder (VAE) and attention-enhanced U-Net denoiser.

I discussed the math behind Stable Diffusion [here](https://medium.com/@shovonsharma/the-math-behind-stable-diffusion-232ac2f9f263).

## Key Achievements
#### Complete DDPM Implementation: 
Built the entire diffusion process including forward/reverse sampling.
#### Latent Space Optimization: 
Integrated Conv-VAE to compress images into latent space as well as to generate noise samples for reverse diffusion.
#### Attention-Enhanced U-Net: 
Implemented a time-conditioned U-Net with residual blocks and self-attention layers at multiple resolutions, improving global coherence and sample quality.
#### Mathematical Rigor: 
Implemented proper noise scheduling, DDPM posterior calculations, and reparameterization tricks.

## 🏗️ Architecture
### 1. Variational Autoencoder (VAE)
<u>Purpose</u>: Compress 64×64 images to 8×8 latent representations.
<u>Architecture</u>: Convolutional encoder-decoder with reparameterization trick.
<u>Key Advantage</u>: Spatial latent encoding for better reconstruction and noise generation.

mu, logvar = self.encode(x)  # (B, 4, 8, 8) spatial latents
z = self.reparameterize(mu, logvar)

### 2. Attention-Enhanced U-Net
<u>Multi-scale Processing</u>: Down/up sampling with skip connections.
<u>Self-Attention Blocks</u>: Spatial attention for global context modeling.
<u>Time Conditioning</u>: Sinusoidal time embeddings integrated at each layer.
<u>Residual Design</u>: GroupNorm + SiLU activation for stable training.

### 3. DDPM Sampler
<u>Forward Process</u>: Systematic noise addition with learned β schedule.
<u>Reverse Process</u>: Learned denoising with proper variance calculation.
<u>Mathematical Foundation</u>: Implements full DDPM posterior derivatio.

##  Mathematical Implementation
#### Noise Scheduling:
β_t = linear_schedule(β_start=1e-4, β_end=0.02, T=1000) # Linear β schedule for T timesteps
α_t = 1 - β_t
ᾱ_t = ∏(α_s) for s=1 to t
#### Forward Process (q):
q(x_t | x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t)I)
#### Reverse Process (p_θ)
p_θ(x_{t-1} | x_t) = N(μ_θ(x_t,t), Σ_θ(x_t,t))
<h3>Overview</h3>

## Training Efficiency

#### Two-Stage Training: 
Pretrained the VAE to learn a compact latent representation, then froze its weights and trained the attention-enhanced U-Net for diffusion in latent space
#### Latent Space Benefits: 
64× compression (64²→8²) enables efficient diffusion
#### Gradient Stability: 
Proper initialization and normalization prevent mode collapse

### Forward Diffusion
![Forward Diffusion Process](https://github.com/shovonSharma/Stable-difusion-from-scratch/blob/main/forward%20sample.jpg)

### Conv-VAE
![vae1](https://github.com/shovonSharma/Stable-difusion-from-scratch/blob/main/vae1.jpg)

### Sample images
![ddpm1](https://github.com/shovonSharma/Stable-difusion-from-scratch/blob/main/ddpm1.jpg)
![ddpm2](https://github.com/shovonSharma/Stable-difusion-from-scratch/blob/main/ddpm2.jpg)
![ddpm3](https://github.com/shovonSharma/Stable-difusion-from-scratch/blob/main/ddpm3.jpg)
![ddpm4](https://github.com/shovonSharma/Stable-difusion-from-scratch/blob/main/ddpm4.jpg)
![ddpm5](https://github.com/shovonSharma/Stable-difusion-from-scratch/blob/main/ddpm5.jpg)

### Loss curve
| <img src="https://github.com/shovonSharma/Stable-difusion-from-scratch/blob/main/VAE_trainloss.jpg" width="200"/> | <img src="https://github.com/shovonSharma/Stable-difusion-from-scratch/blob/main/VAE_trainloss.jpg" width="200"/> |
