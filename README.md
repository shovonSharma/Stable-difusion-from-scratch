<h2> Stable Diffusion DDPM from Scratch</h2>

In this notebook i implemented a Denoising Diffusion Probabilistic Model (DDPM) from scratch using PyTorch for unconditional image generation.

<h3>Overview</h3>

This project builds a DDPM to generate images from noise using an attention-based U-Net. Trained on a Pokémon dataset (~800 images, 64x64), it’s a work in progress to improve image quality by addressing noisy outputs.

<h3>Features</h3>
-Attention-based U-Net with sinusoidal time embeddings for reverse diffusion
-Linear noise schedule (num_training_steps=100, beta_start=0.0001, beta_end=0.002).(for simplicity)
-Training with AdamW optimizer and MSE loss.
-Debugging noisy outputs via loss analysis and intermediate visualizations.

<h3>Forward-Diffusion</h3>
![Forward Diffusion Process](https://github.com/shovonSharma/Stable-difusion-from-scratch/blob/main/forward-pass.jpg)

I discussed the math behind Stable Diffusion [here](https://medium.com/@shovonsharma/the-math-behind-stable-diffusion-232ac2f9f263).
