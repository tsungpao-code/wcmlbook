# Exercise 4.12: End-to-End Communication System with Conditional GAN over Rayleigh Channel

This repository provides the starter code for the exercise corresponding to Section 4.2.2. Your task is to learn the distribution of channel outputs by constructing and training a learned end-to-end communication system over a **Rayleigh fading channel** using a **conditional Generative Adversarial Network (GAN)**.

## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Code** | Open `End2EndConvAWGN_starter.py` and modify the base architecture. You need to:<br> &nbsp;• Construct the neural network architecture based on **Table 4.5**.<br> &nbsp;• Implement the training loop for the whole network and the conditional GAN according to **Algorithm 1**.<br> &nbsp;• Replace or adapt the channel simulation to model a **Rayleigh fading channel**.<br> &nbsp;• Write evaluation code to calculate and plot the Bit Error Rate (**BER**) performance. |
| **Run** | Execute: `python End2EndConvAWGN_starter.py`<br>*(Note: Ensure you have installed the required libraries such as `torch`, `numpy`, and `matplotlib` before running).* |
| **Observe** | Observe the adversarial training process of the generator and discriminator. Once trained, the script should generate and display a plot showing the **BER vs. SNR** curve specifically over the SNR range of **[0, 8] dB**. |

## Files
| File | Purpose |
|------|---------|
| `End2EndConvAWGN_starter.py` | Starter script containing the baseline AWGN implementation. You will extend it to include the conditional GAN, Rayleigh channel, and BER plotting logic. |