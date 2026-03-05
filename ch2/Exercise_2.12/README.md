# Exercise 2.12: QuaDRiGa Data Generation for LISTA-CE

This repository provides the starter code for Exercise 2.12. In this exercise, you will use the **QuaDRiGa** (Quasi Deterministic Radio Channel Generator) channel model to generate realistic channel datasets for training a Learned ISTA (LISTA) Channel Estimation network.

## Experiment Setup

Instead of using simple statistical models, we use the 3GPP standards-compliant QuaDRiGa channel model to generate site-specific channel coefficients.

*   **Scenarios:** 
    *   Outdoor: 3GPP 38.901 UMi_NLOS (`id = 1`)
    *   Indoor: 3GPP 38.901 Indoor Open (`id = 2`)
*   **Source:** [QuaDRiGa](https://quadriga-channel-model.de/)
*   **Simulation Configuration:**
    *   Center Frequency: 2.655 GHz
    *   Bandwidth: 10 MHz
    *   Antennas: BS 32 elements (Omni/3GPP-3D), UE Single Antenna
    *   Trajectories: Linear user tracks

## What You Need to Do

| Checklist | Details |
| :--- | :--- |
| **Setup** | unzip `quadriga_src.zip` to the current directory ensuring the `quadriga_src` folder is available. |
| **Code** | Open `Exercise2_12_starter.m`. Complete the TODO sections to configure the simulation environment, create layout/tracks, and extract channel coefficients. |
| **Run** | Execute `Exercise2_12_starter.m` in MATLAB to generate the datasets (`Train_data_...mat`, etc.). |
| **Train** | (Optional) Use the generated data to train the LISTA-CE network (referenced in `https://github.com/King-SmallA/LISTA-CE`). |

## Key Differences from Previous Exercises

In previous exercises, we might have used simplified channel models. Here, you are using **QuaDRiGa**, a geometry-based stochastic channel model that follows 3GPP standards. This allows for capturing spatial consistency and time evolution of radio channels.

Your focus in this exercise is on the **data generation** aspect using MATLAB.

## Files

| File | Purpose |
| :--- | :--- |
| `Exercise2_12_starter.m` | Starter MATLAB script for configuring QuaDRiGa and generating data. |
| `quadriga_src.zip` | Source code for the QuaDRiGa channel model (needs to be unzipped). |
| `README.md` | This file. |

> **Note:** Ensure you have the correct path settings in the script for saving the generated `.mat` files. The default paths might need adjustment to your local environment.