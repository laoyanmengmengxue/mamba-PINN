# Learning Long-Term Dynamics from Short-Window Sparse PIV: A Mamba-Integrated Physics-Informed Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation of the paper **"Learning Long-Term Dynamics from Short-Window Sparse PIV: A Mamba-Integrated Physics-Informed Framework"**.

## Introduction

We introduce a novel framework that integrates **State Space Models (Mamba)** with **Physics-Informed Neural Networks (PINNs)**. This approach addresses a critical challenge in fluid dynamics: accurately learning and predicting **long-term evolutionary dynamics** using only **short-window** and **spatially sparse** Particle Image Velocimetry (PIV) data.

By leveraging the Mamba module, the model effectively captures long-range temporal dependencies within the fluid evolution, while PINNs enforce Navier-Stokes equation constraints to ensure physical consistency.

## 📂 Project Structure

The project follows this directory structure:

```text
Project_Root/
├── mainsigle.py          # Core training and prediction script
├── Single data/          # Contains single-frame initial condition data
├── Triple data/          # Contains short-window sequence data for training
├── figure/               # Automatically generated flow field visualizations and loss curves
├── requirements.txt      # Python dependency list
├── .gitignore            # Git exclusion rules
└── README.md             # Project documentation