# Predictive accuracy diverges from physical consistency in physics-informed neural networks

This repository contains representative code and sample data supporting the manuscript:

**Predictive accuracy diverges from physical consistency in physics-informed neural networks**

## Overview

Physics-informed neural networks (PINNs) are often evaluated using predictive accuracy, but accurate predictions do not necessarily indicate physically reliable behavior. This study examines how loss formulation, network architecture, and data availability affect predictive accuracy and physics consistency in PINNs for indoor airflow reconstruction governed by the Navier–Stokes equations.

The repository provides a simplified representative training script and a sample dataset for demonstration.

## Repository contents

- `code/pinn_2d_representative_training.py`  
  Representative Python script for 2D PINN training, including data loss, physics residual loss, and boundary condition loss.

- `data_sample/sample_airflow_data.csv`  
  Sample airflow dataset used for demonstration.

- `requirements.txt`  
  Python package requirements for running the representative script.

## Expected input and output variables

### Inputs
- `x` — x-coordinate
- `y` — y-coordinate
- `invel` — inlet velocity magnitude

### Outputs
- `magvel` — velocity magnitude
- `xvel` — x-velocity component
- `yvel` — y-velocity component
- `press` — pressure

## Usage

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
