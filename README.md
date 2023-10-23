# Bayesian Linear Regression Example

This Python code demonstrates the principle of Bayesian linear regression using synthetic data. It visualizes the evolution of the likelihood, prior, and posterior distributions through various stages of the Bayesian updating process.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- SciPy

To install these packages, use pip:

```bash
pip install numpy matplotlib scipy
```

## Quick Start

To execute the example:

```bash
python bayesian_linear_regression_example.py
```

## How It Works

### Importing Necessary Libraries

The script utilizes NumPy for numerical computations, Matplotlib for visual representation, and SciPy for accessing statistical functions.

### Synthetic Data Generation

We generate synthetic linear regression data, determined by a predefined slope (`true_w1`) and intercept (`true_w0`). Gaussian noise is introduced to emulate real-world data variations.

### Bayesian Update Mechanism

1. **update_posterior_analytically**: This function calculates the analytical posterior distribution for the coefficients `w0` (intercept) and `w1` (slope) given observed data points. The update considers the likelihood arising from new data and the prior distribution.

### Visualization

The visualization includes contour plots showcasing the likelihood and the posterior distribution at distinct stages of data assimilation. Additionally, it draws lines sampled from these distributions, providing a visual perspective on the evolving estimates of `w0` and `w1`.

## Personalizing the Code

- **Data Points Count (`N`)**: Modify the value of `N` to alter the number of data points.
  
- **Initial Prior (`prior_mean` and `prior_cov`)**: The starting mean and covariance for the prior can be adjusted.

- **Noise Intensity (`scale` in `np.random.normal`)**: Regulate the noise level incorporated in the synthetic dataset.

- **Beta (`beta` in `update_posterior_analytically`)**: Represents the precision of the Gaussian likelihood function.
