# Bayesian Linear Regression Example

This Python code demonstrates how Bayesian linear regression works using synthetic data. The code generates plots at different stages of the Bayesian update process, showcasing how the likelihood, prior, and posterior distributions evolve.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- SciPy

You can install these packages using pip:

```bash
pip install numpy matplotlib scipy
```

## Quick Start

To run the code, simply execute the Python script:

```bash
python bayesian_linear_regression_example.py
```

## How It Works

### Import Required Libraries

The code uses NumPy for numerical operations, Matplotlib for plotting, and SciPy for statistical functions.

### Generate Synthetic Data

The code creates synthetic data for linear regression using a true slope (\`true_w1\`) and intercept (\`true_w0\`). Gaussian noise is added to simulate real-world observations.

### Bayesian Update Functions

1. **compute_likelihood**: Given a single data point, this function computes the likelihood of various combinations of slope and intercept (\`w0\` and \`w1\`).

2. **update_posterior**: This function takes all observed data points and updates the posterior distribution for \`w0\` and \`w1\` based on the prior and likelihood.

### Plotting

The code generates contour plots for the likelihood and posterior at different stages of data collection. It also samples lines from these distributions to give a visual sense of the current estimates for \`w0\` and \`w1\`.

## Customizing the Code

- **Number of Data Points (\`N\`)**: You can change the value of \`N\` to use more or fewer data points.
  
- **Initial Prior (\`prior_mean\` and \`prior_cov\`)**: The mean and covariance of the initial prior can be adjusted.

- **Noise Level (\`scale\` in \`np.random.normal\`)**: The amount of noise added to the synthetic data can be changed.

- **Beta (\`beta\` in \`compute_likelihood\` and \`update_posterior\`)**: This is the precision parameter for the Gaussian likelihood function.

---

Feel free to clone this repository and modify the code as per your requirements!
