# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Generate synthetic data
np.random.seed(42)
true_w0, true_w1 = -0.3, 0.5
N = 20
X = np.linspace(-1, 1, N)
Y = true_w0 + true_w1 * X + np.random.normal(scale=0.2, size=N)

# Initial prior
prior_mean = np.array([0.0, 0.0])
prior_cov = np.array([[0.1, 0.0], [0.0, 0.1]])

# Grid for w0 and w1
w0_range = np.linspace(-1, 1, 100)
w1_range = np.linspace(-1, 1, 100)
W0, W1 = np.meshgrid(w0_range, w1_range)
pos = np.dstack((W0, W1))

# Function to compute likelihood for a single data point
def compute_likelihood(x, y, w0_range, w1_range, beta=10.0):
    likelihood = np.ones((len(w0_range), len(w1_range)))
    for i, w0 in enumerate(w0_range):
        for j, w1 in enumerate(w1_range):
            likelihood[i, j] = np.exp(-0.5 * beta * (y - (w0 + w1 * x)) ** 2)
    return likelihood

# Function to update posterior based on all observed data points
def update_posterior(x, y, w0_range, w1_range, prior_mean, prior_cov, beta=10.0):
    posterior = multivariate_normal.pdf(pos, mean=prior_mean, cov=prior_cov)
    for xi, yi in zip(x, y):
        likelihood = compute_likelihood(xi, yi, w0_range, w1_range, beta)
        unnormalized_posterior = likelihood * posterior
        posterior = unnormalized_posterior / np.sum(unnormalized_posterior)
    return posterior

# Explicitly specify the number of data points to observe at each stage
stages = [[], [0], [0, 19], list(range(20))]

# Start plotting
fig, axs = plt.subplots(len(stages), 3, figsize=(15, 20))
axs[0, 0].axis('off')

for i, num_points in enumerate(stages):
    # Add axis labels and titles
    axs[i, 0].set_xlabel('w0')
    axs[i, 0].set_ylabel('w1')
    axs[i, 1].set_xlabel('w0')
    axs[i, 1].set_ylabel('w1')
    axs[i, 2].set_xlabel('X')
    axs[i, 2].set_ylabel('Y')
    
    if not num_points:
        # Prior only (no data observed yet)
        prior_values = multivariate_normal.pdf(pos, mean=prior_mean, cov=prior_cov)
        axs[i, 1].contourf(W0, W1, prior_values, levels=50, cmap='jet')
        axs[i, 1].set_title('Prior')
        # Sample lines from the prior
        for _ in range(6):
            w_sample = np.random.multivariate_normal(prior_mean, prior_cov)
            axs[i, 2].plot(X, w_sample[0] + w_sample[1] * X, lw=1)
        axs[i, 2].set_title('Sample Lines from Prior')
    else:
        # Data points observed so far
        x_observed = X[num_points]
        y_observed = Y[num_points]
        # Plot observed data points
        axs[i, 2].scatter(x_observed, y_observed, color='blue')
        # Compute likelihood based on all observed data points
        likelihood = compute_likelihood(x_observed[-1], y_observed[-1], w0_range, w1_range)
        axs[i, 0].contourf(W0, W1, likelihood, levels=50, cmap='jet')
        axs[i, 0].set_title(f'Likelihood (Data Points: {len(num_points)})')
        # Update and plot posterior
        posterior = update_posterior(x_observed, y_observed, w0_range, w1_range, prior_mean, prior_cov)
        axs[i, 1].contourf(W0, W1, posterior, levels=50, cmap='jet')
        axs[i, 1].set_title('Posterior')
        # Sample lines from the posterior
        for _ in range(6):
            idx = np.unravel_index(np.random.choice(posterior.size, p=posterior.ravel()), posterior.shape)
            w_sample = [w0_range[idx[0]], w1_range[idx[1]]]
            axs[i, 2].plot(X, w_sample[0] + w_sample[1] * X, lw=1)
        axs[i, 2].set_title('Sample Lines from Posterior')

# Show the plots
plt.tight_layout()
plt.show()
