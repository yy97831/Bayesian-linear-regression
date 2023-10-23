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
alpha = 2.0
beta = 10.0
prior_mean = np.array([0.0, 0.0])
prior_cov = np.array([[alpha, 0.0], [0.0, alpha]])

# Grid for w0 and w1
w0_range = np.linspace(-1, 1, 100)
w1_range = np.linspace(-1, 1, 100)
W0, W1 = np.meshgrid(w0_range, w1_range)
pos = np.dstack((W0, W1))

def update_posterior_analytically(x, y, prior_mean, prior_cov, beta=10.0):
    X_matrix = np.column_stack([np.ones_like(x), x])
    S_N_inv = np.linalg.inv(prior_cov) + beta * X_matrix.T @ X_matrix
    S_N = np.linalg.inv(S_N_inv)
    m_N = S_N @ (np.linalg.inv(prior_cov) @ prior_mean + beta * X_matrix.T @ y)
    return m_N, S_N

stages = [[], [0], [0, 19], list(range(20))]

fig, axs = plt.subplots(len(stages), 3, figsize=(15, 20))
axs[0, 0].axis('off')

for i, num_points in enumerate(stages):
    axs[i, 0].set_xlabel('w0')
    axs[i, 0].set_ylabel('w1')
    axs[i, 1].set_xlabel('w0')
    axs[i, 1].set_ylabel('w1')
    axs[i, 2].set_xlabel('X')
    axs[i, 2].set_ylabel('Y')
    
    x_observed = X[num_points]
    y_observed = Y[num_points]
    
    if not num_points:
        prior_values = multivariate_normal.pdf(pos, mean=prior_mean, cov=prior_cov)
        axs[i, 1].contourf(W0, W1, prior_values, levels=50, cmap='jet')
        axs[i, 1].set_title('Prior')
        for _ in range(6):
            w_sample = np.random.multivariate_normal(prior_mean, prior_cov)
            axs[i, 2].plot(X, w_sample[0] + w_sample[1] * X, lw=1)
        axs[i, 2].set_title('Sample Lines from Prior')
    else:
        likelihood = np.exp(-0.5 * beta * np.sum((y_observed.reshape(-1, 1, 1) - W0 - W1 * x_observed.reshape(-1, 1, 1))**2, axis=0))
        axs[i, 0].contourf(W0, W1, likelihood, levels=50, cmap='jet')
        axs[i, 0].set_title(f'Likelihood (Data Points: {len(num_points)})')
        
        post_mean, post_cov = update_posterior_analytically(x_observed, y_observed, prior_mean, prior_cov)
        posterior = multivariate_normal.pdf(pos, mean=post_mean, cov=post_cov)
        axs[i, 1].contourf(W0, W1, posterior, levels=50, cmap='jet')
        axs[i, 1].set_title('Posterior')
        
        for _ in range(6):
            w_sample = np.random.multivariate_normal(post_mean, post_cov)
            axs[i, 2].plot(X, w_sample[0] + w_sample[1] * X, lw=1)
        axs[i, 2].scatter(x_observed, y_observed, color='blue')
        axs[i, 2].set_title('Sample Lines from Posterior')

plt.tight_layout()
plt.show()
