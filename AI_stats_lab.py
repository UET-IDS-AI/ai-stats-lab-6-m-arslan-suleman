import math
import numpy as np


# -----------------------------
# Bernoulli Log-Likelihood
# -----------------------------
def bernoulli_log_likelihood(data, theta):
    data = np.array(data)

    # Validation
    if data.size == 0:
        raise ValueError("Data cannot be empty")

    if not (0 < theta < 1):
        raise ValueError("Theta must be in (0,1)")

    if not np.all((data == 0) | (data == 1)):
        raise ValueError("Data must contain only 0 and 1")

    # Compute log-likelihood
    return np.sum(data * np.log(theta) + (1 - data) * np.log(1 - theta))


# -----------------------------
# Bernoulli MLE Analysis
# -----------------------------
def bernoulli_mle_with_comparison(data, candidate_thetas=None):
    data = np.array(data)

    # Validation
    if data.size == 0:
        raise ValueError("Data cannot be empty")

    if not np.all((data == 0) | (data == 1)):
        raise ValueError("Data must contain only 0 and 1")

    if candidate_thetas is None:
        candidate_thetas = [0.2, 0.5, 0.8]

    # Counts
    num_successes = int(np.sum(data))
    n = data.size
    num_failures = n - num_successes

    # MLE
    mle = num_successes / n

    # Log-likelihoods
    log_likelihoods = {}
    for theta in candidate_thetas:
        log_likelihoods[theta] = bernoulli_log_likelihood(data, theta)

    # Best candidate (first max)
    best_candidate = max(log_likelihoods, key=log_likelihoods.get)

    return {
        "mle": mle,
        "num_successes": num_successes,
        "num_failures": num_failures,
        "log_likelihoods": log_likelihoods,
        "best_candidate": best_candidate,
    }


# -----------------------------
# Poisson Log-Likelihood
# -----------------------------
def poisson_log_likelihood(data, lam):
    data = np.array(data)

    # Validation
    if data.size == 0:
        raise ValueError("Data cannot be empty")

    if lam <= 0:
        raise ValueError("Lambda must be > 0")

    if not np.all(data >= 0):
        raise ValueError("Data must be nonnegative")

    if not np.all(np.floor(data) == data):
        raise ValueError("Data must be integers")

    # Compute log-likelihood
    return np.sum(data * np.log(lam) - lam - np.vectorize(lambda x: math.lgamma(x + 1))(data))


# -----------------------------
# Poisson MLE Analysis
# -----------------------------
def poisson_mle_analysis(data, candidate_lambdas=None):
    data = np.array(data)

    # Validation
    if data.size == 0:
        raise ValueError("Data cannot be empty")

    if not np.all(data >= 0):
        raise ValueError("Data must be nonnegative")

    if not np.all(np.floor(data) == data):
        raise ValueError("Data must be integers")

    if candidate_lambdas is None:
        candidate_lambdas = [1.0, 3.0, 5.0]

    # Stats
    n = data.size
    total_count = int(np.sum(data))
    sample_mean = total_count / n

    # MLE
    mle = sample_mean

    # Log-likelihoods
    log_likelihoods = {}
    for lam in candidate_lambdas:
        log_likelihoods[lam] = poisson_log_likelihood(data, lam)

    # Best candidate
    best_candidate = max(log_likelihoods, key=log_likelihoods.get)

    return {
        "mle": mle,
        "sample_mean": sample_mean,
        "total_count": total_count,
        "n": n,
        "log_likelihoods": log_likelihoods,
        "best_candidate": best_candidate,
    }
