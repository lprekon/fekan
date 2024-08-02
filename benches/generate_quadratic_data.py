# Generate random samples for the Jacobian elliptic functions benchmark.

import numpy as np
from scipy.special import ellipj
import json
import sys


class Sample:
    def __init__(self, features, label):
        self.features = features
        self.label = label


def generate_complex_data(num_samples):
    samples = []
    for _ in range(num_samples):
        x = np.random.uniform(0, 3)  # Random value for the elliptic parameter
        features = [x]  # Store inputs as a list of features
        y = 1.5 * (2.2 * x - 3) ** 2  # Compute the quadratic function
        label = y  # Use the first of the four outputs as the label
        sample = {"features": features, "label": label}  # Store the sample as a dictionary"
        samples.append(sample)
    return samples


# Usage example
num_samples = int(sys.argv[1])
samples = generate_complex_data(num_samples)

# print the data to stdout so it can be piped where it needs to go
print(json.dumps(samples))
