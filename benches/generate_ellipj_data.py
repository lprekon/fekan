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
        u = np.random.uniform(0, np.pi / 2)  # Random value for the elliptic parameter
        m = np.random.uniform(0, 1)  # Random value for the modulus parameter
        features = [u, m]  # Store inputs as a list of features
        sn, cn, dn, ph = ellipj(u, m)  # Compute the Jacobian elliptic functions
        label = sn  # Use the first of the four outputs as the label
        sample = {"features": features, "labels": [label]}  # Store the sample as a dictionary"
        samples.append(sample)
    return samples


# Usage example
num_samples = int(sys.argv[1])
samples = generate_complex_data(num_samples)

# print the data to stdout so it can be piped where it needs to go
print(json.dumps(samples))
