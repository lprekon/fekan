# Generate random samples for the Feynman I-9-18 benchmark.

import numpy as np
import json
import sys


def generate_function_data(num_samples):
    samples = []
    for _ in range(num_samples):
        a = np.random.uniform(0, 10)
        b = np.random.uniform(-1, 3)
        c = np.random.uniform(1, 4)
        d = np.random.uniform(-2, 5)
        e = np.random.uniform(2, 6)
        f = np.random.uniform(-3, 7)
        features = [a, b, c, d, e, f]
        label = a / ((b - 1) ** 2 + (c - d) ** 2 + (e - f) ** 2)
        sample = {"features": features, "label": label}  # Store the sample as a dictionary"
        samples.append(sample)
    return samples


# Usage example
num_samples = int(sys.argv[1])
samples = generate_function_data(num_samples)
print(json.dumps(samples))
