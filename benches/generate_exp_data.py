# Generate random samples for the Feynman I-9-18 benchmark.

import numpy as np
import json
import sys
import math


def generate_function_data(num_samples):
    samples = []
    for _ in range(num_samples):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        label = np.exp(math.sin(math.pi * x) + y**2)
        features = [x, y]
        sample = {"features": features, "label": label}  # Store the sample as a dictionary"
        samples.append(sample)
    return samples


# Usage example
num_samples = int(sys.argv[1])
samples = generate_function_data(num_samples)
print(json.dumps(samples))
