import numpy as np

def prepare_inference(data):
    # Simplest version to match your model's expected 1-feature input
    return np.array([[float(data['year'])]])
