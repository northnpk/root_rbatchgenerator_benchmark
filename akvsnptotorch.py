import awkward as ak
import torch
import numpy as np
import time

# Create a sample Awkward Array
awk_array = ak.Array(np.random.rand(1000000))

# Method 1: Awkward Array to NumPy, then NumPy to PyTorch Tensor
start_time = time.time()
np_array = ak.to_numpy(awk_array)
tensor_from_numpy = torch.from_numpy(np_array)
end_time = time.time()

time_numpy_method = end_time - start_time
print(f"Time for Awkward Array to NumPy, then NumPy to PyTorch Tensor: {time_numpy_method} seconds")

# Method 2: Direct Conversion from Awkward Array to PyTorch Tensor
start_time = time.time()
tensor_direct = torch.as_tensor(awk_array)
end_time = time.time()

time_direct_method = end_time - start_time
print(f"Time for Direct Conversion from Awkward Array to PyTorch Tensor: {time_direct_method} seconds")

# Validate the results
assert torch.equal(tensor_from_numpy, tensor_direct), "The tensors are not equal!"
print("The tensors are equal.")
