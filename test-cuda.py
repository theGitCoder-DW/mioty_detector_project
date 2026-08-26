import cupy as cp

# Data lives on the GPU instead of CPU
a = cp.random.rand(10_000_000)
b = cp.random.rand(10_000_000)

# This runs in parallel on the GPU automatically
c = a + b * 2

print(c[:5])          # pulls just a few values back to CPU to display
result_cpu = cp.asnumpy(c)  # bring the whole array back to CPU/numpy