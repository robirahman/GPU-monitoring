
import jax
import jax.numpy as jnp
import time

def stress():
    print(f"Devices: {jax.devices()}")
    
    # Large tensors
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (8192, 8192))
    b = jax.random.normal(key, (8192, 8192))
    
    print("Starting JAX stress loop...")
    for i in range(200):
        c = jnp.dot(a, b)
        c.block_until_ready()
        if i % 20 == 0:
            print(f"Iteration {i}")
    
    print("Done.")

if __name__ == "__main__":
    stress()
