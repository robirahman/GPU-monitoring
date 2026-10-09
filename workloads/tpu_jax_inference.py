import jax
import jax.numpy as jnp
import time

def main():
    print(f"Devices: {jax.devices()}")
    key = jax.random.PRNGKey(0)

    def predict(params, x):
        h = jnp.dot(x, params['w1']) + params['b1']
        h = jax.nn.relu(h)
        return jnp.dot(h, params['w2']) + params['b2']

    @jax.jit
    def infer_step(params, x):
        return predict(params, x)

    batch_size = 16384
    input_dim = 2048
    hidden_dim = 4096
    output_dim = 100

    k1, k2, k3, k4, kx = jax.random.split(key, 5)
    params = {
        'w1': jax.random.normal(k1, (input_dim, hidden_dim)),
        'b1': jax.random.normal(k2, (hidden_dim,)),
        'w2': jax.random.normal(k3, (hidden_dim, output_dim)),
        'b2': jax.random.normal(k4, (output_dim,))
    }

    x = jax.random.normal(kx, (batch_size, input_dim))

    print("Starting JAX inference loop (300 seconds)...")
    start_time = time.time()
    step = 0

    while time.time() - start_time < 300:
        # Process continuous forward passes
        for _ in range(50):
            preds = infer_step(params, x)
        preds.block_until_ready()
        
        # Simulate slight network jitter for continuous batch serving
        time.sleep(0.05)
        step += 50
        if step % 1000 == 0:
            print(f"Step {step}, Elapsed: {time.time() - start_time:.1f}s")

    print("Inference loop finished.")

if __name__ == "__main__":
    main()