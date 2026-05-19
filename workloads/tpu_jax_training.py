import jax
import jax.numpy as jnp
import time

def main():
    print(f"Devices: {jax.devices()}")
    key = jax.random.PRNGKey(0)

    # Simulated MLP
    def predict(params, x):
        h = jnp.dot(x, params['w1']) + params['b1']
        h = jax.nn.relu(h)
        return jnp.dot(h, params['w2']) + params['b2']

    def loss_fn(params, x, y):
        preds = predict(params, x)
        return jnp.mean((preds - y) ** 2)

    @jax.jit
    def update(params, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        new_params = {
            k: v - 0.001 * grads[k] for k, v in params.items()
        }
        return new_params, loss

    # Initialize
    batch_size = 8192
    input_dim = 2048
    hidden_dim = 4096
    output_dim = 100

    k1, k2, k3, k4, kx, ky = jax.random.split(key, 6)
    params = {
        'w1': jax.random.normal(k1, (input_dim, hidden_dim)),
        'b1': jax.random.normal(k2, (hidden_dim,)),
        'w2': jax.random.normal(k3, (hidden_dim, output_dim)),
        'b2': jax.random.normal(k4, (output_dim,))
    }

    x = jax.random.normal(kx, (batch_size, input_dim))
    y = jax.random.normal(ky, (batch_size, output_dim))

    print("Starting JAX training loop (300 seconds)...")
    start_time = time.time()
    step = 0
    
    while time.time() - start_time < 300:
        # We process in batches of 100 steps to simulate an epoch
        for _ in range(100):
            params, loss = update(params, x, y)
        loss.block_until_ready()
        
        # Simulate dataloader/checkpoint delay at the end of an "epoch"
        time.sleep(0.5) 
        step += 100
        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss:.4f}, Elapsed: {time.time() - start_time:.1f}s")

    print("Training loop finished.")

if __name__ == "__main__":
    main()