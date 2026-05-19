
import jax
import jax.numpy as jnp
from tpu_info import device, metrics
import threading
import time

def stress():
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (8192, 8192))
    b = jax.random.normal(key, (8192, 8192))
    for i in range(100):
        c = jnp.dot(a, b)
        c.block_until_ready()

def monitor():
    chip_type, _ = device.get_local_chips()
    for _ in range(10):
        try:
            usage = metrics.get_chip_usage(chip_type)
            if usage:
                u = usage[0]
                print(f"Usage Object: {u}")
                print(f"  accel_tpu_utilization_percent: {u.accel_tpu_utilization_percent}")
                print(f"  total_memory_usage_bytes: {u.total_memory_usage_bytes}")
            else:
                print("No usage data")
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(1)

if __name__ == "__main__":
    t_stress = threading.Thread(target=stress)
    t_stress.start()
    
    # Wait a bit for initialization
    time.sleep(2)
    
    monitor()
    t_stress.join()
