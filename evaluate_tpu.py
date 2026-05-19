import pandas as pd
import numpy as np
from classifier.threeway_improved import fft_dominant_period, fft_peak_power, acf_lag

def analyze_trace(file_path):
    df = pd.read_parquet(file_path)
    # The temporal classifier uses "gpu_utilization_pct" and "mem_used_mb"
    # We will map "tpu_utilization_pct" -> "gpu_utilization_pct"
    # and "tpu_mem_used_gb" * 1024 -> "mem_used_mb"
    
    util = df['tpu_utilization_pct'].values
    mem = df['tpu_mem_used_gb'].values * 1024
    
    print(f"--- Analysis for {file_path} ---")
    print(f"Mean Util: {util.mean():.2f}%")
    print(f"Std Util:  {util.std():.2f}%")
    
    period = fft_dominant_period(util)
    power = fft_peak_power(util)
    print(f"FFT Dominant Period: {period:.2f} samples")
    print(f"FFT Peak Power: {power:.4f}")
    
    acf_1 = acf_lag(util, 1)
    acf_5 = acf_lag(util, 5)
    print(f"ACF(1): {acf_1:.4f}")
    print(f"ACF(5): {acf_5:.4f}")
    print()

if __name__ == "__main__":
    analyze_trace('data/tpu_jax_training.parquet')
    analyze_trace('data/tpu_jax_inference.parquet')
