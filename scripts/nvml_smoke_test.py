from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName, nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage,
    nvmlDeviceGetTemperature, NVML_TEMPERATURE_GPU
)

nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)

name = nvmlDeviceGetName(h)
util = nvmlDeviceGetUtilizationRates(h)
mem = nvmlDeviceGetMemoryInfo(h)
power_mw = nvmlDeviceGetPowerUsage(h)
temp = nvmlDeviceGetTemperature(h, NVML_TEMPERATURE_GPU)

print("GPU:", name)
print("Utilization: gpu=%d%% mem=%d%%" % (util.gpu, util.memory))
print("Memory: used=%d MB total=%d MB" % (mem.used // (1024**2), mem.total // (1024**2)))
print("Power: %.1f W" % (power_mw / 1000.0))
print("Temp: %d C" % temp)

nvmlShutdown()
