"""
check_env.py
 
Prints the machine and library details needed for the Tools and Technologies section.
Read-only: it only prints results.
 
Run it with the same Python environment you use for training:
    python check_env.py
"""
import platform
import sys
 
print("Python:", sys.version.split()[0])
print("Operating system:", platform.platform())
print("Processor (generic string):", platform.processor())
 
try:
    import psutil
    print("RAM (GB):", round(psutil.virtual_memory().total / 1e9, 1))
    print("CPU cores (physical, logical):", psutil.cpu_count(logical=False), psutil.cpu_count(logical=True))
except ImportError:
    print("psutil is not installed. Run: pip install psutil   (or read RAM and cores from Task Manager)")
 
try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
    print("GPUs visible to TensorFlow:", tf.config.list_physical_devices("GPU"))
    try:
        import keras
        print("Keras:", keras.__version__)
    except Exception as e:
        print("Keras version not read:", e)
except ImportError:
    print("TensorFlow is not installed in this environment.")
 
for name in ("keras_hub", "numpy", "PIL"):
    try:
        module = __import__(name)
        print(f"{name}:", getattr(module, "__version__", "version not available"))
    except ImportError:
        print(f"{name}: not installed")
 