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
    import torch
    print("PyTorch:", torch.__version__)
    if torch.cuda.is_available():
        print("GPUs visible to PyTorch:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    else:
        print("GPUs visible to PyTorch: none, training runs on the CPU")
except ImportError:
    print("PyTorch is not installed in this environment.")

from importlib.metadata import PackageNotFoundError, version #reads the installed version of a package

for name in ("timm", "numpy", "pillow", "h5py", "matplotlib", "flask"):
    try:
        print(f"{name}:", version(name))
    except PackageNotFoundError:
        print(f"{name}: not installed")
