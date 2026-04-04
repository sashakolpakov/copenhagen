import os
# Accelerate (used by the C++ extension) and torch both link OpenMP on macOS.
# Allow multiple copies to coexist; without this pytest aborts on import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
