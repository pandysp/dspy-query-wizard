import os
import sys
import re

def get_colbert_path():
    try:
        import colbert
        if hasattr(colbert, '__file__'):
            return os.path.dirname(colbert.__file__)
    except ImportError:
        pass
    
    for path in sys.path:
        candidate = os.path.join(path, 'colbert')
        if os.path.exists(candidate):
            return candidate
    return None

colbert_path = get_colbert_path()
if not colbert_path:
    print("Error: Could not find colbert package.")
    exit(1)

residual_path = os.path.join(colbert_path, "indexing", "codecs", "residual.py")
if not os.path.exists(residual_path):
    print(f"Error: File not found at {residual_path}")
    exit(1)

print(f"Patching {residual_path}...")
with open(residual_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Rename original try_load
content = re.sub(r"def\s+try_load_torch_extensions\s*\(", "def _ignore_try_load_torch_extensions(", content)

# 2. Insert empty try_load
fallback_code = """
    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        pass
"""

class_def = "class ResidualCodec:"
if class_def in content:
    parts = content.split(class_def)
    new_content = parts[0] + class_def + "\n" + fallback_code + parts[1]
    
    with open(residual_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Applied patch to ResidualCodec.")
else:
    print("Error: Could not find class ResidualCodec.")
