import os
import re

target_path = r"C:\Users\stdag\AppData\Local\uv\cache\archive-v0\mdpp6F6wi_f8_CcbVUqKb\Lib\site-packages\colbert\search\strided_tensor.py"

if not os.path.exists(target_path):
    print(f"Error: File not found at {target_path}")
    exit(1)

with open(target_path, "r", encoding="utf-8") as f:
    content = f.read()

fallback_code = """
    @classmethod
    def segmented_lookup(cls, tensor, pids, lengths, offsets):
        import torch
        # Pure Python fallback for segmented_lookup
        results = []
        
        lengths_cpu = lengths.cpu().long()
        offsets_cpu = offsets.cpu().long()
        
        for i in range(len(pids)):
            start = offsets_cpu[i].item()
            length = lengths_cpu[i].item()
            results.append(tensor[start : start + length])
            
        return torch.cat(results)
"""

# Remove existing segmented_lookup if present
if "def segmented_lookup" in content:
    print("segmented_lookup exists. Removing old version...")
    
    # Pattern 1: The one returning tuple
    pattern1 = re.compile(r"    @classmethod\s+def segmented_lookup.+?return torch\.cat\(results\), lengths_cpu(\[pids_cpu\])?\n", re.DOTALL)
    
    if pattern1.search(content):
        content = pattern1.sub("", content)
        print("Removed previous tuple-returning implementation.")
    else:
        # Fallback regex
        pattern_generic = re.compile(r"    @classmethod\s+def segmented_lookup.+?return [^\n]+\n", re.DOTALL)
        if pattern_generic.search(content):
            content = pattern_generic.sub("", content)
            print("Removed generic implementation.")

# Now inject the new one
class_def = "class StridedTensor(object):"
if class_def not in content:
    class_def = "class StridedTensor(StridedTensorCore):"

if class_def in content:
    parts = content.split(class_def)
    new_content = parts[0] + class_def + "\n" + fallback_code + parts[1]
    
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Successfully patched strided_tensor.py with new code")
else:
    print("Error: Could not find class StridedTensor definition.")

