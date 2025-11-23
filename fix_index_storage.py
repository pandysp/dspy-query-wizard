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

index_storage_path = os.path.join(colbert_path, "search", "index_storage.py")
if not os.path.exists(index_storage_path):
    print(f"Error: File not found at {index_storage_path}")
    exit(1)

print(f"Patching {index_storage_path}...")
with open(index_storage_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Clean up previous mess (empty @classmethod lines)
content = re.sub(r"    @classmethod\s*\n\s*\n", "", content)
content = re.sub(r"    @classmethod\s*\n", "", content)

# 2. Define fallback code
fallback_code = """
    @classmethod
    def filter_pids(cls, pids, centroid_scores, codes, doclens, offsets, idx, ndocs):
        import torch
        return pids

    @classmethod
    def decompress_residuals(cls, pids, doclens, offsets, bucket_weights, reversed_bit_map, decompression_lookup_table, residuals, codes, centroids, dim, nbits):
        import torch
        results = []
        pids_cpu = pids.cpu().long()
        doclens_cpu = doclens.cpu().long()
        offsets_cpu = offsets.cpu().long()
        
        if bucket_weights.device.type != "cpu": bucket_weights = bucket_weights.cpu()
        if reversed_bit_map.device.type != "cpu": reversed_bit_map = reversed_bit_map.cpu()
        if decompression_lookup_table.device.type != "cpu": decompression_lookup_table = decompression_lookup_table.cpu()
        if centroids.device.type != "cpu": centroids = centroids.cpu()
        if residuals.device.type != "cpu": residuals = residuals.cpu()
        if codes.device.type != "cpu": codes = codes.cpu()

        for i in range(len(pids)):
            pid = pids_cpu[i].item()
            start = offsets_cpu[pid].item()
            length = doclens_cpu[pid].item()
            
            c_ = codes[start : start + length]
            r_ = residuals[start : start + length]
            
            centroids_ = centroids[c_.long()]
            r_ = reversed_bit_map[r_.long()]
            r_ = decompression_lookup_table[r_.long()]
            r_ = r_.reshape(r_.shape[0], -1)
            r_ = bucket_weights[r_.long()]
            centroids_ = centroids_ + r_
            
            results.append(centroids_)
            
        return torch.cat(results)

    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        pass
"""

# 3. Insert methods
class_def = "class IndexScorer(IndexLoader, CandidateGeneration):"
if class_def in content:
    parts = content.split(class_def)
    
    # 4. Disable original try_load_torch_extensions to avoid conflict/override
    # We rename it in the original code part (parts[1])
    parts[1] = parts[1].replace("def try_load_torch_extensions(cls, use_gpu):", "def _original_try_load_torch_extensions(cls, use_gpu):")
    
    # We insert at the beginning of the class
    new_content = parts[0] + class_def + "\n" + fallback_code + parts[1]
    
    with open(index_storage_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Applied fix.")
else:
    print("Error: Could not find class IndexScorer.")
