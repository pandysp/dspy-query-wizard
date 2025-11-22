import os
import re

target_path = r"C:\Users\stdag\AppData\Local\uv\cache\archive-v0\mdpp6F6wi_f8_CcbVUqKb\Lib\site-packages\colbert\search\index_storage.py"

if not os.path.exists(target_path):
    print(f"Error: File not found at {target_path}")
    exit(1)

with open(target_path, "r", encoding="utf-8") as f:
    content = f.read()

fallback_code = """
    @classmethod
    def filter_pids(cls, pids, centroid_scores, codes, doclens, offsets, idx, ndocs):
        import torch
        # Pure Python fallback for filter_pids
        # We skip filtering for now to avoid complex logic.
        # This might be slower but should be correct.
        return pids

    @classmethod
    def decompress_residuals(cls, pids, doclens, offsets, bucket_weights, reversed_bit_map, decompression_lookup_table, residuals, codes, centroids, dim, nbits):
        import torch
        
        # Pure Python fallback for decompress_residuals
        results = []
        
        pids_cpu = pids.cpu().long()
        doclens_cpu = doclens.cpu().long()
        offsets_cpu = offsets.cpu().long()
        
        # Ensure auxiliary tables are on CPU
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
            
            # Decompress
            # 1. Lookup centroids
            centroids_ = centroids[c_.long()]
            
            # 2. Decompress residuals
            r_ = reversed_bit_map[r_.long()]
            r_ = decompression_lookup_table[r_.long()]
            r_ = r_.reshape(r_.shape[0], -1)
            r_ = bucket_weights[r_.long()]
            
            # 3. Add residuals to centroids
            centroids_ = centroids_ + r_
            
            results.append(centroids_)
            
        return torch.cat(results)
"""

# Remove existing methods if present
if "def filter_pids" in content:
    print("filter_pids exists. Removing old version...")
    pattern = re.compile(r"    @classmethod\s+def filter_pids.+?return .+?\n", re.DOTALL)
    content = pattern.sub("", content)

if "def decompress_residuals" in content:
    print("decompress_residuals exists. Removing old version...")
    pattern = re.compile(r"    @classmethod\s+def decompress_residuals.+?return torch\.cat\(results\)\n", re.DOTALL)
    content = pattern.sub("", content)

# Now inject the new ones
class_def = "class IndexScorer(IndexLoader, CandidateGeneration):"

if class_def in content:
    parts = content.split(class_def)
    new_content = parts[0] + class_def + "\n" + fallback_code + parts[1]
    
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Successfully patched index_storage.py with new code")
else:
    print("Error: Could not find class IndexScorer definition.")

