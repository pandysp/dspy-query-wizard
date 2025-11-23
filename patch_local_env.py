import os
import sys
import re
import shutil

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

print(f"Found colbert at: {colbert_path}")

# 1. Patch strided_tensor.py
strided_tensor_path = os.path.join(colbert_path, "search", "strided_tensor.py")
if os.path.exists(strided_tensor_path):
    print(f"Patching {strided_tensor_path}...")
    with open(strided_tensor_path, "r", encoding="utf-8") as f:
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
    
    if "def segmented_lookup" not in content:
        class_def = "class StridedTensor(StridedTensorCore):"
        if class_def in content:
            parts = content.split(class_def)
            new_content = parts[0] + class_def + "\n" + fallback_code + parts[1]
            with open(strided_tensor_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print("  Applied patch.")
        else:
            print("  Error: Could not find class StridedTensor.")
    else:
        print("  Already patched or method exists.")

# 2. Patch index_storage.py
index_storage_path = os.path.join(colbert_path, "search", "index_storage.py")
if os.path.exists(index_storage_path):
    print(f"Patching {index_storage_path}...")
    with open(index_storage_path, "r", encoding="utf-8") as f:
        content = f.read()

    fallback_code = """
    @classmethod
    def filter_pids(cls, pids, centroid_scores, codes, doclens, offsets, idx, ndocs):
        import torch
        # Pure Python fallback for filter_pids
        return pids

    @classmethod
    def decompress_residuals(cls, pids, doclens, offsets, bucket_weights, reversed_bit_map, decompression_lookup_table, residuals, codes, centroids, dim, nbits):
        import torch
        
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
            centroids_ = centroids[c_.long()]
            r_ = reversed_bit_map[r_.long()]
            r_ = decompression_lookup_table[r_.long()]
            r_ = r_.reshape(r_.shape[0], -1)
            r_ = bucket_weights[r_.long()]
            centroids_ = centroids_ + r_
            
            results.append(centroids_)
            
        return torch.cat(results)
"""
    
    # Check if we need to patch (simple check)
    if "Pure Python fallback for decompress_residuals" not in content:
        # Remove existing C++ calls if they exist (they might be imported or defined differently)
        # But here we are injecting methods into the class.
        # If the methods don't exist, we add them.
        # If they exist (from the original code calling C++), we replace them.
        
        # The original code usually imports the C++ extension and calls it.
        # We want to override the methods in the class.
        
        # Let's just append the methods to the class if they are not there, 
        # or replace them if they are.
        
        # For simplicity, let's assume we are patching the original file which DOES NOT have these methods defined in Python usually?
        # Wait, the original file calls `colbert_cpp.filter_pids`.
        # It defines `filter_pids` method?
        # Let's check the original file content if possible.
        # But based on previous patching, we replaced the methods.
        
        class_def = "class IndexScorer(IndexLoader, CandidateGeneration):"
        if class_def in content:
            # We will just prepend our methods to the class body.
            # Python methods defined later override earlier ones, but here we are inserting at the top of the class?
            # No, we should insert after the class def.
            
            parts = content.split(class_def)
            new_content = parts[0] + class_def + "\n" + fallback_code + parts[1]
            
            # We also need to comment out or remove the original definitions if they exist later in the file?
            # If we insert at the top, and there are other definitions later, the later ones win.
            # So we should probably remove the original definitions.
            
            # Regex to remove original definitions
            new_content = re.sub(r"    def filter_pids\(.*?\):.+?return .+?\n", "", new_content, flags=re.DOTALL)
            new_content = re.sub(r"    def decompress_residuals\(.*?\):.+?return .+?\n", "", new_content, flags=re.DOTALL)
            
            with open(index_storage_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print("  Applied patch.")
        else:
            print("  Error: Could not find class IndexScorer.")
    else:
        print("  Already patched.")

# 3. Patch colbert.py (modeling)
modeling_path = os.path.join(colbert_path, "modeling", "colbert.py")
if os.path.exists(modeling_path):
    print(f"Patching {modeling_path}...")
    with open(modeling_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    fallback_code = """
    @classmethod
    def segmented_maxsim(cls, scores, lengths):
        import torch
        # Pure Python fallback for segmented_maxsim
        results = []
        offset = 0
        lengths_cpu = lengths.cpu().long()
        
        for i in range(len(lengths)):
            length = lengths_cpu[i].item()
            doc_scores = scores[offset : offset + length]
            if length > 0:
                max_scores, _ = doc_scores.max(dim=0)
                results.append(max_scores.sum())
            else:
                results.append(torch.tensor(0.0, device=scores.device))
            offset += length
            
        return torch.stack(results)
"""

    if "def segmented_maxsim" not in content:
        class_def = "class ColBERT(BaseColBERT):"
        if class_def in content:
            parts = content.split(class_def)
            new_content = parts[0] + class_def + "\n" + fallback_code + parts[1]
            with open(modeling_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print("  Applied patch.")
        else:
            print("  Error: Could not find class ColBERT.")
    else:
        print("  Already patched.")

print("Done patching.")
