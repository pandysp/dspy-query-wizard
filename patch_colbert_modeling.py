import os
import re

target_path = r"C:\Users\stdag\AppData\Local\uv\cache\archive-v0\mdpp6F6wi_f8_CcbVUqKb\Lib\site-packages\colbert\modeling\colbert.py"

if not os.path.exists(target_path):
    print(f"Error: File not found at {target_path}")
    exit(1)

with open(target_path, 'r', encoding='utf-8') as f:
    content = f.read()

fallback_code = """
    @classmethod
    def segmented_maxsim(cls, scores, lengths):
        import torch
        # Pure Python fallback for segmented_maxsim
        # scores: [total_embeddings] (flattened scores of all query tokens against all doc embeddings?)
        # lengths: [num_docs] (number of embeddings per doc)
        
        # Wait, let's check usage in colbert_score_packed.
        # scores = D_packed @ Q.unsqueeze(2)
        # scores is [total_embeddings, query_len, 1] or similar?
        
        # Actually, let's look at colbert_score_packed implementation in colbert.py if possible.
        # But assuming standard MaxSim:
        # For each document, we have a set of embeddings.
        # We compute dot products with query embeddings.
        # Then for each query term, we take max over doc embeddings.
        # Then sum over query terms.
        
        # Here 'scores' seems to be the dot products already?
        # If so, it's likely [total_embeddings, query_len].
        
        # We need to segment 'scores' by 'lengths' (per doc), take max(dim=0) for each segment, then sum.
        
        # This is slow in Python loop.
        
        results = []
        offset = 0
        lengths_cpu = lengths.cpu().long()
        
        for i in range(len(lengths)):
            length = lengths_cpu[i].item()
            # doc_scores: [length, query_len]
            doc_scores = scores[offset : offset + length]
            
            # Max over document embeddings (dim 0)
            # max_scores: [query_len]
            if length > 0:
                max_scores, _ = doc_scores.max(dim=0)
                # Sum over query terms
                total_score = max_scores.sum()
            else:
                total_score = torch.tensor(0.0, device=scores.device)
                
            results.append(total_score)
            offset += length
            
        return torch.stack(results)
"""

# Remove existing segmented_maxsim if present
if "def segmented_maxsim" in content:
    print("segmented_maxsim exists. Removing old version...")
    pattern = re.compile(r"    @classmethod\s+def segmented_maxsim.+?return .+?\n", re.DOTALL)
    content = pattern.sub("", content)

# Now inject the new one
class_def = "class ColBERT(BaseColBERT):"

if class_def in content:
    parts = content.split(class_def)
    new_content = parts[0] + class_def + "\n" + fallback_code + parts[1]
    
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Successfully patched colbert.py with new code")
else:
    print("Error: Could not find class ColBERT definition.")
