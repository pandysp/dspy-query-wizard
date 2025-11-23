import os
import sys
from flask import Flask, request, jsonify
from huggingface_hub import snapshot_download

# Ensure we can import colbert
# It should be installed in the environment
try:
    from colbert import Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
except ImportError:
    print("Error: colbert-ai not installed.")
    sys.exit(1)

app = Flask(__name__)

searcher = None

def setup_searcher():
    global searcher
    print("Loading ColBERT index...")
    
    repo_id = "nielsgl/colbert-wiki2017"
    
    print(f"Resolving paths for {repo_id}...")
    try:
        path = snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=["indexes/*", "collection/*"])
    except Exception as e:
        print(f"Error downloading/resolving dataset: {e}")
        sys.exit(1)
    
    index_root = os.path.join(path, "indexes")
    # The index name in the repo is 'wiki17.nbits.2' but sometimes 'wiki17.nbits.local'
    # Check what exists
    if os.path.exists(os.path.join(index_root, "wiki17.nbits.local")):
        index_name = "wiki17.nbits.local"
    else:
        index_name = "wiki17.nbits.2"
        
    # Check if collection is in a subdirectory
    if os.path.exists(os.path.join(path, "collection", "wiki.abstracts.2017", "collection.tsv")):
        collection_path = os.path.join(path, "collection", "wiki.abstracts.2017", "collection.tsv")
    else:
        collection_path = os.path.join(path, "collection", "wiki.abstracts.2017.tsv")
    
    print(f"Index Root: {index_root}")
    print(f"Index Name: {index_name}")
    print(f"Collection: {collection_path}")
    
    # Patch metadata.json if needed
    metadata_path = os.path.join(index_root, index_name, "metadata.json")
    if os.path.exists(metadata_path):
        import json
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            config = metadata.get("config", {})
            checkpoint = config.get("checkpoint", "")
            if checkpoint and "/iris/u/sherylh" in checkpoint:
                print(f"Patching metadata.json checkpoint path (was {checkpoint})...")
                config["checkpoint"] = "colbert-ir/colbertv2.0"
                
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to patch metadata.json: {e}")

    import torch
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    if not os.path.exists(os.path.join(index_root, index_name)):
        print(f"Error: Index not found at {os.path.join(index_root, index_name)}")
        # Check what's in indexes
        print(f"Contents of {index_root}: {os.listdir(index_root)}")
        sys.exit(1)

    # Initialize Searcher
    # We use a context to set the experiment name, but Searcher can take index path directly
    # if we provide index_root.
    
    # Searcher(index=index_name, index_root=index_root, collection=collection_path)
    
    try:
        with Run().context(RunConfig(nranks=1, experiment="colbert-server")):
            searcher = Searcher(index=index_name, index_root=index_root, collection=collection_path)
    except Exception as e:
        print(f"Error initializing Searcher: {e}")
        sys.exit(1)
    
    print("Searcher loaded successfully!")

@app.route("/api/search", methods=["GET"])
def search():
    if not searcher:
        return jsonify({"error": "Searcher not initialized"}), 500

    query = request.args.get("query", "")
    k = int(request.args.get("k", 10))
    
    if not query:
        return jsonify({"error": "Query parameter required"}), 400
        
    # print(f"Searching for: {query}")
    try:
        pids, ranks, scores = searcher.search(query, k=k)
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({"error": str(e)}), 500
    
    passages = [searcher.collection[pid] for pid in pids]
    
    response = []
    for i in range(len(pids)):
        response.append({
            "pid": int(pids[i]),
            "rank": int(ranks[i]),
            "score": float(scores[i]),
            "text": passages[i],
            "prob": 0.0 
        })
        
    return jsonify({
        "query": query,
        "topk": response
    })

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "ready": searcher is not None})

if __name__ == "__main__":
    setup_searcher()
    app.run(host="0.0.0.0", port=2017)
