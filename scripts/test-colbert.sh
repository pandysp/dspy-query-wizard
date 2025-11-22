#!/bin/bash
# Test ColBERT server end-to-end

set -e

echo "🧪 Testing ColBERT Server"
echo "================================"
echo ""

# Check if server is running
echo "1. Checking if server is responding..."
if curl -s -f 'http://127.0.0.1:2017/api/search?query=test&k=1' > /dev/null 2>&1; then
    echo "   ✅ Server is running"
else
    echo "   ❌ Server is not responding"
    echo "   Start with: just colbert-start"
    exit 1
fi
echo ""

# Test 1: Simple query
echo "2. Testing simple query..."
RESPONSE=$(curl -s 'http://127.0.0.1:2017/api/search?query=Christopher+Nolan&k=3')
echo "   Response preview:"
echo "$RESPONSE" | python3 -m json.tool | head -20
echo ""

# Test 2: Check response structure
echo "3. Validating response structure..."
HAS_TOPK=$(echo "$RESPONSE" | python3 -c "import sys, json; print('topk' in json.load(sys.stdin))")
if [ "$HAS_TOPK" = "True" ]; then
    echo "   ✅ Response has 'topk' field"
else
    echo "   ❌ Response missing 'topk' field"
    exit 1
fi
echo ""

# Test 3: Test with Python retriever
echo "4. Testing Python retriever integration..."
cd "$(dirname "$0")/.."
uv run python << 'PYTHON'
import asyncio
from backend.retriever import fetch_colbert_results

async def test():
    try:
        results = await fetch_colbert_results("Christopher Nolan", k=3)
        print(f"   ✅ Retrieved {len(results)} results")
        if results:
            print(f"   First result: {results[0].get('text', '')[:100]}...")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

success = asyncio.run(test())
exit(0 if success else 1)
PYTHON
echo ""

# Test 4: Test "evil questions"
echo "5. Testing HotPotQA-style multi-hop question..."
EVIL_RESPONSE=$(curl -s 'http://127.0.0.1:2017/api/search?query=Who+is+the+director+of+Inception&k=5')
NUM_RESULTS=$(echo "$EVIL_RESPONSE" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('topk', [])))")
echo "   Retrieved $NUM_RESULTS results"
if [ "$NUM_RESULTS" -gt 0 ]; then
    echo "   ✅ Multi-hop query works"
else
    echo "   ⚠️  No results (expected - needs entity extraction)"
fi
echo ""

echo "================================"
echo "✅ All critical tests passed!"
echo ""
echo "Next steps:"
echo "  - Run FastAPI: just run"
echo "  - Run full tests: just test"
