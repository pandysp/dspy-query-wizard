#!/bin/bash
# Start the local ColBERT server for DSPy Query Wizard
# Uses nielsgl/colbert-server: https://github.com/nielsgl/colbert-server

set -e

PORT="${1:-2017}"
LOG_FILE="${2:-/tmp/colbert-server.log}"

echo "Starting ColBERT server on port $PORT..."
echo "Logs: $LOG_FILE"
echo ""
echo "⚠️  First run will download ~13GB Wikipedia 2017 index from HuggingFace"
echo "    This may take 10-30 minutes depending on your connection."
echo "    Subsequent runs use cached data and start immediately."
echo ""

# Kill any existing colbert-server processes
pkill -f "colbert-server serve" 2>/dev/null || true

# Start the server
# Using nohup to keep it running in background
nohup uvx colbert-server serve \
    --from-cache \
    --port "$PORT" \
    --host 127.0.0.1 \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"
echo ""
echo "📊 Monitor progress:"
echo "    tail -f $LOG_FILE"
echo ""
echo "🛑 Stop server:"
echo "    pkill -f 'colbert-server serve'"
echo ""
echo "✅ Test server (once running):"
echo "    curl 'http://127.0.0.1:$PORT/api/search?query=inception&k=3'"
