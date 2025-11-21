from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the DSPy Query Wizard API"})

@app.route('/api/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # Placeholder for logic
    return jsonify({
        "question": question,
        "human_answer": "Not implemented yet",
        "machine_answer": "Not implemented yet"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)