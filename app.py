from flask import Flask, request, jsonify
from model_script import nn_predict  # Import your nn_predict function

app = Flask(__name__)

@app.route('/classify_ticket', methods=['POST'])
def classify_ticket():
    data = request.get_json()
    ticket_description = data['description']
    department = nn_predict(ticket_description)
    return jsonify({"department": department})

if __name__ == '__main__':
    app.run(debug=True)
