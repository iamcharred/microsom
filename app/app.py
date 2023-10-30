import sys
import pickle
import numpy as np
import json
sys.path.append('./src')
import microsom as msom

from flask import Flask, jsonify, request

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

with open('./models/kohonen_som.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

app = Flask(__name__)

@app.route('/api/weights', methods=['GET'])
def get_weights():
    return jsonify(loaded_model.weights.tolist())

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # print(data)
    input_data = np.array([np.fromstring(data['input'], dtype=float, sep=', ')])
    # print("input ", input_data)
    winner = loaded_model.predict(input_data)
    # print(winner)
    response = json.dumps(winner, cls=NpEncoder)
    return jsonify(response), 200

# @app.route('/api/items/<int:item_id>', methods=['GET'])
# def get_item(item_id):
#     item = next((item for item in data if item["id"] == item_id), None)
#     if item is not None:
#         return jsonify(item)
#     return "Item not found", 404

if __name__ == '__main__':
    app.run(debug=False)
