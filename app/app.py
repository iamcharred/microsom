import sys
import pickle
import json
import numpy as np
from flask import Flask, jsonify, request

sys.path.append('./src')

class NpEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for numpy objects.
    """
    def default(self, obj):
        """
        Convert numpy objects to Python objects for JSON serialization.

        Args:
            obj (Any): The object to be serialized.

        Returns:
            Any: The serialized object.

        Raises:
            TypeError: If the object cannot be serialized.
        """
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
def get_weights() -> str:
    """
    Get the weights of the loaded model.

    Returns:
        str: JSON string representing the weights of the loaded model.
    """
    return jsonify(loaded_model.weights.tolist())

@app.route('/api/predict', methods=['POST'])
def predict() -> str:
    """
    Predict the winner based on the input data.

    Returns:
        str: JSON string representing the predicted winner.
    """
    data = request.get_json()
    input_data = np.array([np.fromstring(data['input'], dtype=float, sep=', ')])
    print(input_data)
    winner = loaded_model.predict(input_data)
    response = json.dumps(winner, cls=NpEncoder)
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=False)
