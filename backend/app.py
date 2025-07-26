from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from classifier import AviationClassifier
import json

app = Flask(__name__)
CORS(app)

# Initialize the classifier
classifier = AviationClassifier()

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make prediction
        result = classifier.predict(data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get the list of features that the model expects"""
    try:
        features = classifier.get_feature_names()
        return jsonify({'features': features})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available models"""
    try:
        models = classifier.get_available_models()
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/<model_name>', methods=['POST'])
def switch_model(model_name):
    """Switch to a different model"""
    try:
        success = classifier.load_model(model_name)
        if success:
            return jsonify({'message': f'Successfully switched to {model_name}', 'current_model': model_name})
        else:
            return jsonify({'error': f'Failed to load model {model_name}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Handle batch predictions from CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.csv'):
            # Read CSV file
            df = pd.read_csv(file)
            
            # Get predictions for all rows
            results = []
            for index, row in df.iterrows():
                row_data = row.to_dict()
                prediction = classifier.predict(row_data)
                results.append({
                    'row_index': index,
                    'prediction': prediction['prediction'],
                    'confidence': prediction.get('confidence', 0)
                })
            
            return jsonify({'predictions': results, 'total_rows': len(results)})
        else:
            return jsonify({'error': 'Please upload a CSV file'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Aviation Damage Prediction API is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)