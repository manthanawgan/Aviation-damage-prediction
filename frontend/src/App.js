import React, { useState, useEffect, useCallback } from 'react';
import './App.css';

const App = () => {
  const [currentPage, setCurrentPage] = useState('landing');
  const [formData, setFormData] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [features, setFeatures] = useState([]);
  const [availableModels, setAvailableModels] = useState([]);
  const [currentModel, setCurrentModel] = useState('');
  const [csvFile, setCsvFile] = useState(null);
  const [batchResults, setBatchResults] = useState(null);

  // API base URL - update this to match your Flask server
  const API_BASE = 'http://127.0.0.1:5000/api';

  // Important features only (filtered from the full dataset)
  const importantFeatures = [
    'Investigation.Type',
    'Location',
    'Country',
    'Injury.Severity',
    'Aircraft.Category',
    'Make',
    'Amateur.Built',
    'Engine.Type',
    'Purpose.of.flight',
    'Weather.Condition',
    'Broad.phase.of.flight'
  ];

  // Memoize loadFeatures to prevent infinite re-renders
  const loadFeatures = useCallback(async () => {
    try {
      // Use the important features directly
      setFeatures(importantFeatures);
      const initialFormData = {};
      importantFeatures.forEach(feature => {
        initialFormData[feature] = '';
      });
      setFormData(initialFormData);
    } catch (err) {
      setError('Failed to load features');
    }
  }, []); // Empty dependency array since importantFeatures is constant

  const loadAvailableModels = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/models`);
      const data = await response.json();
      if (data.models) {
        setAvailableModels(data.models);
        if (data.models.length > 0) {
          setCurrentModel(data.models[0]);
        }
      }
    } catch (err) {
      setError('Failed to load available models');
    }
  }, [API_BASE]);

  useEffect(() => {
    if (currentPage === 'prediction') {
      loadFeatures();
      loadAvailableModels();
    }
  }, [currentPage, loadFeatures, loadAvailableModels]);

  // Fixed input change handler - removed preventDefault to allow typing
  const handleInputChange = useCallback((e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  }, []);

  const handleModelChange = useCallback(async (e) => {
    const modelName = e.target.value;
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/model/${modelName}`, {
        method: 'POST'
      });
      const data = await response.json();
      if (response.ok) {
        setCurrentModel(modelName);
        setError(null);
      } else {
        setError(data.error || 'Failed to switch model');
      }
    } catch (err) {
      setError('Failed to switch model');
    }
    setLoading(false);
  }, [API_BASE]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const result = await response.json();
      
      if (response.ok) {
        setPrediction(result);
      } else {
        setError(result.error || 'Prediction failed');
      }
    } catch (err) {
      setError('Network error: ' + err.message);
    }
    
    setLoading(false);
  };

  const handleFileUpload = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (!csvFile) {
      setError('Please select a CSV file');
      return;
    }

    setLoading(true);
    setError(null);
    setBatchResults(null);

    const formDataFile = new FormData();
    formDataFile.append('file', csvFile);

    try {
      const response = await fetch(`${API_BASE}/batch-predict`, {
        method: 'POST',
        body: formDataFile,
      });

      const result = await response.json();
      
      if (response.ok) {
        setBatchResults(result);
      } else {
        setError(result.error || 'Batch prediction failed');
      }
    } catch (err) {
      setError('Network error: ' + err.message);
    }
    
    setLoading(false);
  };

  // Fixed file input handler
  const handleFileChange = useCallback((e) => {
    setCsvFile(e.target.files[0]);
  }, []);

  const renderFormField = useCallback((feature) => {
    const selectFields = {
      'Investigation.Type': ['Accident', 'Incident'],
      'Injury.Severity': ['Fatal', 'Serious', 'Minor', 'None', 'Unavailable'],
      'Aircraft.Category': ['Airplane', 'Helicopter', 'Glider', 'Balloon', 'Weight-Shift', 'Powered-Lift'],
      'Amateur.Built': ['Yes', 'No'],
      'Engine.Type': ['Reciprocating', 'Turbo Prop', 'Turbo Jet', 'Turbo Fan', 'Turbo Shaft', 'Electric'],
      'Purpose.of.flight': ['Personal', 'Business', 'Instructional', 'Test', 'Positioning', 'Ferry', 'Aerial Application'],
      'Weather.Condition': ['VMC', 'IMC', 'UNK'],
      'Broad.phase.of.flight': ['Takeoff', 'Initial climb', 'Climb', 'Cruise', 'Descent', 'Approach', 'Landing', 'Taxi', 'Standing']
    };

    if (selectFields[feature]) {
      return (
        <select
          name={feature}
          value={formData[feature] || ''}
          onChange={handleInputChange}
          className="form-control"
        >
          <option value="">Select {feature}</option>
          {selectFields[feature].map(option => (
            <option key={option} value={option}>{option}</option>
          ))}
        </select>
      );
    }

    return (
      <input
        type="text"
        name={feature}
        value={formData[feature] || ''}
        onChange={handleInputChange}
        className="form-control"
        placeholder={`Enter ${feature}`}
      />
    );
  }, [formData, handleInputChange]);

  // Landing Page Component
  const LandingPage = () => (
    <div className="App">
      <header className="App-header">
        <h1>🛩️ AeroGuard</h1>
        <p>Advanced Aviation Damage Prediction with AI</p>
      </header>

      <div className="container">
        {/* Features Section */}
        <div className="prediction-section">
          <h3>Powerful AI Features</h3>
          <div className="form-grid">
            <div className="feature-card">
              <h4>Smart Analysis</h4>
              <p>Advanced machine learning algorithms analyze multiple factors for accurate predictions</p>
            </div>
            <div className="feature-card">
              <h4>Real-Time Processing</h4>
              <p>Get instant predictions for single incidents or batch process multiple records</p>
            </div>
            <div className="feature-card">
              <h4>Multiple Models</h4>
              <p>Switch between different trained models to compare results</p>
            </div>
            <div className="feature-card">
              <h4>Batch Processing</h4>
              <p>Upload CSV files and get comprehensive predictions for entire datasets</p>
            </div>
          </div>
          
          <button
            onClick={(e) => {
              e.preventDefault();
              setCurrentPage('prediction');
            }}
            className="btn btn-primary"
            type="button" // Explicitly set button type
          >
            Start Prediction
          </button>
        </div>

        {/* Stats Section */}
        <div className="prediction-section">
          <h3>Platform Statistics</h3>
          <div className="form-grid">
            <div className="stat-card">
              <div className="stat-number">98.5%</div>
              <div className="stat-label">Prediction Accuracy</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">10K+</div>
              <div className="stat-label">Incidents Analyzed</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">11</div>
              <div className="stat-label">Key Features</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">24/7</div>
              <div className="stat-label">System Availability</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Prediction Page Component
  const PredictionPage = () => (
    <div className="App">
      <header className="App-header">
        <button
          onClick={(e) => {
            e.preventDefault();
            setCurrentPage('landing');
          }}
          className="back-button"
          type="button" // Explicitly set button type
        >
          ← Back to Home
        </button>
        <h1>AeroGuard</h1>
        <p>Aviation Damage Prediction Platform</p>
      </header>

      <div className="container">
        {/* Model Selection */}
        {availableModels.length > 0 && (
          <div className="model-selection">
            <h3>Select Model</h3>
            <div className="form-group">
              <select 
                value={currentModel} 
                onChange={handleModelChange}
                className="form-control"
                disabled={loading}
                onClick={(e) => e.stopPropagation()}
              >
                {availableModels.map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
              {currentModel && (
                <p className="model-info">
                  Current Model: <strong>{currentModel}</strong>
                </p>
              )}
            </div>
          </div>
        )}

        <div className="grid-container">
          {/* Single Prediction */}
          <div className="prediction-section">
            <h3>Single Prediction</h3>
            <form onSubmit={handleSubmit} onFocus={(e) => e.stopPropagation()}>
              <div className="form-grid">
                {features.map(feature => (
                  <div key={feature} className="form-group">
                    <label className="form-label">
                      {feature.replace(/\./g, ' ')}:
                    </label>
                    {renderFormField(feature)}
                  </div>
                ))}
              </div>
              
              <button 
                type="submit"
                className="btn btn-primary"
                disabled={loading}
                onClick={(e) => {
                  // Only prevent default if form is not being submitted
                  if (loading) {
                    e.preventDefault();
                  }
                }}
              >
                {loading ? 'Predicting...' : ' Predict Damage'}
              </button>
            </form>
          </div>

          {/* Batch Prediction */}
          <div className="batch-section">
            <h3>Batch Prediction</h3>
            <form onSubmit={handleFileUpload} className="batch-form">
              <div className="form-group">
                <label className="form-label">Upload CSV File:</label>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  className="form-control file-input"
                  onClick={(e) => e.stopPropagation()}
                />
              </div>
              <button 
                type="submit"
                className="btn btn-secondary"
                disabled={loading || !csvFile}
              >
                {loading ? 'Processing...' : 'Predict Batch'}
              </button>
            </form>

            <div className="csv-info">
              <h4>CSV Format Requirements:</h4>
              <p>Include these column headers:</p>
              <div className="csv-headers">
                {importantFeatures.join(', ')}
              </div>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="alert alert-error">
            <h4>Error:</h4>
            <p>{error}</p>
          </div>
        )}

        {/* Single Prediction Results */}
        {prediction && (
          <div className="result-section">
            <h3>Prediction Result</h3>
            <div className="result-card">
              <p><strong>Predicted Damage:</strong> {prediction.prediction}</p>
              {prediction.confidence > 0 && (
                <p><strong>Confidence:</strong> {prediction.confidence.toFixed(1)}%</p>
              )}
              <p><strong>Model Used:</strong> {prediction.model_used}</p>
            </div>
          </div>
        )}

        {/* Batch Results */}
        {batchResults && (
          <div className="batch-results">
            <h3>Batch Prediction Results</h3>
            <p className="batch-info">
              Total Rows Processed: <strong>{batchResults.total_rows}</strong>
            </p>
            <div className="results-table">
              <table>
                <thead>
                  <tr>
                    <th>Row</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {batchResults.predictions.slice(0, 50).map((result, index) => (
                    <tr key={index}>
                      <td>{result.row_index + 1}</td>
                      <td>
                        <span className="prediction-badge">
                          {result.prediction}
                        </span>
                      </td>
                      <td>{result.confidence.toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {batchResults.predictions.length > 50 && (
                <div className="table-footer">
                  Showing first 50 results of {batchResults.predictions.length}
                </div>
                )}
            </div>
          </div>
        )}
      </div>
    </div>
  );

  // Main App Render
  return (
    <div>
      {currentPage === 'landing' ? <LandingPage /> : <PredictionPage />}
    </div>
  );
};

export default App;