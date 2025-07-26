import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import glob

class AviationClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.current_model_name = None
        self.models_directory = "models"
        
        # Define the important features (matching frontend)
        self.important_features = [
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
        ]
        
        self.expected_features = [
            'Investigation.Type', 'Location', 'Country', 'Injury.Severity', 
            'Aircraft.Category', 'Make', 'Amateur.Built', 'Engine.Type', 
            'Purpose.of.flight', 'Weather.Condition', 'Broad.phase.of.flight',
            'Number.of.Engines', 'Aircraft.damage', 'Total.Fatal.Injuries',
            'Total.Serious.Injuries', 'Total.Minor.Injuries', 'Total.Uninjured',
            'Event.Date', 'Airport.Code', 'Model', 'Schedule', 'Air.carrier',
            'FAR.Description'
        ]
        
        self.load_default_model()

    def load_default_model(self):
        """Load the first available model as default"""
        try:
            available_models = self.get_available_models()
            if available_models:
                self.load_model(available_models[0])
                print(f"Loaded default model: {available_models[0]}")
            else:
                print("No models found in the models directory")
        except Exception as e:
            print(f"Error loading default model: {str(e)}")

    def get_available_models(self):
        """Get list of available pickle models"""
        try:
            if not os.path.exists(self.models_directory):
                print(f"Models directory '{self.models_directory}' does not exist")
                return []
            
            model_files = glob.glob(os.path.join(self.models_directory, "*.pkl"))
            models = [os.path.basename(f).replace('.pkl', '') for f in model_files]
            print(f"Found models: {models}")
            return models
        except Exception as e:
            print(f"Error getting available models: {str(e)}")
            return []

    def load_model(self, model_name):
        """Load a specific model from pickle file"""
        try:
            model_path = os.path.join(self.models_directory, f"{model_name}.pkl")
            
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return False

            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.label_encoders = model_data.get('label_encoders', {})
                stored_features = model_data.get('feature_names', self.expected_features)
            else:
                self.model = model_data
                self.scaler = None
                self.label_encoders = {}
                stored_features = self.expected_features

            if hasattr(self.model, 'n_features_in_'):
                expected_feature_count = self.model.n_features_in_
                print(f"Model expects {expected_feature_count} features")
                
                if len(stored_features) >= expected_feature_count:
                    self.feature_names = stored_features[:expected_feature_count]
                else:
                    self.feature_names = stored_features + self.expected_features[:expected_feature_count - len(stored_features)]
                    self.feature_names = self.feature_names[:expected_feature_count]
            else:
                self.feature_names = stored_features

            self.current_model_name = model_name
            print(f"Successfully loaded model: {model_name}")
            print(f"Model type: {type(self.model)}")
            print(f"Feature count used: {len(self.feature_names)}")
            print(f"Features: {self.feature_names}")
            return True

        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            return False

    def preprocess_data(self, data):
        """Preprocess the input data for prediction"""
        try:
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)

            print(f"Input data columns: {list(df.columns)}")
            print(f"Input data shape: {df.shape}")
            print(f"Model expects features: {self.feature_names}")

            processed_df = pd.DataFrame()
            
            for feature in self.feature_names:
                if feature in df.columns:
                    processed_df[feature] = df[feature]
                elif feature in ['Number.of.Engines', 'Total.Fatal.Injuries', 
                               'Total.Serious.Injuries', 'Total.Minor.Injuries', 
                               'Total.Uninjured']:
                    processed_df[feature] = 0
                else:
                    processed_df[feature] = 'Unknown'

            numeric_columns = ['Number.of.Engines', 
                             'Total.Fatal.Injuries', 'Total.Serious.Injuries',
                             'Total.Minor.Injuries', 'Total.Uninjured']
            
            for col in numeric_columns:
                if col in processed_df.columns:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)

            categorical_columns = [col for col in processed_df.columns if col not in numeric_columns]
            
            for col in categorical_columns:
                if col in processed_df.columns:
                    processed_df[col] = processed_df[col].astype(str).fillna('Unknown')
                    
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        unique_values = processed_df[col].unique()
                        for val in unique_values:
                            if val not in le.classes_:
                                le.classes_ = np.append(le.classes_, val)
                        try:
                            processed_df[col] = le.transform(processed_df[col])
                        except ValueError as ve:
                            print(f"Label encoding error for {col}: {ve}")
                            le = LabelEncoder()
                            processed_df[col] = le.fit_transform(processed_df[col])
                            self.label_encoders[col] = le
                    else:
                        le = LabelEncoder()
                        processed_df[col] = le.fit_transform(processed_df[col])
                        self.label_encoders[col] = le

            processed_df = processed_df.fillna(0)

            print(f"Processed data shape: {processed_df.shape}")
            print(f"Processed data columns: {list(processed_df.columns)}")

            if processed_df.shape[1] != len(self.feature_names):
                raise ValueError(f"Feature count mismatch: expected {len(self.feature_names)}, got {processed_df.shape[1]}")

            if self.scaler:
                df_scaled = self.scaler.transform(processed_df)
                print("Applied scaling to data")
                return df_scaled
            else:
                print("No scaler available, using raw data")
                return processed_df.values

        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

    def predict(self, data):
        """Make prediction on input data"""
        try:
            if self.model is None:
                return {'error': 'No model loaded'}

            print(f"Making prediction with model: {self.current_model_name}")
            print(f"Input data: {data}")

            processed_data = self.preprocess_data(data)
            print(f"Processed data shape: {processed_data.shape}")

            prediction = self.model.predict(processed_data)
            print(f"Raw prediction: {prediction}")

            try:
                prediction_proba = self.model.predict_proba(processed_data)
                confidence = np.max(prediction_proba) * 100
                print(f"Prediction probabilities: {prediction_proba}")
                print(f"Confidence: {confidence}")
            except Exception as prob_error:
                print(f"Could not get prediction probabilities: {prob_error}")
                confidence = 0

            if hasattr(prediction, 'tolist'):
                prediction = prediction.tolist()
            
            if isinstance(prediction, list) and len(prediction) == 1:
                prediction = prediction[0]

            result = {
                'prediction': str(prediction),
                'confidence': float(confidence),
                'model_used': self.current_model_name or 'Unknown'
            }
            
            print(f"Final result: {result}")
            return result

        except Exception as e:
            error_msg = f'Prediction failed: {str(e)}'
            print(error_msg)
            import traceback
            traceback.print_exc()
            return {'error': error_msg}

    def get_feature_names(self):
        """Return the list of important feature names for frontend"""
        return self.important_features

    def get_all_feature_names(self):
        """Return the list of all feature names used by the model"""
        return self.feature_names

    def save_model(self, model_name, model, scaler=None, label_encoders=None):
        """Save model to pickle file"""
        try:
            if not os.path.exists(self.models_directory):
                os.makedirs(self.models_directory)

            model_data = {
                'model': model,
                'scaler': scaler,
                'label_encoders': label_encoders or {},
                'feature_names': self.feature_names
            }

            model_path = os.path.join(self.models_directory, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            print(f"Model saved successfully: {model_path}")
            return True

        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False

    def create_sample_data(self):
        """Create sample data for testing"""
        sample_data = {}
        for feature in self.important_features:
            if feature == 'Investigation.Type':
                sample_data[feature] = 'Accident'
            elif feature == 'Location':
                sample_data[feature] = 'New York, NY'
            elif feature == 'Country':
                sample_data[feature] = 'United States'
            elif feature == 'Injury.Severity':
                sample_data[feature] = 'Minor'
            elif feature == 'Aircraft.Category':
                sample_data[feature] = 'Airplane'
            elif feature == 'Make':
                sample_data[feature] = 'Cessna'
            elif feature == 'Amateur.Built':
                sample_data[feature] = 'No'
            elif feature == 'Engine.Type':
                sample_data[feature] = 'Reciprocating'
            elif feature == 'Purpose.of.flight':
                sample_data[feature] = 'Personal'
            elif feature == 'Weather.Condition':
                sample_data[feature] = 'VMC'
            elif feature == 'Broad.phase.of.flight':
                sample_data[feature] = 'Landing'
            else:
                sample_data[feature] = 'Unknown'
        
        return sample_data