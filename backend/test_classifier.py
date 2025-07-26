
"""
Test script for the Aviation Classifier
Run this to test if your models are loading correctly
"""

from classifier import AviationClassifier
import json

def test_classifier():
    print("=" * 50)
    print("Testing Aviation Classifier")
    print("=" * 50)
    
    # Initialize classifier
    classifier = AviationClassifier()
    
    # Test 1: Check available models
    print("\n1. Testing available models:")
    models = classifier.get_available_models()
    print(f"Available models: {models}")
    
    if not models:
        print("❌ No models found! Please check if you have .pkl files in the 'models' directory")
        return False
    
    # Test 2: Check current model
    print(f"\n2. Current model: {classifier.current_model_name}")
    print(f"Model loaded: {'✅' if classifier.model is not None else '❌'}")
    
    # Test 3: Check features
    print(f"\n3. Important features: {classifier.get_feature_names()}")
    print(f"All model features: {len(classifier.get_all_feature_names())} features")
    
    # Test 4: Test prediction with sample data
    print("\n4. Testing prediction with sample data:")
    sample_data = classifier.create_sample_data()
    print(f"Sample data: {json.dumps(sample_data, indent=2)}")
    
    result = classifier.predict(sample_data)
    print(f"Prediction result: {json.dumps(result, indent=2)}")
    
    if 'error' in result:
        print("❌ Prediction failed!")
        return False
    else:
        print("✅ Prediction successful!")
    
    # Test 5: Test model switching (if multiple models available)
    if len(models) > 1:
        print(f"\n5. Testing model switching to: {models[1]}")
        success = classifier.load_model(models[1])
        print(f"Model switch: {'✅' if success else '❌'}")
        
        # Test prediction with new model
        result2 = classifier.predict(sample_data)
        print(f"New model prediction: {json.dumps(result2, indent=2)}")
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    test_classifier()