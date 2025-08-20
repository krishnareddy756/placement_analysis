"""
Simple Campus Placement Prediction Script
This is a simpler version for command-line usage
"""

import pandas as pd
import numpy as np
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_model_and_features():
    """Load the trained model and feature list"""
    try:
        # Load model
        model = joblib.load('models/lr_model.pkl')
        
        # Load feature columns
        with open('models/feature_columns.json', 'r') as f:
            feature_columns = json.load(f)
        
        return model, feature_columns
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict_placement(student_data):
    """
    Predict placement for a student
    
    Args:
        student_data (dict): Dictionary with student features
    
    Returns:
        dict: Prediction results
    """
    model, feature_columns = load_model_and_features()
    
    if model is None:
        return {"error": "Could not load model"}
    
    try:
        # Create DataFrame with the input data
        input_df = pd.DataFrame([student_data])
        
        # Ensure all required columns are present
        for col in feature_columns:
            if col not in input_df.columns:
                print(f"Warning: Missing feature {col}, setting to 0")
                input_df[col] = 0
        
        # Reorder columns to match training data
        input_df = input_df[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        return {
            "prediction": "Placed" if prediction == 1 else "Not Placed",
            "probability_not_placed": float(probability[0]),
            "probability_placed": float(probability[1]),
            "confidence": float(max(probability))
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

def main():
    """Main function to demonstrate the prediction"""
    print("🎓 Campus Placement Prediction System")
    print("=" * 50)
    
    # Example student data
    example_student = {
        'CGPA': 8.5,
        'Internships': 2,
        'Projects': 4,
        'Workshops/Certifications': 3,
        'AptitudeTestScore': 85,
        'SoftSkillsRating': 4,
        'ExtracurricularActivities': 1,  # 1 for Yes, 0 for No
        'PlacementTraining': 1,  # 1 for Yes, 0 for No
        'SSC_Marks': 88.5,
        'HSC_Marks': 89.2
    }
    
    print("Example Student Profile:")
    for key, value in example_student.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    
    # Make prediction
    result = predict_placement(example_student)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("📊 PREDICTION RESULTS:")
        print(f"  Outcome: {result['prediction']}")
        print(f"  Placement Probability: {result['probability_placed']:.1%}")
        print(f"  Confidence: {result['confidence']:.1%}")
        
        if result['probability_placed'] > 0.5:
            print("  ✅ High likelihood of placement!")
        else:
            print("  ⚠️  Consider improving key areas for better chances.")
    
    print("\n" + "=" * 50)
    print("For interactive predictions, run: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
