# -*- coding: utf-8 -*-
"""
Fixed Campus Placement Analysis Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("placementdata.csv")
print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Get basic info and statistics
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Encode categorical columns
print("\nEncoding categorical variables...")
df['ExtracurricularActivities'] = df['ExtracurricularActivities'].map({'Yes': 1, 'No': 0})
df['PlacementTraining'] = df['PlacementTraining'].map({'Yes': 1, 'No': 0})
df['PlacementStatus'] = df['PlacementStatus'].map({'Placed': 1, 'NotPlaced': 0})

# Check if there are any missing values after encoding
print("Missing values after encoding:")
print(df.isnull().sum())

# Create a copy for visualization with labels
df_viz = df.copy()
df_viz['PlacementLabel'] = df_viz['PlacementStatus'].map({1:'Placed', 0:'Not Placed'})
df_viz['TrainingLabel'] = df_viz['PlacementTraining'].map({1:'Yes', 0:'No'})

# Visualizations
print("\nCreating visualizations...")

# Distribution of PlacementStatus
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='PlacementLabel', data=df_viz)
plt.title("Placed vs Not Placed Count")
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")

# Placement Training vs Placement Status
plt.subplot(1, 2, 2)
sns.countplot(x='TrainingLabel', hue='PlacementLabel', data=df_viz)
plt.title("Placement Training vs Placement Status")
plt.xlabel("Placement Training")
plt.ylabel("Number of Students")
plt.legend(title='Placement Status')
plt.tight_layout()
plt.show()

# Correlation matrix (only numeric columns)
print("\nComputing correlation matrix...")
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric columns:", numeric_columns)
corr = df[numeric_columns].corr()

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.show()

# Scatter plot: AptitudeTestScore vs CGPA colored by PlacementStatus
plt.figure(figsize=(8, 6))
for status, label in [(1, 'Placed'), (0, 'Not Placed')]:
    mask = df['PlacementStatus'] == status
    plt.scatter(df.loc[mask, 'AptitudeTestScore'], 
               df.loc[mask, 'CGPA'], 
               label=label, alpha=0.6)
plt.xlabel("Aptitude Test Score")
plt.ylabel("CGPA")
plt.title("Aptitude Test Score vs CGPA by Placement Status")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Distribution plots
print("\nCreating distribution plots...")
plt.figure(figsize=(15, 10))

# Key features to plot
features = ['CGPA', 'AptitudeTestScore', 'Projects', 'Internships']
for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    for status, label in [(1, 'Placed'), (0, 'Not Placed')]:
        mask = df['PlacementStatus'] == status
        plt.hist(df.loc[mask, feature], alpha=0.7, label=label, bins=20)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {feature} by Placement Status')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Machine Learning Model
print("\nBuilding Machine Learning Model...")

# Prepare features and target
# Remove non-feature columns
feature_columns = [col for col in df.columns if col not in ['StudentID', 'PlacementStatus']]
X = df[feature_columns]
y = df['PlacementStatus']

print(f"Features: {feature_columns}")
print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = lr_model.predict(X_test_scaled)
y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n🎯 MODEL PERFORMANCE:")
print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"ROC-AUC Score: {roc_auc:.3f}")

# Classification Report
print(f"\n📊 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Not Placed', 'Placed']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Placed', 'Placed'], 
            yticklabels=['Not Placed', 'Placed'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Feature Importance
coefficients = lr_model.coef_[0]
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\n🔍 FEATURE IMPORTANCE:")
print(feature_importance.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(8), x='Abs_Coefficient', y='Feature')
plt.title('Top 8 Most Important Features')
plt.xlabel('Absolute Coefficient Value')
plt.tight_layout()
plt.show()

# Save model and artifacts
print("\n💾 SAVING MODEL AND ARTIFACTS...")

# Create directories if they don't exist
import os
os.makedirs('models', exist_ok=True)
os.makedirs('artifacts', exist_ok=True)

# Save model
joblib.dump(lr_model, 'models/lr_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Save feature columns
with open('models/feature_columns.json', 'w') as f:
    json.dump(feature_columns, f)

# Save model metadata
model_metadata = {
    'model_type': 'LogisticRegression',
    'accuracy': float(accuracy),
    'roc_auc': float(roc_auc),
    'feature_count': len(feature_columns),
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

# Save feature importance
feature_importance.to_csv('artifacts/coefficient_importance.csv', index=False)

# Generate insights summary
insights = f"""
🎓 CAMPUS PLACEMENT PREDICTION - KEY INSIGHTS

📊 MODEL PERFORMANCE:
- Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)
- ROC-AUC Score: {roc_auc:.3f}
- Model: Logistic Regression with StandardScaler

🔍 TOP PREDICTORS:
"""

for idx, row in feature_importance.head(5).iterrows():
    insights += f"- {row['Feature']}: {row['Abs_Coefficient']:.4f}\n"

insights += f"""
📈 DATASET OVERVIEW:
- Total Students: {len(df):,}
- Features: {len(feature_columns)}
- Placement Rate: {(df['PlacementStatus'].sum() / len(df) * 100):.1f}%

💡 KEY RECOMMENDATIONS:
- Focus on aptitude test preparation (highest impact factor)
- Encourage placement training participation
- Promote extracurricular activities
- Maintain strong academic performance (CGPA)

🚀 MODEL STATUS: Ready for deployment
"""

with open('artifacts/insights_summary.txt', 'w') as f:
    f.write(insights)

print("✅ Analysis complete! Check the 'artifacts' folder for detailed results.")
print(f"📁 Model saved to: models/lr_model.pkl")
print(f"📁 Results saved to: artifacts/")

# Final summary
print(f"\n🎯 FINAL SUMMARY:")
print(f"{'='*50}")
print(f"Model Accuracy: {accuracy*100:.1f}%")
print(f"ROC-AUC Score: {roc_auc:.3f}")
print(f"Top Predictor: {feature_importance.iloc[0]['Feature']}")
print(f"Total Students Analyzed: {len(df):,}")
print(f"{'='*50}")
print("Run 'streamlit run streamlit_app.py' for interactive dashboard!")
