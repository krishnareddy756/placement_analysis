import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Campus Placement Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and metadata
@st.cache_resource
def load_model_and_data():
    try:
        # Load the trained model
        model = joblib.load('models/lr_model.pkl')
        
        # Load feature columns
        with open('models/feature_columns.json', 'r') as f:
            feature_columns = json.load(f)
        
        # Load model metadata
        with open('models/model_metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        # Load data for EDA
        data = pd.read_csv('placementdata.csv', encoding='utf-8')
        
        # Load insights
        try:
            with open('artifacts/insights_summary.txt', 'r', encoding='utf-8') as f:
                insights = f.read()
        except UnicodeDecodeError:
            # Try with UTF-16 encoding if UTF-8 fails
            with open('artifacts/insights_summary.txt', 'r', encoding='utf-16') as f:
                insights = f.read()
        except Exception:
            # Fallback to reading with default encoding
            with open('artifacts/insights_summary.txt', 'r') as f:
                insights = f.read()
            
        return model, feature_columns, metadata, data, insights
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return None, None, None, None, None

# Load everything
model, feature_columns, metadata, data, insights = load_model_and_data()

# Sidebar
st.sidebar.title("🎓 Campus Placement Predictor")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Choose a section:",
    ["📊 Overview", "📈 Data Analysis", "🔮 Make Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance")
if metadata:
    st.sidebar.metric("Accuracy", f"{metadata['test_accuracy']:.1%}")
    st.sidebar.metric("ROC-AUC", f"{metadata['test_roc_auc']:.3f}")

# Main content
if page == "📊 Overview":
    st.title("🎓 Campus Placement Prediction Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    if metadata:
        with col1:
            st.metric("Model Accuracy", f"{metadata['test_accuracy']:.1%}")
        with col2:
            st.metric("ROC-AUC Score", f"{metadata['test_roc_auc']:.3f}")
        with col3:
            st.metric("Training Samples", f"{metadata['training_samples']:,}")
    
    st.markdown("---")
    
    # Project overview
    st.header("📋 Project Overview")
    st.write("""
    This machine learning model predicts the likelihood of campus placement for students 
    based on various academic and personal factors. The model achieves over 80% accuracy 
    in predicting placement outcomes.
    """)
    
    # Key insights
    st.header("🔍 Key Insights")
    if insights:
        st.text(insights)
    
    # Feature importance
    st.header("📊 Most Important Features")
    try:
        coef_df = pd.read_csv('artifacts/coefficient_importance.csv')
        
        fig = px.bar(
            coef_df.head(8), 
            x='coef_abs', 
            y='feature',
            orientation='h',
            title="Top 8 Features by Importance",
            labels={'coef_abs': 'Coefficient (Absolute Value)', 'feature': 'Features'}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load feature importance: {e}")

elif page == "📈 Data Analysis":
    st.title("📈 Exploratory Data Analysis")
    
    if data is not None:
        # Dataset overview
        st.header("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", len(data))
        with col2:
            placed_count = data['PlacementStatus'].value_counts().get('Placed', 0)
            st.metric("Placed Students", placed_count)
        with col3:
            placement_rate = placed_count / len(data) * 100
            st.metric("Placement Rate", f"{placement_rate:.1f}%")
        with col4:
            st.metric("Features", len(feature_columns))
        
        st.markdown("---")
        
        # Placement distribution
        st.header("📋 Placement Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                data, 
                names='PlacementStatus', 
                title="Placement Status Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # CGPA distribution by placement
            fig = px.box(
                data, 
                x='PlacementStatus', 
                y='CGPA',
                title="CGPA Distribution by Placement Status"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.header("🔗 Feature Correlations")
        
        # Prepare numeric data
        numeric_cols = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications', 
                       'AptitudeTestScore', 'SoftSkillsRating', 'SSC_Marks', 'HSC_Marks']
        
        if all(col in data.columns for col in numeric_cols):
            corr_matrix = data[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Feature Correlation Matrix",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions
        st.header("📊 Feature Distributions")
        
        selected_feature = st.selectbox(
            "Select a feature to visualize:",
            numeric_cols
        )
        
        if selected_feature:
            fig = px.histogram(
                data, 
                x=selected_feature, 
                color='PlacementStatus',
                title=f"Distribution of {selected_feature} by Placement Status",
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("Could not load dataset for analysis.")

elif page == "🔮 Make Prediction":
    st.title("🔮 Placement Prediction")
    
    st.write("Enter student details to predict placement probability:")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, 0.1)
            internships = st.number_input("Number of Internships", 0, 10, 1)
            projects = st.number_input("Number of Projects", 0, 20, 3)
            workshops = st.number_input("Workshops/Certifications", 0, 20, 2)
            aptitude_score = st.slider("Aptitude Test Score", 0, 100, 75)
        
        with col2:
            soft_skills = st.slider("Soft Skills Rating", 1, 5, 3)
            extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
            placement_training = st.selectbox("Placement Training", ["Yes", "No"])
            ssc_marks = st.slider("SSC Marks (%)", 0.0, 100.0, 85.0)
            hsc_marks = st.slider("HSC Marks (%)", 0.0, 100.0, 85.0)
        
        submit_button = st.form_submit_button("🔮 Predict Placement")
    
    if submit_button and model is not None:
        # Prepare input data
        input_data = pd.DataFrame({
            'CGPA': [cgpa],
            'Internships': [internships],
            'Projects': [projects],
            'Workshops/Certifications': [workshops],
            'AptitudeTestScore': [aptitude_score],
            'SoftSkillsRating': [soft_skills],
            'ExtracurricularActivities': [1 if extracurricular == "Yes" else 0],
            'PlacementTraining': [1 if placement_training == "Yes" else 0],
            'SSC_Marks': [ssc_marks],
            'HSC_Marks': [hsc_marks]
        })
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            # Display results
            st.markdown("---")
            st.header("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.success("✅ LIKELY TO BE PLACED")
                else:
                    st.error("❌ UNLIKELY TO BE PLACED")
            
            with col2:
                placement_prob = probability[1] * 100
                st.metric("Placement Probability", f"{placement_prob:.1f}%")
            
            with col3:
                confidence = max(probability) * 100
                st.metric("Confidence", f"{confidence:.1f}%")
            
            # Probability breakdown
            st.subheader("📊 Probability Breakdown")
            prob_df = pd.DataFrame({
                'Outcome': ['Not Placed', 'Placed'],
                'Probability': [probability[0], probability[1]]
            })
            
            fig = px.bar(
                prob_df, 
                x='Outcome', 
                y='Probability',
                title="Prediction Probabilities",
                color='Outcome',
                color_discrete_map={'Placed': '#2E8B57', 'Not Placed': '#DC143C'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            
            if placement_prob < 50:
                st.warning("**Areas for Improvement:**")
                if aptitude_score < 70:
                    st.write("• Focus on improving aptitude test performance")
                if placement_training == "No":
                    st.write("• Consider enrolling in placement training programs")
                if extracurricular == "No":
                    st.write("• Participate in extracurricular activities")
                if soft_skills < 4:
                    st.write("• Work on developing soft skills")
                if projects < 3:
                    st.write("• Build more practical projects")
            else:
                st.success("**Strong Profile!** Keep up the good work and continue enhancing your skills.")
        
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        🎓 Campus Placement Prediction Dashboard | Built with Streamlit & Machine Learning
    </div>
    """, 
    unsafe_allow_html=True
)
