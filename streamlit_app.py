import streamlit as st
import pandas as pd
import numpy as np
import json
try:
    import joblib
except ImportError:
    st.error("joblib is not installed. Please install it using: pip install joblib")
    st.stop()
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
        # Load model with version handling
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load('models/lr_model.pkl')
        
        # Load feature columns
        with open('models/feature_columns.json', 'r', encoding='utf-8') as f:
            feature_columns = json.load(f)
        
        # Load model metadata
        with open('models/model_metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        # Load data for EDA
        data = pd.read_csv('placementdata.csv', encoding='utf-8')
        
        # Load insights
        insights = None
        
        # Try to load the clean version first
        for filename in ['artifacts/insights_summary_clean.txt', 'artifacts/insights_summary.txt']:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    insights = f.read()
                    break
            except UnicodeDecodeError:
                try:
                    # Try UTF-16 with BOM
                    with open(filename, 'r', encoding='utf-16') as f:
                        insights = f.read()
                        break
                except UnicodeDecodeError:
                    try:
                        # Try latin-1 as fallback
                        with open(filename, 'r', encoding='latin-1') as f:
                            insights = f.read()
                            break
                    except UnicodeDecodeError:
                        continue
            except FileNotFoundError:
                continue
        
        # If still no insights, create a default one
        if not insights:
            insights = """CAMPUS PLACEMENT PREDICTION - KEY INSIGHTS

MODEL PERFORMANCE:
- Achieved 80.9% accuracy and 0.884 ROC-AUC on test set
- Logistic Regression proved most effective and interpretable

TOP PREDICTIVE FEATURES:
1. AptitudeTestScore (0.6108) - STRONGEST predictor
2. PlacementTraining (0.3920) - Significant positive impact  
3. ExtracurricularActivities (0.3604) - Strong correlation
4. SSC_Marks (0.2826) - Academic foundation matters
5. SoftSkillsRating (0.2634) - Communication skills crucial

KEY INSIGHTS:
- Aptitude test performance is the #1 predictor
- Placement training programs have measurable ROI
- Extracurricular activities significantly boost placement chances
- Academic performance provides solid foundation

RECOMMENDATIONS:
- Implement aptitude test preparation workshops
- Expand placement training programs
- Encourage extracurricular participation
- Provide targeted support for low-scoring students"""
            
        return model, feature_columns, metadata, data, insights
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        # Try to load at least the data and metadata
        try:
            with open('models/feature_columns.json', 'r', encoding='utf-8') as f:
                feature_columns = json.load(f)
            with open('models/model_metadata.json', 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            data = pd.read_csv('placementdata.csv', encoding='utf-8')
            # Load insights with better encoding handling
            insights = None
            for filename in ['artifacts/insights_summary_clean.txt', 'artifacts/insights_summary.txt']:
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        insights = f.read()
                        break
                except UnicodeDecodeError:
                    try:
                        with open(filename, 'r', encoding='utf-16') as f:
                            insights = f.read()
                            break
                    except UnicodeDecodeError:
                        try:
                            with open(filename, 'r', encoding='latin-1') as f:
                                insights = f.read()
                                break
                        except UnicodeDecodeError:
                            continue
                except FileNotFoundError:
                    continue
            
            # If still no insights, create a default one
            if not insights:
                insights = """CAMPUS PLACEMENT PREDICTION - KEY INSIGHTS

MODEL PERFORMANCE:
- Achieved 80.9% accuracy and 0.884 ROC-AUC on test set
- Logistic Regression proved most effective and interpretable

TOP PREDICTIVE FEATURES:
1. AptitudeTestScore (0.6108) - STRONGEST predictor
2. PlacementTraining (0.3920) - Significant positive impact  
3. ExtracurricularActivities (0.3604) - Strong correlation
4. SSC_Marks (0.2826) - Academic foundation matters
5. SoftSkillsRating (0.2634) - Communication skills crucial

KEY INSIGHTS:
- Aptitude test performance is the #1 predictor
- Placement training programs have measurable ROI
- Extracurricular activities significantly boost placement chances
- Academic performance provides solid foundation

RECOMMENDATIONS:
- Implement aptitude test preparation workshops
- Expand placement training programs
- Encourage extracurricular participation
- Provide targeted support for low-scoring students"""
            st.warning("Model could not be loaded due to version mismatch, but data analysis is still available.")
            return None, feature_columns, metadata, data, insights
        except Exception as e2:
            st.error(f"Critical error: {e2}")
            return None, None, None, None, None

# Load everything
model, feature_columns, metadata, data, insights = load_model_and_data()

# Simple backup prediction function
def backup_prediction(input_data):
    """Simple rule-based prediction when ML model fails"""
    score = 0
    
    # CGPA weight (30%)
    if input_data['CGPA'].iloc[0] >= 8.0:
        score += 30
    elif input_data['CGPA'].iloc[0] >= 7.0:
        score += 20
    elif input_data['CGPA'].iloc[0] >= 6.0:
        score += 10
    else:
        score += 5
    
    # Aptitude weight (25%)
    if input_data['AptitudeTestScore'].iloc[0] >= 80:
        score += 25
    elif input_data['AptitudeTestScore'].iloc[0] >= 70:
        score += 15
    else:
        score += 5
    
    # Experience weight (20%)
    exp_score = input_data['Internships'].iloc[0] + input_data['Projects'].iloc[0]
    if exp_score >= 4:
        score += 20
    elif exp_score >= 2:
        score += 15
    else:
        score += 5
    
    # Training and activities (25%)
    if input_data['PlacementTraining'].iloc[0] == 1:
        score += 15
    if input_data['ExtracurricularActivities'].iloc[0] == 1:
        score += 10
    
    # Convert to probability
    probability = min(score / 100.0, 0.95)  # Cap at 95%
    prediction = 1 if probability > 0.5 else 0
    
    return prediction, [1-probability, probability]

# Sidebar
st.sidebar.title("🎓 Campus Placement Predictor")
st.sidebar.markdown("---")

# Show loading status
if model is not None:
    st.sidebar.success("✅ Model Loaded")
else:
    st.sidebar.warning("⚠️ Using Backup Prediction")

if data is not None:
    st.sidebar.success("✅ Data Loaded")
else:
    st.sidebar.error("❌ Data Missing")

# Navigation
page = st.sidebar.selectbox(
    "Choose a section:",
    ["📊 Overview", "🔍 Key Insights", "📈 Data Analysis", "🔮 Make Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance")
if metadata:
    st.sidebar.metric("Accuracy", f"{metadata['test_accuracy']:.1%}")
    st.sidebar.metric("ROC-AUC", f"{metadata['test_roc_auc']:.3f}")
else:
    st.sidebar.info("Metadata not available")

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
    
    # Feature importance
    st.header("📊 Most Important Features")
    try:
        coef_df = pd.read_csv('artifacts/coefficient_importance.csv')
        
        # Ensure we have the required columns
        if 'coef_abs' not in coef_df.columns:
            if 'coef' in coef_df.columns:
                coef_df['coef_abs'] = abs(coef_df['coef'])
            else:
                st.error("Missing coefficient data in importance file")
                st.stop()
        
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
    except FileNotFoundError:
        st.warning("Feature importance file not found. Please ensure the model has been trained.")
    except Exception as e:
        st.error(f"Could not load feature importance: {e}")
        st.write("Debug info:", str(e))

elif page == "🔍 Key Insights":
    st.title("🔍 Key Insights & Analysis")
    
    # Key insights
    st.header("📝 Model Insights Summary")
    if insights:
        st.markdown(insights)
    else:
        st.warning("Key insights data not available")
    
    # Feature importance detailed view
    st.header("📊 Feature Importance Analysis")
    try:
        coef_df = pd.read_csv('artifacts/coefficient_importance.csv')
        
        # Ensure we have the required columns
        if 'coef_abs' not in coef_df.columns:
            if 'coef' in coef_df.columns:
                coef_df['coef_abs'] = abs(coef_df['coef'])
            else:
                st.error("Missing coefficient data in importance file")
                coef_df = None
        
        if coef_df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    coef_df, 
                    x='coef_abs', 
                    y='feature',
                    orientation='h',
                    title="All Features by Importance",
                    labels={'coef_abs': 'Coefficient (Absolute Value)', 'feature': 'Features'}
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Top features table
                st.subheader("Top Features Table")
                st.dataframe(coef_df.head(10), use_container_width=True)
                
            # Insights about features
            st.subheader("🎯 Feature Analysis")
            st.write("""
            **Key Findings:**
            
            1. **Aptitude Test Score** is the strongest predictor - focus on test preparation
            2. **Placement Training** significantly improves chances - ROI is measurable  
            3. **Extracurricular Activities** show strong positive correlation
            4. **Academic Foundation** (SSC/HSC marks) provides solid base
            5. **Soft Skills Rating** is crucial for final selection
            """)
        
    except FileNotFoundError:
        st.warning("Feature importance file not found. Please ensure the model has been trained.")
    except Exception as e:
        st.error(f"Could not load feature importance: {e}")
        st.write("Debug info:", str(e))
    
    # Model performance metrics
    if metadata:
        st.header("🎯 Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metadata['test_accuracy']:.1%}")
        with col2:
            st.metric("ROC-AUC", f"{metadata['test_roc_auc']:.3f}")
        with col3:
            st.metric("Training Samples", f"{metadata['training_samples']:,}")
        with col4:
            st.metric("Model Type", "Logistic Regression")

elif page == "📈 Data Analysis":
    st.title("📈 Exploratory Data Analysis")
    
    if data is not None and not data.empty:
        # Dataset overview
        st.header("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", len(data))
        with col2:
            if 'PlacementStatus' in data.columns:
                placed_count = data['PlacementStatus'].value_counts().get('Placed', 0)
                st.metric("Placed Students", placed_count)
            else:
                st.metric("Placed Students", "N/A")
        with col3:
            if 'PlacementStatus' in data.columns:
                placed_count = data['PlacementStatus'].value_counts().get('Placed', 0)
                placement_rate = placed_count / len(data) * 100 if len(data) > 0 else 0
                st.metric("Placement Rate", f"{placement_rate:.1f}%")
            else:
                st.metric("Placement Rate", "N/A")
        with col4:
            st.metric("Features", len(data.columns))
        
        st.markdown("---")
        
        # Show data sample
        st.header("📋 Data Sample")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Placement distribution
        if 'PlacementStatus' in data.columns:
            st.header("📊 Placement Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                # Ensure proper data handling for pie chart
                placement_counts = data['PlacementStatus'].value_counts()
                
                fig = px.pie(
                    values=placement_counts.values,
                    names=placement_counts.index,
                    title="Placement Status Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # CGPA distribution by placement
                if 'CGPA' in data.columns:
                    # Ensure CGPA is numeric
                    try:
                        data_clean = data.copy()
                        data_clean['CGPA'] = pd.to_numeric(data_clean['CGPA'], errors='coerce')
                        data_clean = data_clean.dropna(subset=['CGPA'])
                        
                        fig = px.box(
                            data_clean, 
                            x='PlacementStatus', 
                            y='CGPA',
                            title="CGPA Distribution by Placement Status"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating CGPA box plot: {e}")
        else:
            st.warning("PlacementStatus column not found in data")
        
        # Correlation analysis
        st.header("🔗 Feature Correlations")
        
        # Prepare numeric data
        numeric_cols = []
        potential_cols = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications', 
                         'AptitudeTestScore', 'SoftSkillsRating', 'SSC_Marks', 'HSC_Marks']
        
        # Clean and convert data to numeric
        data_numeric = data.copy()
        for col in potential_cols:
            if col in data.columns:
                try:
                    data_numeric[col] = pd.to_numeric(data_numeric[col], errors='coerce')
                    if not data_numeric[col].isna().all():  # Check if column has any valid numeric data
                        numeric_cols.append(col)
                except Exception:
                    continue
        
        if len(numeric_cols) >= 2:
            try:
                # Drop rows with NaN values for correlation calculation
                corr_data = data_numeric[numeric_cols].dropna()
                
                if len(corr_data) > 0:
                    corr_matrix = corr_data.corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        title="Feature Correlation Matrix",
                        aspect="auto",
                        text_auto=".2f",
                        color_continuous_scale="RdBu_r"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No valid numeric data available for correlation analysis")
            except Exception as e:
                st.error(f"Error creating correlation matrix: {e}")
        else:
            st.warning("Not enough numeric columns for correlation analysis")
        
        # Feature distributions
        if numeric_cols:
            st.header("📊 Feature Distributions")
            
            selected_feature = st.selectbox(
                "Select a feature to visualize:",
                numeric_cols
            )
            
            if selected_feature and 'PlacementStatus' in data.columns:
                try:
                    # Clean the data for visualization
                    viz_data = data.copy()
                    viz_data[selected_feature] = pd.to_numeric(viz_data[selected_feature], errors='coerce')
                    viz_data = viz_data.dropna(subset=[selected_feature])
                    
                    fig = px.histogram(
                        viz_data, 
                        x=selected_feature, 
                        color='PlacementStatus',
                        title=f"Distribution of {selected_feature} by Placement Status",
                        barmode='overlay',
                        opacity=0.7
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating histogram: {e}")
                    
            elif selected_feature:
                try:
                    viz_data = data.copy()
                    viz_data[selected_feature] = pd.to_numeric(viz_data[selected_feature], errors='coerce')
                    viz_data = viz_data.dropna(subset=[selected_feature])
                    
                    fig = px.histogram(
                        viz_data, 
                        x=selected_feature,
                        title=f"Distribution of {selected_feature}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating histogram: {e}")
        else:
            st.warning("No numeric columns available for distribution analysis")
    
    else:
        st.error("Could not load dataset for analysis. Please check if placementdata.csv exists and is properly formatted.")

elif page == "🔮 Make Prediction":
    st.title("🔮 Placement Prediction")
    
    if model is not None and feature_columns is not None:
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
        
        if submit_button:
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
            
            # Ensure the columns are in the same order as the model expects
            if feature_columns:
                try:
                    input_data = input_data[feature_columns]
                except KeyError as e:
                    st.error(f"Feature alignment error: {e}")
                    input_data = input_data.reindex(columns=feature_columns, fill_value=0)
            
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
                st.write("Error details:", str(e))
                
                # Use backup prediction
                try:
                    prediction, probability = backup_prediction(input_data)
                    
                    st.warning("Using backup prediction method due to model issues.")
                    
                    # Display results
                    st.markdown("---")
                    st.header("🎯 Prediction Results (Backup Method)")
                    
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
                        
                except Exception as backup_error:
                    st.error(f"Backup prediction also failed: {backup_error}")
                    st.write("Backup error details:", str(backup_error))
    
    else:
        st.error("⚠️ Model or feature columns not loaded properly.")
        st.write("**Using Rule-Based Backup Prediction**")
        
        # Create input form for backup prediction
        with st.form("backup_prediction_form"):
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
            
            submit_button = st.form_submit_button("🔮 Predict Placement (Backup)")
        
        if submit_button:
            # Prepare input data for backup prediction
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
            
            # Use backup prediction
            try:
                prediction, probability = backup_prediction(input_data)
                
                # Display results
                st.markdown("---")
                st.header("🎯 Prediction Results (Rule-Based)")
                
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
                    
            except Exception as backup_error:
                st.error(f"Even backup prediction failed: {backup_error}")

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
