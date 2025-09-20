import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set up the page
st.set_page_config(page_title="Jamboree Admission Analysis", layout="wide")

# Title and description
st.title("Jamboree Admission Analysis Dashboard")
st.markdown("""
This dashboard provides insights into the Jamboree Admission dataset, which contains information about 
student profiles and their chances of admission to graduate programs.
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Jamboree_Admission.csv')
    df = df.drop(columns=['Serial No.'])
    return df

df = load_data()

st.sidebar.header("Overview Dataset")

# Display dataset information
if st.sidebar.checkbox("Show Dataset Overview"):
    st.header("Dataset Overview")
    st.write(f"Shape of the dataset: {df.shape}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("First 10 rows")
        st.dataframe(df.head(10))
    
    with col2:
        st.subheader("Dataset Information")
        st.write("**Columns:**", list(df.columns))
        st.write("**Numerical Columns:**", df.select_dtypes(include=np.number).columns.tolist())
        st.write("**Missing Values:**", df.isnull().sum().sum())

# Show basic statistics
if st.sidebar.checkbox("Show Basic Statistics"):
    st.header("Basic Statistics")
    st.dataframe(df.describe())

# Show unique values
if st.sidebar.checkbox("Show Unique Values"):
    st.header("Unique Values in Each Column")
    st.dataframe(df.nunique())

st.sidebar.header("Try Prediction")

if st.sidebar.checkbox("Predict"):
    st.header("Prediction Section")
    st.write("Use the inputs below to make a prediction using one of the saved pipelines.")

    # Map readable model names to filenames
    model_files = {
        'Linear Regression': 'model_pipeline.pkl',
        'Polynomial Regression': 'Polynomial_pipeline.pkl',
        'Ridge Regression': 'Ridge_pipeline.pkl'
    }

    @st.cache_data
    def load_model(path: str):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    model_name = st.selectbox('Choose model for prediction', list(model_files.keys()))

    # Use dataset statistics as sensible defaults
    gre_default = int(df['GRE Score'].mean()) if 'GRE Score' in df else 320
    toefl_default = int(df['TOEFL Score'].mean()) if 'TOEFL Score' in df else 110
    univ_default = int(df['University Rating'].mode()[0]) if 'University Rating' in df else 3
    sop_default = float(df['SOP'].mean()) if 'SOP' in df else 3.0
    lor_default = float(df['LOR '].mean()) if 'LOR ' in df else 3.0
    cgpa_default = float(df['CGPA'].mean()) if 'CGPA' in df else 8.5
    research_default = int(df['Research'].mode()[0]) if 'Research' in df else 0

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        gre = st.number_input('GRE Score', min_value=0, max_value=340, value=gre_default)
        toefl = st.number_input('TOEFL Score', min_value=0, max_value=120, value=toefl_default)
        univ = st.selectbox('University Rating', options=sorted(df['University Rating'].unique()) if 'University Rating' in df else [1,2,3,4,5], index=0)
    with col2:
        sop = st.slider('SOP (Statement of Purpose)', min_value=1.0, max_value=5.0, step=0.5, value=round(sop_default*2)/2)
        lor = st.slider('LOR (Letter of Recommendation Strength)', min_value=1.0, max_value=5.0, step=0.5, value=round(lor_default*2)/2)
        cgpa = st.number_input('CGPA', min_value=0.0, max_value=10.0, value=round(cgpa_default,2), step=0.01, format="%.2f")
    with col3:
        research = st.selectbox('Research Experience', options=[0,1], format_func=lambda x: 'Yes' if x==1 else 'No', index=research_default)

    if st.button('Predict'):
        model_path = model_files.get(model_name)
        model = load_model(model_path)
        if model is None:
            st.error(f"Model file not found: {model_path}. Make sure the .pkl file exists in the app directory.")
        else:
            # Prepare input in the same order used during training
            X_input = np.array([[gre, toefl, univ, sop, lor, cgpa, research]])
            try:
                pred = model.predict(X_input)
                if hasattr(pred, '__len__'):
                    pred_val = float(pred[0])
                else:
                    pred_val = float(pred)
                # Clip to [0,1] for chance
                pred_val = max(0.0, min(1.0, pred_val))

                st.success(f"Predicted Chance of Admit: {pred_val:.3f} ({pred_val*100:.1f}%)")
                # visual progress bar
                prog = int(pred_val * 100)
                st.progress(prog)
            except Exception as e:
                st.exception(e)

# Visualizations
st.sidebar.header("Visualizations")

# Distribution plots
if st.sidebar.checkbox("Show Distribution Plots"):
    st.header("Distribution of Numerical Features")
    
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_col = st.selectbox("Select a feature to visualize", num_cols)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[selected_col], kde=True, ax=ax)
    ax.set_title(f"Distribution of {selected_col}")
    st.pyplot(fig)

# Box plots
if st.sidebar.checkbox("Show Box Plots"):
    st.header("Box Plots of Features")
    
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_col = st.selectbox("Select a feature for box plot", num_cols)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=df[selected_col], ax=ax)
    ax.set_title(f"Box Plot of {selected_col}")
    st.pyplot(fig)

# Correlation heatmap
if st.sidebar.checkbox("Show Correlation Heatmap"):
    st.header("Correlation Heatmap")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title("Correlation Between Features")
    st.pyplot(fig)

# Pairplot
if st.sidebar.checkbox("Show Pairplot (Sample)"):
    st.header("Pairplot of Numerical Features (Sample Data)")
    st.warning("This might take a while for large datasets. Showing sample of 100 records.")
    
    sample_df = df.sample(n=100, random_state=42)
    fig = sns.pairplot(sample_df)
    st.pyplot(fig)

# Chance of Admit analysis
st.sidebar.header("Admission Chance Analysis")

if st.sidebar.checkbox("Show Chance of Admit Analysis"):
    st.header("Chance of Admit Analysis")
    
    # Histogram of Chance of Admit
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Chance of Admit '], kde=True, ax=ax)
    ax.set_title("Distribution of Chance of Admit")
    ax.axvline(df['Chance of Admit '].mean(), color='red', linestyle='--', label=f"Mean: {df['Chance of Admit '].mean():.2f}")
    ax.legend()
    st.pyplot(fig)
    
    # Relationship between features and chance of admit
    feature = st.selectbox("Select feature to compare with Chance of Admit", 
                          ['GRE Score', 'TOEFL Score', 'CGPA', 'University Rating', 'SOP', 'LOR ', 'Research'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df[feature], y=df['Chance of Admit '], ax=ax)
    ax.set_title(f"{feature} vs Chance of Admit")
    ax.set_xlabel(feature)
    ax.set_ylabel("Chance of Admit")
    st.pyplot(fig)

# Categorical analysis
if st.sidebar.checkbox("Show Categorical Analysis"):
    st.header("Categorical Feature Analysis")
    
    cat_features = ['University Rating', 'Research']
    selected_cat = st.selectbox("Select categorical feature", cat_features)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    if selected_cat == 'Research':
        df['Research'] = df['Research'].map({0: 'No', 1: 'Yes'})
    
    sns.boxplot(x=df[selected_cat], y=df['Chance of Admit '], ax=ax)
    ax.set_title(f"Chance of Admit by {selected_cat}")
    st.pyplot(fig)

# Insights from the notebook
if st.sidebar.checkbox("Show Key Insights"):
    st.header("Key Insights from the Analysis")
    
    insights = """
    - The dataset contains 500 records with 8 features after removing the Serial No. column
    - There are no missing values in the dataset
    - University Rating, SOP, LOR, and Research can be considered as categorical variables
    - No significant outliers were found in the dataset based on box plot analysis
    - CGPA, GRE Score, and TOEFL Score show strong positive correlation with Chance of Admit
    - Research experience appears to have a positive impact on admission chances
    - Higher university ratings are associated with higher chances of admission
    """
    
    st.markdown(insights)



# Footer
st.sidebar.markdown("---")
st.sidebar.info("Jamboree Admission Analysis Dashboard | Created with Streamlit")

# Main area footer
st.markdown("---")
st.markdown("### About")
st.markdown("""
This dashboard provides an interactive exploration of the Jamboree Admission dataset, 
which contains information about student profiles and their chances of admission to graduate programs.

**Features included:**
- Dataset overview and basic statistics
- Try out different regression models for prediction
- Distribution visualizations
- Correlation analysis
- Admission chance analysis
- Key insights from the data
""")