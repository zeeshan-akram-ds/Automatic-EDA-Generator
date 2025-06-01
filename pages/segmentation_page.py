import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from uuid import uuid4
import matplotlib.pyplot as plt
# Page setup
st.title("Customer Segmentation with KMeans")
st.markdown(
    """
    Upload a CSV file, configure clustering settings, and discover customer segments interactively.
    Follow the steps below to preprocess and analyze your data.
    """
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'artifact_id' not in st.session_state:
    st.session_state.artifact_id = str(uuid4())

# Modular functions
def handle_missing_values(df, selected_features):
    """Handle missing values based on user choice."""
    missing_values = df[selected_features].isnull().sum()
    if missing_values.sum() > 0:
        st.warning("Missing values detected in selected features:")
        st.write(missing_values[missing_values > 0])
        strategy = st.radio(
            "Choose how to handle missing values:",
            ["Drop rows with missing values", "Impute numerical with mean, categorical with mode"],
            key="missing_strategy",
            help="Drop removes rows with missing values. Impute fills numerical with mean and categorical with mode."
        )
        if strategy == "Drop rows with missing values":
            df = df.dropna(subset=selected_features)
            st.success("Rows with missing values dropped successfully.")
        else:
            numerical_cols = [col for col in selected_features if df[col].dtype in ['int64', 'float64']]
            categorical_cols = [col for col in selected_features if df[col].dtype == 'object']
            for col in numerical_cols:
                df[col] = df[col].fillna(df[col].mean())
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
            st.success("Missing values imputed successfully.")
    else:
        st.success("No missing values detected in selected features!")
    return df

def encode_categoricals(df, selected_categorical):
    """Encode categorical features based on user choice."""
    if not selected_categorical:
        return df, selected_categorical
    encoding_method = st.selectbox(
        "Choose encoding method for categorical features:",
        ["Label Encoding", "One-Hot Encoding"],
        key="encoding_method",
        help="Label Encoding assigns integers to categories. One-Hot Encoding creates binary columns."
    )
    if encoding_method == "Label Encoding":
        for col in selected_categorical:
            df[col] = LabelEncoder().fit_transform(df[col])
        st.success("Categorical features encoded using Label Encoding.")
        return df, selected_categorical
    else:
        df = pd.get_dummies(df, columns=selected_categorical, prefix=selected_categorical)
        new_cols = [col for col in df.columns if any(cat in col for cat in selected_categorical)]
        st.success("Categorical features encoded using One-Hot Encoding.")
        return df, new_cols

def scale_data(df, selected_features, scaler_option):
    """Scale the selected features."""
    scaler = StandardScaler() if scaler_option == "StandardScaler" else MinMaxScaler()
    X_scaled = scaler.fit_transform(df[selected_features])
    return X_scaled, scaler

def run_kmeans(X_scaled, k):
    """Run KMeans clustering and return labels and model."""
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    return kmeans.labels_, kmeans

def plot_elbow(X_scaled):
    """Plot elbow curve to help choose optimal k."""
    inertia = []
    K = range(2, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(K), 
            y=inertia, 
            mode='lines+markers',
            line=dict(color='royalblue', width=2),
            marker=dict(size=8)
        )
    )
    fig.update_layout(
        title="Elbow Curve for Optimal k",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia",
        template="plotly_white",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

def visualize_pca(X_scaled, labels, selected_features):
    """Visualize clusters in 2D using PCA."""
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(components, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = labels.astype(str)
    explained_variance = pca.explained_variance_ratio_.sum() * 100
    fig = px.scatter(
        df_pca, 
        x='PC1', 
        y='PC2', 
        color='Cluster',
        title=f"Cluster Visualization (PCA, {explained_variance:.1f}% Variance Explained)",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig, use_container_width=True)

# Sidebar for configuration
st.sidebar.header("Clustering Configuration")
st.sidebar.markdown("Set parameters for feature selection, scaling, and clustering.", help="Use this panel to configure your clustering process.")

# Step 1: File upload
st.header("Step 1: Upload Your Data")
uploaded_file = st.file_uploader(
    "Upload a CSV file", 
    type=["csv"], 
    help="Upload a CSV file containing your dataset."
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
            st.stop()
        else:
            st.session_state.df = df
            st.success("File uploaded successfully!")
            with st.expander("Data Preview"):
                st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error reading file: {str(e)}. Please ensure it's a valid CSV.")
        st.stop()
else:
    st.info("Please upload a CSV file to begin.")
    st.stop()

# Step 2: Feature selection (in sidebar)
df = st.session_state.df.copy()
st.header("Step 2: Select Features")
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

with st.sidebar:
    st.subheader("Feature Selection")
    selected_numerical = st.multiselect(
        "Select Numerical Features",
        options=numerical_cols,
        help="Choose numerical features for clustering."
    )
    selected_categorical = st.multiselect(
        "Select Categorical Features",
        options=categorical_cols,
        help="Choose categorical features to encode and include in clustering."
    )

selected_features = selected_numerical + selected_categorical

if not selected_features:
    st.info("Please select at least two features from the sidebar to proceed.")
    st.stop()
elif len(selected_features) < 2:
    st.warning("Please select at least two features for meaningful clustering.")
    st.stop()
elif not selected_numerical and selected_categorical:
    st.warning(
        "You have selected only categorical features. KMeans may not be ideal for purely categorical data. "
        "Consider including numerical features or using a different clustering algorithm."
    )

# Step 3: Handle missing values
st.header("Step 3: Handle Missing Values")
df = handle_missing_values(df, selected_features)

# Step 4: Encode categorical features
if selected_categorical:
    st.header("Step 4: Encode Categorical Features")
    df, selected_features = encode_categoricals(df, selected_categorical)
    selected_features = selected_numerical + selected_features

# Step 5: Configure scaling
st.header("Step 5: Configure Scaling")
scaler_option = st.sidebar.selectbox(
    "Select Scaling Method",
    ["StandardScaler", "MinMaxScaler"],
    help="StandardScaler standardizes features (mean=0, std=1). MinMaxScaler scales to [0,1]."
)

# Preview scaled data
X_scaled, scaler = scale_data(df, selected_features, scaler_option)
with st.expander("Preview Scaled Data"):
    st.dataframe(pd.DataFrame(X_scaled, columns=selected_features).head(), use_container_width=True)

# Step 6: Select number of clusters
k = st.sidebar.slider(
    "Number of Clusters (k)",
    min_value=2, 
    max_value=10, 
    value=3,
    help="Select the number of clusters for KMeans."
)

# Step 7: Run clustering
st.header("Step 6: Run KMeans Clustering")
if st.button("Run Clustering", help="Click to perform KMeans clustering with the specified settings."):
    with st.spinner("Running KMeans clustering..."):
        try:
            # Run KMeans
            labels, kmeans = run_kmeans(X_scaled, k)
            df['Cluster'] = labels
            silhouette_avg = silhouette_score(X_scaled, labels)
            
            st.success("Clustering completed successfully!")
            st.markdown(f"🔹 **Silhouette Score**: `{silhouette_avg:.3f}` (Range: [-1, 1], higher is better)")
            
            # Step 7: Display results
            st.header("Step 7: Clustering Results")
            
            # Cluster assignments
            with st.expander("Cluster Assignments"):
                display_cols = ['Cluster'] + selected_numerical
                st.dataframe(df[display_cols].head(), use_container_width=True)
            
            # Cluster summary
            with st.expander("Cluster Summary (Averages)"):
                summary_cols = selected_numerical if selected_categorical else selected_features
                if summary_cols:
                    summary = df.groupby('Cluster')[summary_cols].mean()
                    st.dataframe(summary.style.background_gradient(cmap='YlGnBu'), use_container_width=True)
            
            # Elbow curve
            with st.expander("Elbow Curve for Optimal k"):
                plot_elbow(X_scaled)
            
            # PCA visualization
            with st.expander("Cluster Visualization"):
                visualize_pca(X_scaled, labels, selected_features)
            
            # Cluster centers
            if not selected_categorical or scaler_option == "Label Encoding":
                with st.expander("Cluster Centers"):
                    centers = scaler.inverse_transform(kmeans.cluster_centers_)
                    centers_df = pd.DataFrame(centers, columns=selected_features)
                    st.dataframe(centers_df.style.background_gradient(cmap='YlOrRd'), use_container_width=True)
                    st.subheader("📊 Cluster Centers Comparison")
                    fig, ax = plt.subplots(figsize=(10, 5))

                    for i in range(centers_df.shape[0]):
                        ax.plot(centers_df.columns, centers_df.iloc[i], marker='o', label=f'Cluster {i}')
                        
                    ax.set_title('Cluster Centers by Feature')
                    ax.set_ylabel('Feature Values')
                    ax.legend()
                    st.pyplot(fig)
            # Step 8: Download results
            st.header("Step 8: Download Results")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Clustered Data",
                csv,
                "clustered_data.csv",
                "text/csv",
                help="Download the dataset with cluster labels."
            )
        
        except Exception as e:
            st.error(f"Error during clustering: {str(e)}. Please check your inputs and try again.")