"""
Credit Risk Model Dashboard - Enhanced with Pre-Processing Visualizations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
from datetime import datetime
import os
# Page configuration
st.set_page_config(
    page_title="Credit Risk Model Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # streamlit/
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # repo root
DATA_DIR = os.path.join(ROOT_DIR, "data")
# Custom CSS
st.markdown("""
<style>
    /* Màu cho các số liệu metric */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: bold;
        font-size: 28px;
            
    }
    
    [data-testid="stMetricLabel"] {
        color: #CCCCCC !important;
        font-size: 14px;
    }
    
    [data-testid="stMetricDelta"] {
        color: #4CAF50 !important;
    }
    
    /* Màu cho badge stages - theo CSS bạn cung cấp */
    .stage-badge {
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
        color: white !important;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        text-align: center;
    }
    
    .stage-raw {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .stage-cleaned {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border: 1px solid rgba(240, 147, 251, 0.3);
    }
    
    .stage-trained {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border: 1px solid rgba(79, 172, 254, 0.3);
    }
    
    /* Container cho badge để căn giữa */
    .badge-container {
        display: flex;
        justify-content: center;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_resource
def load_model_artifacts():
    """Load all model artifacts"""
    try:
        model_dir = Path(os.path.join(DATA_DIR, "/models"))
        
        # Get latest files
        model_files = sorted(model_dir.glob('logistic_regression_*.pkl'))
        scaler_files = sorted(model_dir.glob('scaler_*.pkl'))
        encoder_files = sorted(model_dir.glob('label_encoders_*.pkl'))
        feature_files = sorted(model_dir.glob('feature_names_*.pkl'))
        
        if not all([model_files, scaler_files, feature_files]):
            return None, None, None, None, "Model files not found"
        
        # Load latest artifacts
        model = joblib.load(model_files[-1])
        scaler = joblib.load(scaler_files[-1])
        feature_names = joblib.load(feature_files[-1])
        label_encoders = joblib.load(encoder_files[-1]) if encoder_files else {}
        
        return model, scaler, feature_names, label_encoders, None
    except Exception as e:
        return None, None, None, None, str(e)

@st.cache_data
def load_training_results():
    """Load training results"""
    try:
        report_dir = Path(os.path.join(DATA_DIR, '/reports'))
        result_files = sorted(report_dir.glob('training_results_*.json'))
        
        if not result_files:
            return None
        
        with open(result_files[-1], 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading training results: {e}")
        return None

@st.cache_data
def load_raw_data():
    """Load raw data (before cleaning)"""
    try:
        data_path = Path(os.path.join(DATA_DIR, "application_train.csv"))
        if data_path.exists():
            return pd.read_csv(data_path)
        return None
    except Exception as e:
        st.error(f"Error loading raw data: {e}")
        return None

@st.cache_data
def load_processed_data():
    """Load processed data (after cleaning, before training)"""
    try:
        data_path = Path(os.path.join(DATA_DIR, "df_processed.csv"))
        if data_path.exists():
            return pd.read_csv(data_path)
        return None
    except Exception as e:
        st.error(f"Error loading processed data: {e}")
        return None

@st.cache_data
def load_training_data():
    """Load final training data (after feature engineering)"""
    try:
        data_path = Path(os.path.join(DATA_DIR, 'df_final.csv'))
        if data_path.exists():
            return pd.read_csv(data_path)
        return None
    except Exception as e:
        st.error(f"Error loading training data: {e}")
        return None

def create_gauge_chart(value, title, max_value=1):
    """Create a gauge chart for metrics"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.5], 'color': '#ffcccc'},
                {'range': [0.5, 0.7], 'color': '#ffffcc'},
                {'range': [0.7, max_value], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.5
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_missing_values_plot(df, title="Missing Values Analysis"):
    """Create missing values visualization"""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        return None
    
    # Take top 20 for readability
    missing_top = missing.head(20)
    missing_pct = (missing_top / len(df) * 100)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=missing_pct.values,
        y=missing_pct.index,
        orientation='h',
        marker=dict(
            color=missing_pct.values,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Missing %")
        ),
        text=[f'{v:.1f}%' for v in missing_pct.values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Missing Percentage (%)',
        yaxis_title='Feature',
        height=max(400, len(missing_top) * 25),
        showlegend=False
    )
    
    return fig

def create_data_quality_summary(df, stage_name):
    """Create data quality summary metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        st.metric("Total Features", len(df.columns))
    
    with col3:
        missing_total = df.isnull().sum().sum()
        missing_pct = (missing_total / (len(df) * len(df.columns))) * 100
        st.metric("Missing Values", f"{missing_total:,}", delta=f"{missing_pct:.2f}%")
    
    with col4:
        cols_with_missing = (df.isnull().sum() > 0).sum()
        st.metric("Columns with Missing", cols_with_missing)

def create_target_comparison(df_list, names):
    """Compare target distribution across datasets"""
    fig = go.Figure()
    
    for df, name in zip(df_list, names):
        if df is not None and 'TARGET' in df.columns:
            target_counts = df['TARGET'].value_counts()
            fig.add_trace(go.Bar(
                name=name,
                x=['No Default', 'Default'],
                y=[target_counts.get(0, 0), target_counts.get(1, 0)]
            ))
    
    fig.update_layout(
        title='Target Distribution Comparison Across Stages',
        xaxis_title='Class',
        yaxis_title='Count',
        barmode='group',
        height=400
    )
    
    return fig

def create_feature_distribution_comparison(df_list, names, feature):
    """Compare feature distribution across datasets"""
    fig = make_subplots(
        rows=1, cols=len(df_list),
        subplot_titles=names
    )
    
    for idx, (df, name) in enumerate(zip(df_list, names), 1):
        if df is not None and feature in df.columns:
            fig.add_trace(
                go.Histogram(x=df[feature], name=name, nbinsx=50),
                row=1, col=idx
            )
    
    fig.update_layout(
        title_text=f'Distribution of {feature} Across Stages',
        showlegend=False,
        height=400
    )
    
    return fig

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/bank-card-back-side.png", width=100)
    st.title("🏦 Credit Risk Model")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📊 Data Quality",
            "🔍 Before Cleaning",
            "🧹 After Cleaning", 
            "🎯 Before Training",
            "📈 Model Performance",
            "🔮 Make Prediction",
            "ℹ️ About"
        ]
    )
    
    st.markdown("---")
    st.markdown("### Data Processing Stages")
    st.markdown('<div class="badge-container">Raw Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="badge-container">Cleaned Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="badge-container">Model Ready</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Model Info")
    
    # Load model info
    results = load_training_results()
    if results:
        st.metric("Model Type", "Logistic Regression")
        if 'test_metrics' in results:
            st.metric("Test ROC-AUC", f"{results['test_metrics']['roc_auc']:.4f}")

# Main content
if page == "🏠 Home":
    st.title("💳 Credit Risk Assessment Dashboard")
    st.markdown("### Welcome to the Credit Risk Model Dashboard")
    
    # Data pipeline overview
    st.markdown("### 📊 Data Pipeline Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="badge-container">Stage 1: Raw Data</div>', unsafe_allow_html=True)
        raw_df = load_raw_data()
        if raw_df is not None:
            st.metric("Records", f"{len(raw_df):,}")
            st.metric("Features", len(raw_df.columns))
            missing_pct = (raw_df.isnull().sum().sum() / (len(raw_df) * len(raw_df.columns))) * 100
            st.metric("Missing", f"{missing_pct:.1f}%")
    
    with col2:
        st.markdown('<div class="badge-container">Stage 2: Cleaned Data</div>', unsafe_allow_html=True)
        processed_df = load_processed_data()
        if processed_df is not None:
            st.metric("Records", f"{len(processed_df):,}")
            st.metric("Features", len(processed_df.columns))
            missing_pct = (processed_df.isnull().sum().sum() / (len(processed_df) * len(processed_df.columns))) * 100
            st.metric("Missing", f"{missing_pct:.1f}%")
    
    with col3:
        st.markdown('<div class="badge-container">Stage 3: Model Ready</div>', unsafe_allow_html=True)
        final_df = load_training_data()
        if final_df is not None:
            st.metric("Records", f"{len(final_df):,}")
            st.metric("Features", len(final_df.columns) - 1)
            st.metric("Missing", "0%")
    
    st.markdown("---")
    
    # Model Performance
    results = load_training_results()
    
    if results and 'test_metrics' in results:
        st.markdown("### 🎯 Model Performance")
        metrics = results['test_metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        
        with col2:
            st.metric("ROC-AUC Score", f"{metrics['roc_auc']:.4f}")
        
        with col3:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        
        with col4:
            st.metric("Recall", f"{metrics['recall']:.4f}")
    
    st.markdown("---")
    
    # Key Features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Dashboard Features")
        st.markdown("""
        - **Data Quality Analysis**: Compare raw vs cleaned data
        - **Pre-Processing Insights**: Visualize data transformations
        - **Model Performance**: Track all metrics
        - **Real-time Predictions**: Instant risk assessments
        - **Interactive Visualizations**: Explore data at each stage
        """)
    
    with col2:
        st.markdown("### 📊 Analysis Stages")
        st.markdown("""
        1. **Before Cleaning**: Raw data analysis with missing values
        2. **After Cleaning**: Data quality improvements
        3. **Before Training**: Final feature engineering
        4. **After Training**: Model performance and predictions
        """)

elif page == "📊 Data Quality":
    st.title("📊 Data Quality Dashboard")
    st.markdown("### Compare data quality across all processing stages")
    
    # Load all datasets
    raw_df = load_raw_data()
    processed_df = load_processed_data()
    final_df = load_training_data()
    
    # Overall comparison
    st.markdown("### 📈 Overall Comparison")
    
    comparison_data = []
    
    if raw_df is not None:
        comparison_data.append({
            'Stage': 'Raw Data',
            'Records': len(raw_df),
            'Features': len(raw_df.columns),
            'Missing Values': raw_df.isnull().sum().sum(),
            'Missing %': f"{(raw_df.isnull().sum().sum() / (len(raw_df) * len(raw_df.columns)) * 100):.2f}%",
            'Columns with Missing': (raw_df.isnull().sum() > 0).sum()
        })
    
    if processed_df is not None:
        comparison_data.append({
            'Stage': 'Cleaned Data',
            'Records': len(processed_df),
            'Features': len(processed_df.columns),
            'Missing Values': processed_df.isnull().sum().sum(),
            'Missing %': f"{(processed_df.isnull().sum().sum() / (len(processed_df) * len(processed_df.columns)) * 100):.2f}%",
            'Columns with Missing': (processed_df.isnull().sum() > 0).sum()
        })
    
    if final_df is not None:
        comparison_data.append({
            'Stage': 'Model Ready',
            'Records': len(final_df),
            'Features': len(final_df.columns),
            'Missing Values': final_df.isnull().sum().sum(),
            'Missing %': f"{(final_df.isnull().sum().sum() / (len(final_df) * len(final_df.columns)) * 100):.2f}%",
            'Columns with Missing': (final_df.isnull().sum() > 0).sum()
        })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualize improvements
        st.markdown("---")
        st.markdown("### 📉 Data Quality Improvements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Records comparison
            fig = px.bar(
                comparison_df,
                x='Stage',
                y='Records',
                title='Records Count Across Stages',
                color='Stage',
                color_discrete_map={
                    'Raw Data': '#ffc107',
                    'Cleaned Data': '#0d6efd',
                    'Model Ready': '#198754'
                }
            )
            st.plotly_chart(fig, use_container_width=True, key='dataquality_bar1')
        
        with col2:
            # Missing values comparison
            fig = px.bar(
                comparison_df,
                x='Stage',
                y='Missing Values',
                title='Missing Values Reduction',
                color='Stage',
                color_discrete_map={
                    'Raw Data': '#ffc107',
                    'Cleaned Data': '#0d6efd',
                    'Model Ready': '#198754'
                }
            )
            st.plotly_chart(fig, use_container_width=True, key='dataquality_bar2')
        
        # Target distribution comparison
        st.markdown("---")
        st.markdown("### 🎯 Target Distribution Across Stages")
        
        datasets = [raw_df, processed_df, final_df]
        names = ['Raw Data', 'Cleaned Data', 'Model Ready']
        available_datasets = [(df, name) for df, name in zip(datasets, names) if df is not None]
        
        if available_datasets:
            fig = create_target_comparison(
                [df for df, _ in available_datasets],
                [name for _, name in available_datasets]
            )
            st.plotly_chart(fig, use_container_width=True, key='dataquality_bar3')

elif page == "🔍 Before Cleaning":
    st.title("🔍 Raw Data Analysis (Before Cleaning)")
    st.markdown('<div class="stage-badge stage-raw">Stage 1: Raw Data</div>', unsafe_allow_html=True)
    
    raw_df = load_raw_data()
    
    if raw_df is None:
        st.error("Raw data not found. Please ensure application_train.csv exists in data/ directory.")
    else:
        # Data quality summary
        st.markdown("### 📊 Data Quality Summary")
        create_data_quality_summary(raw_df, "Raw Data")
        
        st.markdown("---")
        
        # Missing values analysis
        st.markdown("### 🔴 Missing Values Analysis")
        
        missing_fig = create_missing_values_plot(raw_df, "Top 20 Features with Missing Values")
        if missing_fig:
            st.plotly_chart(missing_fig, use_container_width=True)
            
            # Missing values statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Missing Values Statistics")
                total_cells = len(raw_df) * len(raw_df.columns)
                missing_cells = raw_df.isnull().sum().sum()
                st.metric("Total Cells", f"{total_cells:,}")
                st.metric("Missing Cells", f"{missing_cells:,}")
                st.metric("Missing Percentage", f"{(missing_cells/total_cells*100):.2f}%")
            
            with col2:
                st.markdown("#### Columns by Missing %")
                missing_summary = raw_df.isnull().sum()
                missing_summary = missing_summary[missing_summary > 0]
                missing_pct = (missing_summary / len(raw_df) * 100).sort_values(ascending=False)
                
                bins = ['0-10%', '10-25%', '25-50%', '50-75%', '75-100%']
                counts = [
                    ((missing_pct > 0) & (missing_pct <= 10)).sum(),
                    ((missing_pct > 10) & (missing_pct <= 25)).sum(),
                    ((missing_pct > 25) & (missing_pct <= 50)).sum(),
                    ((missing_pct > 50) & (missing_pct <= 75)).sum(),
                    (missing_pct > 75).sum()
                ]
                
                fig = px.bar(x=bins, y=counts, title='Distribution of Missing Values',
                           labels={'x': 'Missing %', 'y': 'Number of Columns'})
                st.plotly_chart(fig, use_container_width=True, key='before_bar4')
        else:
            st.success("✅ No missing values found!")
        
        st.markdown("---")
        
        # Target distribution
        if 'TARGET' in raw_df.columns:
            st.markdown("### 🎯 Target Distribution (Raw)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                target_counts = raw_df['TARGET'].value_counts()
                fig = px.bar(
                    x=['No Default', 'Default'],
                    y=target_counts.values,
                    title='Target Class Distribution',
                    color=['No Default', 'Default'],
                    color_discrete_map={'No Default': 'green', 'Default': 'red'}
                )
                st.plotly_chart(fig, use_container_width=True, key='before_bar2')
            
            with col2:
                fig = px.pie(
                    values=target_counts.values,
                    names=['No Default', 'Default'],
                    title='Target Distribution (%)',
                    color_discrete_map={
                        'No Default': 'green',
                        'Default': 'red'
                    }
                )
                st.plotly_chart(fig, use_container_width=True, key='before_bar5')
            
            # Class imbalance metrics
            default_rate = raw_df['TARGET'].mean()
            st.info(f"📊 Default Rate: {default_rate*100:.2f}% | Class Balance Ratio: 1:{(1/default_rate):.1f}")
        
        st.markdown("---")
        
        # Numerical features distribution
        st.markdown("### 📊 Numerical Features (Sample)")
        
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'TARGET' in numeric_cols:
            numeric_cols.remove('TARGET')
        if 'SK_ID_CURR' in numeric_cols:
            numeric_cols.remove('SK_ID_CURR')
        
        selected_features = st.multiselect(
            "Select features to visualize:",
            numeric_cols,
            default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols[:2]
        )
        
        if selected_features:
            n_cols = 2
            n_rows = (len(selected_features) + 1) // 2
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=selected_features
            )
            
            for idx, feature in enumerate(selected_features):
                row = idx // n_cols + 1
                col = idx % n_cols + 1
                
                fig.add_trace(
                    go.Histogram(x=raw_df[feature], name=feature, nbinsx=50),
                    row=row, col=col
                )
            
            fig.update_layout(
                height=300 * n_rows,
                showlegend=False,
                title_text="Feature Distributions (Raw Data)"
            )
            
            st.plotly_chart(fig, use_container_width=True, key='before_features_hist')
        
        st.markdown("---")
        
        # Data preview
        st.markdown("### 📋 Data Preview (First 100 rows)")
        st.dataframe(raw_df.head(100), use_container_width=True)
        
        # Data types
        st.markdown("### 🔢 Data Types")
        dtype_df = raw_df.dtypes.value_counts().reset_index()
        dtype_df.columns = ["Data Type", "Count"]
        # fix
        dtype_df["Data Type"] = dtype_df["Data Type"].astype(str)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            fig = px.pie(
                dtype_df,
                values="Count",
                names="Data Type",
                title="Distribution of Data Types"
            )
            st.plotly_chart(fig, use_container_width=True, key="before_dtype_pie")

elif page == "🧹 After Cleaning":
    st.title("🧹 Data After Cleaning")
    st.markdown('<div class="stage-badge stage-cleaned">Stage 2: Cleaned Data</div>', unsafe_allow_html=True)
    
    processed_df = load_processed_data()
    raw_df = load_raw_data()
    
    if processed_df is None:
        st.error("Processed data not found. Please run data cleaning scripts first.")
    else:
        # Data quality summary
        st.markdown("### 📊 Data Quality Summary")
        create_data_quality_summary(processed_df, "Cleaned Data")
        
        # Show improvements
        if raw_df is not None:
            st.markdown("---")
            st.markdown("### ✨ Data Quality Improvements")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                records_change = len(processed_df) - len(raw_df)
                st.metric(
                    "Records Change",
                    f"{records_change:,}",
                    delta=f"{(records_change/len(raw_df)*100):.2f}%"
                )
            
            with col2:
                features_change = len(processed_df.columns) - len(raw_df.columns)
                st.metric(
                    "Features Change",
                    f"{features_change:+}",
                    delta="Features added/removed"
                )
            
            with col3:
                raw_missing = raw_df.isnull().sum().sum()
                proc_missing = processed_df.isnull().sum().sum()
                missing_reduction = raw_missing - proc_missing
                st.metric(
                    "Missing Values Reduced",
                    f"{missing_reduction:,}",
                    delta=f"{(missing_reduction/raw_missing*100):.1f}% reduction"
                )
            
            with col4:
                raw_missing_cols = (raw_df.isnull().sum() > 0).sum()
                proc_missing_cols = (processed_df.isnull().sum() > 0).sum()
                st.metric(
                    "Cols with Missing",
                    proc_missing_cols,
                    delta=f"{raw_missing_cols - proc_missing_cols} fewer"
                )
        
        st.markdown("---")
        
        # Missing values (if any remain)
        st.markdown("### 🔍 Remaining Missing Values")
        
        missing_fig = create_missing_values_plot(processed_df, "Features with Missing Values (After Cleaning)")
        if missing_fig:
            st.plotly_chart(missing_fig, use_container_width=True)
            st.warning("⚠️ Some missing values remain. These will be handled during training.")
        else:
            st.success("✅ No missing values! Data is clean.")
        
        st.markdown("---")
        
        # Target distribution
        if 'TARGET' in processed_df.columns:
            st.markdown("### 🎯 Target Distribution (After Cleaning)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                target_counts = processed_df['TARGET'].value_counts()
                fig = px.bar(
                    x=['No Default', 'Default'],
                    y=target_counts.values,
                    title='Target Class Distribution',
                    color=['No Default', 'Default'],
                    color_discrete_map={'No Default': 'green', 'Default': 'red'}
                )
                st.plotly_chart(fig, use_container_width=True, key='after_clean_bar1')
            
            with col2:
                fig = px.pie(
                    values=target_counts.values,
                    names=['No Default', 'Default'],
                    title='Target Distribution (%)',
                    color=['No Default', 'Default'],
                    color_discrete_map={'No Default': 'green', 'Default': 'red'}
                )
                st.plotly_chart(fig, use_container_width=True, key='after_clean_pie1')
        
        # Feature comparison (if raw data available)
        if raw_df is not None:
            st.markdown("---")
            st.markdown("### 📊 Before vs After Comparison")
            
            # Find common numerical features
            raw_numeric = set(raw_df.select_dtypes(include=[np.number]).columns)
            proc_numeric = set(processed_df.select_dtypes(include=[np.number]).columns)
            common_numeric = list((raw_numeric & proc_numeric) - {'TARGET', 'SK_ID_CURR'})
            
            if common_numeric:
                selected_feature = st.selectbox(
                    "Select feature to compare:",
                    common_numeric
                )
                
                if selected_feature:
                    fig = create_feature_distribution_comparison(
                        [raw_df, processed_df],
                        ['Before Cleaning', 'After Cleaning'],
                        selected_feature
                    )
                    st.plotly_chart(fig, use_container_width=True, key='after_clean_comparison')
                    
                    # Statistics comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Before Cleaning")
                        st.dataframe(raw_df[selected_feature].describe(), use_container_width=True)
                    
                    with col2:
                        st.markdown("#### After Cleaning")
                        st.dataframe(processed_df[selected_feature].describe(), use_container_width=True)
        
        st.markdown("---")
        
        # Data preview
        st.markdown("### 📋 Data Preview (First 100 rows)")
        st.dataframe(processed_df.head(100), use_container_width=True)

elif page == "🎯 Before Training":
    st.title("🎯 Data Before Training")
    st.markdown('<div class="stage-badge stage-trained">Stage 3: Model Ready</div>', unsafe_allow_html=True)
    
    final_df = load_training_data()
    
    if final_df is None:
        st.error("Training data not found. Please run the feature engineering script first.")
    else:
        # Data quality summary
        st.markdown("### 📊 Final Dataset Summary")
        create_data_quality_summary(final_df, "Training Data")
        
        st.markdown("---")
        
        # Feature engineering summary
        st.markdown("### 🔧 Feature Engineering Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Selected Features")
            st.info(f"**{len(final_df.columns) - 1}** features selected for modeling")
            
            feature_list = [col for col in final_df.columns if col != 'TARGET']
            st.dataframe(
                pd.DataFrame({'Feature': feature_list}),
                use_container_width=True,
                height=300
            )
        
        with col2:
            st.markdown("#### Data Characteristics")
            st.metric("No Missing Values", "✅")
            st.metric("All Numeric", "✅")
            st.metric("Ready for Training", "✅")
            
            # Data types
            dtype_counts = final_df.dtypes.value_counts()
            st.dataframe(dtype_counts.to_frame('Count'), use_container_width=True)
        
        st.markdown("---")
        
        # Target distribution
        if 'TARGET' in final_df.columns:
            st.markdown("### 🎯 Final Target Distribution")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                target_counts = final_df['TARGET'].value_counts()
                fig = px.bar(
                    x=['No Default', 'Default'],
                    y=target_counts.values,
                    title='Target Class Distribution',
                    color=['No Default', 'Default'],
                    color_discrete_map={'No Default': 'green', 'Default': 'red'}
                )
                st.plotly_chart(fig, use_container_width=True, key='before_train_bar1')
            
            with col2:
                fig = px.pie(
                    values=target_counts.values,
                    names=['No Default', 'Default'],
                    title='Target Distribution (%)',
                    color=['No Default', 'Default'],
                    color_discrete_map={'No Default': 'green', 'Default': 'red'}
                )
                st.plotly_chart(fig, use_container_width=True, key='before_train_pie1')
            
            with col3:
                st.markdown("#### Class Balance")
                default_rate = final_df['TARGET'].mean()
                no_default = target_counts.get(0, 0)
                default = target_counts.get(1, 0)
                
                st.metric("No Default", f"{no_default:,}")
                st.metric("Default", f"{default:,}")
                st.metric("Default Rate", f"{default_rate*100:.2f}%")
                st.metric("Balance Ratio", f"1:{(1/default_rate):.1f}")
        
        st.markdown("---")
        
        # Feature distributions
        st.markdown("### 📊 Feature Distributions")
        
        numeric_cols = [col for col in final_df.columns if col != 'TARGET']
        
        selected_features = st.multiselect(
            "Select features to visualize:",
            numeric_cols,
            default=numeric_cols[:6] if len(numeric_cols) >= 6 else numeric_cols
        )
        
        if selected_features:
            n_cols = 2
            n_rows = (len(selected_features) + 1) // 2
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=selected_features
            )
            
            for idx, feature in enumerate(selected_features):
                row = idx // n_cols + 1
                col = idx % n_cols + 1
                
                fig.add_trace(
                    go.Histogram(x=final_df[feature], name=feature, nbinsx=50),
                    row=row, col=col
                )
            
            fig.update_layout(
                height=300 * n_rows,
                showlegend=False,
                title_text="Final Feature Distributions"
            )
            
            st.plotly_chart(fig, use_container_width=True, key='before_train_hist')
        
        st.markdown("---")
        
        # Correlation analysis
        st.markdown("### 🔗 Feature Correlations")
        
        corr_features = st.multiselect(
            "Select features for correlation analysis:",
            numeric_cols,
            default=numeric_cols[:8] if len(numeric_cols) >= 8 else numeric_cols,
            key='corr_select'
        )
        
        if len(corr_features) >= 2:
            corr_matrix = final_df[corr_features].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect='auto',
                title='Correlation Heatmap',
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1
            )
            st.plotly_chart(fig, use_container_width=True, key='before_train_corr')
        
        st.markdown("---")
        
        # Statistical summary
        st.markdown("### 📈 Statistical Summary")
        st.dataframe(final_df.describe(), use_container_width=True)
        
        st.markdown("---")
        
        # Data preview
        st.markdown("### 📋 Data Preview (First 100 rows)")
        st.dataframe(final_df.head(100), use_container_width=True)

elif page == "📈 Model Performance":
    st.title("📈 Model Performance Analysis")
    st.markdown('<div class="stage-badge stage-trained">Model Trained</div>', unsafe_allow_html=True)
    
    results = load_training_results()
    
    if results is None:
        st.error("Training results not found. Please train the model first.")
        st.info("Run: `cd training && python train.py`")
    else:
        # Performance Overview
        st.markdown("### 🎯 Performance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        train_metrics = results.get('train_metrics', {})
        test_metrics = results.get('test_metrics', {})
        
        with col1:
            fig = create_gauge_chart(test_metrics.get('accuracy', 0), "Accuracy")
            st.plotly_chart(fig, use_container_width=True, key='perf_gauge1')
        
        with col2:
            fig = create_gauge_chart(test_metrics.get('precision', 0), "Precision")
            st.plotly_chart(fig, use_container_width=True, key='perf_gauge2')
        
        with col3:
            fig = create_gauge_chart(test_metrics.get('recall', 0), "Recall")
            st.plotly_chart(fig, use_container_width=True, key='perf_gauge3')
        
        with col4:
            fig = create_gauge_chart(test_metrics.get('roc_auc', 0), "ROC-AUC")
            st.plotly_chart(fig, use_container_width=True, key='perf_gauge4')
        
        st.markdown("---")
        
        # Detailed Metrics Comparison
        st.markdown("### 📋 Train vs Test Metrics")
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Train': [
                train_metrics.get('accuracy', 0),
                train_metrics.get('precision', 0),
                train_metrics.get('recall', 0),
                train_metrics.get('f1', 0),
                train_metrics.get('roc_auc', 0)
            ],
            'Test': [
                test_metrics.get('accuracy', 0),
                test_metrics.get('precision', 0),
                test_metrics.get('recall', 0),
                test_metrics.get('f1', 0),
                test_metrics.get('roc_auc', 0)
            ]
        })
        
        metrics_df['Difference'] = metrics_df['Train'] - metrics_df['Test']
        metrics_df['Overfitting'] = metrics_df['Difference'].apply(
            lambda x: '🔴 Yes' if x > 0.05 else '🟢 No'
        )
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Train', x=metrics_df['Metric'], y=metrics_df['Train'], marker_color='lightblue'))
        fig.add_trace(go.Bar(name='Test', x=metrics_df['Metric'], y=metrics_df['Test'], marker_color='darkblue'))
        
        fig.update_layout(
            barmode='group',
            title='Train vs Test Metrics Comparison',
            xaxis_title='Metric',
            yaxis_title='Score',
            height=400,
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True, key='perf_comparison')
        
        # Metrics Table
        st.dataframe(
            metrics_df.style.format({
                'Train': '{:.4f}',
                'Test': '{:.4f}',
                'Difference': '{:.4f}'
            }),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Cross-Validation Results
        if 'cv_scores' in results:
            st.markdown("### 🔄 Cross-Validation Results")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                cv_scores = results['cv_scores']
                cv_df = pd.DataFrame({
                    'Fold': [f'Fold {i+1}' for i in range(len(cv_scores))],
                    'ROC-AUC Score': cv_scores
                })
                
                fig = px.line(cv_df, x='Fold', y='ROC-AUC Score', markers=True,
                             title='Cross-Validation ROC-AUC Scores')
                fig.add_hline(y=results['cv_mean'], line_dash="dash", 
                            annotation_text=f"Mean: {results['cv_mean']:.4f}",
                            line_color="red")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key='perf_cv')
            
            with col2:
                st.metric("Mean CV Score", f"{results['cv_mean']:.4f}")
                st.metric("Std CV Score", f"{results['cv_std']:.4f}")
                st.metric("Min Score", f"{min(cv_scores):.4f}")
                st.metric("Max Score", f"{max(cv_scores):.4f}")
                
                # CV consistency
                cv_std = results['cv_std']
                if cv_std < 0.01:
                    st.success("🟢 Highly Consistent")
                elif cv_std < 0.02:
                    st.info("🟡 Moderately Consistent")
                else:
                    st.warning("🔴 Variable Performance")
        
        st.markdown("---")
        
        # Hyperparameters
        if 'hyperparameters' in results:
            st.markdown("### ⚙️ Model Hyperparameters")
            hyper_df = pd.DataFrame([results['hyperparameters']]).T
            hyper_df.columns = ['Value']
            st.dataframe(hyper_df, use_container_width=True)
        
        # Training info
        if 'training_time' in results:
            st.markdown("### ⏱️ Training Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Training Time", f"{results['training_time']:.2f} seconds")
            
            with col2:
                st.metric("Model Type", "Logistic Regression")

elif page == "🔮 Make Prediction":
    st.title("🔮 Credit Risk Prediction")
    st.markdown("### Enter applicant details to assess credit risk")
    
    # Load model
    model, scaler, feature_names, label_encoders, error = load_model_artifacts()
    
    if error:
        st.error(f"Error loading model: {error}")
        st.info("Please train the model first by running: `cd training && python train.py`")
    else:
        # Create input form
        st.markdown("### 📝 Applicant Information")
        
        # Use columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            amt_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
            amt_credit = st.number_input("Credit Amount ($)", min_value=0, value=200000, step=5000)
            amt_goods = st.number_input("Goods Price ($)", min_value=0, value=180000, step=5000)
            ext_source_1 = st.slider("External Source 1 Score", 0.0, 1.0, 0.5, 0.01)
            ext_source_2 = st.slider("External Source 2 Score", 0.0, 1.0, 0.5, 0.01)
        
        with col2:
            ext_source_3 = st.slider("External Source 3 Score", 0.0, 1.0, 0.5, 0.01)
            ext_1_missing = st.checkbox("External Source 1 Missing", value=False)
            ext_2_missing = st.checkbox("External Source 2 Missing", value=False)
            ext_3_missing = st.checkbox("External Source 3 Missing", value=False)
            is_retired = st.checkbox("Retired with No Occupation", value=False)
            is_working = st.checkbox("Working with No Occupation", value=False)
        
        # Calculate derived feature
        loan_ratio = amt_credit / amt_goods if amt_goods > 0 else 0
        st.info(f"📊 Calculated Loan-to-Value Ratio: {loan_ratio:.2%}")
        
        # Prediction button
        if st.button("🎯 Predict Credit Risk", type="primary"):
            # Prepare input data
            input_data = pd.DataFrame({
                'AMT_INCOME_TOTAL': [amt_income],
                'AMT_CREDIT': [amt_credit],
                'AMT_GOODS_PRICE': [amt_goods],
                'Tỉ lệ vay so với nhu cầu': [loan_ratio],
                'EXT_SOURCE_1': [ext_source_1],
                'EXT_SOURCE_2': [ext_source_2],
                'EXT_SOURCE_3': [ext_source_3],
                'EXT_SOURCE_1_is_missing': [int(ext_1_missing)],
                'EXT_SOURCE_2_is_missing': [int(ext_2_missing)],
                'EXT_SOURCE_3_is_missing': [int(ext_3_missing)],
                'IS_RETIRED_NO_OCCUPATION': [int(is_retired)],
                'IS_WORKING_NO_OCCUPATION': [int(is_working)]
            })
            
            # Align with training features
            for feat in feature_names:
                if feat not in input_data.columns:
                    input_data[feat] = 0
            
            input_data = input_data[feature_names]
            
            try:
                # Scale features
                input_scaled = scaler.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0]
                
                # Display results
                st.markdown("---")
                st.markdown("### 🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.markdown('<div class="prediction-box rejected"><h3>⚠️ HIGH RISK</h3><p>Default Predicted</p></div>', 
                                   unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="prediction-box approved"><h3>✅ LOW RISK</h3><p>No Default Predicted</p></div>', 
                                   unsafe_allow_html=True)
                
                with col2:
                    st.metric("Default Probability", f"{probability[1]*100:.2f}%")
                    st.progress(probability[1])
                
                with col3:
                    st.metric("No Default Probability", f"{probability[0]*100:.2f}%")
                    st.progress(probability[0])
                
                # Risk gauge
                fig = create_gauge_chart(probability[1], "Risk Score", max_value=1)
                st.plotly_chart(fig, use_container_width=True, key='prediction_gauge')
                
                # Recommendation
                st.markdown("### 💡 Recommendation")
                if probability[1] > 0.7:
                    st.error("🚫 **REJECT**: High risk of default. Consider requesting additional collateral or co-signer.")
                elif probability[1] > 0.5:
                    st.warning("⚠️ **REVIEW**: Moderate risk. Manual review recommended. Consider stricter terms.")
                else:
                    st.success("✅ **APPROVE**: Low risk of default. Standard terms applicable.")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Please ensure all model files are properly loaded.")

elif page == "ℹ️ About":
    st.title("ℹ️ About This Dashboard")
    
    st.markdown("""
    ## Credit Risk Assessment Model
    
    ### 🎯 Purpose
    This dashboard provides a comprehensive interface for credit risk assessment using machine learning,
    with full visibility into the data pipeline from raw data to final predictions.
    
    ### 📊 Dashboard Stages
    
    1. **🔍 Before Cleaning**: Analyze raw data with all its imperfections
       - Missing values analysis
       - Initial data quality assessment
       - Original feature distributions
    
    2. **🧹 After Cleaning**: Review data quality improvements
       - Missing value reduction
       - Data cleaning impact
       - Feature transformations
    
    3. **🎯 Before Training**: Examine model-ready data
       - Final feature set
       - Engineered features
       - Data ready for modeling
    
    4. **📈 Model Performance**: Evaluate trained model
       - Accuracy, precision, recall metrics
       - Cross-validation results
       - Overfitting analysis
    
    ### 🤖 Model Details
    - **Algorithm**: Logistic Regression with balanced class weights
    - **Task**: Binary Classification (Default vs Non-Default)
    - **Validation**: 5-fold Stratified Cross-Validation
    - **Features**: Income, credit amount, external scores, employment status
    
    ### 🔧 Technical Stack
    - **Framework**: Streamlit
    - **ML Library**: scikit-learn
    - **Visualization**: Plotly
    - **Data Processing**: pandas, numpy
    
    ### 📝 How to Use
    1. **Data Quality**: Compare data across all processing stages
    2. **Before Cleaning**: Understand raw data challenges
    3. **After Cleaning**: See data quality improvements
    4. **Before Training**: Review final features
    5. **Model Performance**: Analyze model metrics
    6. **Make Prediction**: Get real-time risk assessments
    
    ### 🚀 Deployment
    This dashboard can be deployed on:
    - Streamlit Cloud (Free)
    - Heroku
    - AWS/GCP/Azure
    - Docker containers
    
    ---
    
    ### 📄 Model Performance Summary
    """)
    
    results = load_training_results()
    if results and 'test_metrics' in results:
        metrics = results['test_metrics']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Classification Metrics")
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            st.metric("Precision", f"{metrics['precision']:.4f}")
            st.metric("Recall", f"{metrics['recall']:.4f}")
            st.metric("F1-Score", f"{metrics['f1']:.4f}")
        
        with col2:
            st.markdown("#### Performance Metrics")
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            st.metric("Average Precision", f"{metrics['avg_precision']:.4f}")
            
            if 'cv_mean' in results:
                st.metric("CV Mean Score", f"{results['cv_mean']:.4f}")
                st.metric("CV Std Score", f"{results['cv_std']:.4f}")
    
    st.markdown("---")
    
    # Data pipeline info
    st.markdown("### 📁 Data Files")
    
    data_files = {
        'application_train.csv': 'Raw training data (before any processing)',
        'df_processed.csv': 'Cleaned data (after handling missing values)',
        'df_final.csv': 'Model-ready data (after feature engineering)',
        'models/*.pkl': 'Trained model artifacts',
        'reports/*.json': 'Training results and metrics'
    }
    
    for file, description in data_files.items():
        st.markdown(f"- **{file}**: {description}")
    
    st.markdown("---")
    st.markdown("*Dashboard Version 2.0 | Last Updated: " + datetime.now().strftime("%Y-%m-%d") + "*") 