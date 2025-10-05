"""
Streamlit Exoplanet Classification App
- Upload CSV file with exoplanet data
- Send to FastAPI backend for classification
- Display results with exoplanet_status as first column
- Validate and edit headers if incorrect
"""

import streamlit as st
import pandas as pd
import requests
import io
from typing import Dict, List

# ---------------------------
# Configuration
# ---------------------------
st.set_page_config(
    page_title="Exoplanet Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .header-valid {
        background-color: #d4edda;
        color: #155724;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 4px 0;
        font-family: monospace;
        font-weight: 500;
    }
    .header-invalid {
        background-color: #f8d7da;
        color: #721c24;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 4px 0;
        font-family: monospace;
        font-weight: 500;
    }
    .header-extra {
        background-color: #fff3cd;
        color: #856404;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 4px 0;
        font-family: monospace;
        font-weight: 500;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
    }
    .nav-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .nav-button {
        background: transparent;
        border: none;
        color: white;
        padding: 12px 24px;
        margin: 0 4px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        flex: 1;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .nav-button:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .nav-button.active {
        background: rgba(255, 255, 255, 0.2);
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
    }
    .nav-button.active::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        border-radius: 0 0 8px 8px;
    }
    .nav-icon {
        margin-right: 8px;
        font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

# FastAPI backend URL
API_URL = "http://16.16.128.44:8000"

# Expected headers for inference (fetch from backend or hardcode)
EXPECTED_HEADERS_INFERENCE = [
    "pl_rade",      # Planet Radius (Earth radii)
    "pl_trandep",   # Transit Depth (%)
    "pl_orbper",    # Orbital Period (days)
    "pl_trandurh",  # Transit Duration (hours)
    "pl_insol",     # Insolation Flux (Earth flux)
    "pl_eqt",       # Equilibrium Temperature (K)
    "st_rad",       # Stellar Radius (Solar radii)
    "st_logg",      # Stellar Surface Gravity (log10(cm/s^2))
    "st_teff",      # Stellar Effective Temperature (K)
    "st_tmag",      # TESS Magnitude
    "st_dist"       # Distance to Star (parsecs)
]

# Expected headers for training (specific features + target variable)
EXPECTED_HEADERS_TRAINING = [
    "pl_orbper",    # Orbital Period (days)
    "pl_trandurh",  # Transit Duration (hours)
    "pl_rade",      # Planet Radius (Earth radii)
    "st_dist",      # Distance to Star (parsecs)
    "st_pmdec",     # Stellar Proper Motion Declination (mas/yr)
    "st_pmra",      # Stellar Proper Motion Right Ascension (mas/yr)
    "dec",          # Declination (degrees)
    "pl_insol",     # Insolation Flux (Earth flux)
    "pl_tranmid",   # Transit Midpoint (BJD)
    "ra",           # Right Ascension (degrees)
    "st_tmag",      # TESS Magnitude
    "pl_trandep",   # Transit Depth (%)
    "pl_eqt",       # Equilibrium Temperature (K)
    "st_rad",       # Stellar Radius (Solar radii)
    "st_logg",      # Stellar Surface Gravity (log10(cm/s^2))
    "st_teff",      # Stellar Effective Temperature (K)
    "exoplanet_status"  # Exoplanet Status (target variable for training)
]

# Backward compatibility
EXPECTED_HEADERS = EXPECTED_HEADERS_INFERENCE

# Header descriptions for tooltips
HEADER_DESCRIPTIONS = {
    "pl_orbper": "Orbital Period (days)",
    "pl_trandurh": "Transit Duration (hours)",
    "pl_rade": "Planet Radius (Earth radii)",
    "st_dist": "Distance to Star (parsecs)",
    "st_pmdec": "Stellar Proper Motion Declination (mas/yr)",
    "st_pmra": "Stellar Proper Motion Right Ascension (mas/yr)",
    "dec": "Declination (degrees)",
    "pl_insol": "Insolation Flux (Earth flux)",
    "pl_tranmid": "Transit Midpoint (BJD)",
    "ra": "Right Ascension (degrees)",
    "st_tmag": "TESS Magnitude",
    "pl_trandep": "Transit Depth (%)",
    "pl_eqt": "Equilibrium Temperature (K)",
    "st_rad": "Stellar Radius (Solar radii)",
    "st_logg": "Stellar Surface Gravity (log10(cm/s²))",
    "st_teff": "Stellar Effective Temperature (K)",
    "exoplanet_status": "Exoplanet Status (CONFIRMED, FALSE POSITIVE, or CANDIDATE)"
}

# ---------------------------
# Helper Functions
# ---------------------------

def fetch_expected_headers(api_url: str) -> List[str]:
    """Fetch expected headers from FastAPI backend"""
    try:
        response = requests.get(f"{api_url}/expected-headers", timeout=5)
        if response.status_code == 200:
            return response.json().get("headers", EXPECTED_HEADERS)
    except:
        pass
    return EXPECTED_HEADERS

def classify_exoplanets(file_content: bytes, filename: str, api_url: str) -> pd.DataFrame:
    """Send CSV to FastAPI backend for classification"""
    try:
        # Ensure API URL doesn't have trailing slash
        api_url = api_url.rstrip('/')

        files = {"csv_file": (filename, file_content, "text/csv")}
        headers = {"Accept": "text/csv"}
        response = requests.post(
            f"{api_url}/inference/classify-csv",
            files=files,
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            # Check if response is JSON (error with header info) or CSV
            content_type = response.headers.get('content-type', '')
            if 'application/json' in content_type:
                return response.json()  # Return error dict
            else:
                # Parse CSV response
                df = pd.read_csv(io.StringIO(response.text))
                return df
        else:
            return {"error": f"Classification failed (HTTP {response.status_code}): {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def train_model(file_content: bytes, filename: str, api_url: str, hyperparameters: Dict = None) -> Dict:
    """Send CSV to FastAPI backend for model training"""
    try:
        # Ensure API URL doesn't have trailing slash
        api_url = api_url.rstrip('/')

        files = {"csv_file": (filename, file_content, "text/csv")}
        headers = {"Accept": "application/json"}

        # Prepare data dictionary with hyperparameters if provided
        data = {}
        if hyperparameters:
            import json
            data["hyperparameters"] = json.dumps(hyperparameters)

        response = requests.post(
            f"{api_url}/model/train",
            files=files,
            data=data,
            headers=headers,
            timeout=60  # Training might take longer
        )

        if response.status_code == 200:
            # Parse JSON response with training results
            return response.json()
        else:
            return {"error": f"Training failed (HTTP {response.status_code}): {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def load_models(api_url: str) -> List[Dict]:
    """Load list of models from FastAPI backend"""
    try:
        # Ensure API URL doesn't have trailing slash
        api_url = api_url.rstrip('/')

        response = requests.get(
            f"{api_url}/inference/get-models",
            timeout=10
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to load models (HTTP {response.status_code}): {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def upload_dataset(file_content: bytes, filename: str, api_url: str) -> Dict:
    """Upload CSV dataset to FastAPI backend"""
    try:
        # Ensure API URL doesn't have trailing slash
        api_url = api_url.rstrip('/')

        files = {"csv_file": (filename, file_content, "text/csv")}
        response = requests.post(
            f"{api_url}/dataset/upload-dataset",
            files=files,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Upload failed (HTTP {response.status_code}): {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def load_datasets(api_url: str) -> Dict:
    """Load list of datasets from FastAPI backend"""
    try:
        # Ensure API URL doesn't have trailing slash
        api_url = api_url.rstrip('/')

        response = requests.get(
            f"{api_url}/dataset/get-datasets",
            timeout=10
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to load datasets (HTTP {response.status_code}): {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def display_dataframe_styled(df: pd.DataFrame):
    """Display DataFrame with beautiful styling"""
    # Reorder columns to show label_confidence first if it exists
    if 'label_confidence' in df.columns:
        cols = ['label_confidence'] + [col for col in df.columns if col != 'label_confidence']
        df = df[cols]

    # Apply styling
    def highlight_status(val):
        if val == 'CONFIRMED':
            return 'background-color: #d4edda; color: #155724; font-weight: bold'
        elif val == 'FALSE POSITIVE':
            return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
        elif val == 'CANDIDATE':
            return 'background-color: #fff3cd; color: #856404; font-weight: bold'
        return ''

    # Style the dataframe
    if 'exoplanet_status' in df.columns:
        styled_df = df.style.applymap(
            highlight_status,
            subset=['exoplanet_status']
        ).format(precision=4)
        st.dataframe(styled_df, use_container_width=True, height=400)
    else:
        st.dataframe(df, use_container_width=True, height=400)

def render_editable_headers(df: pd.DataFrame, page_type: str = "inference"):
    """Render editable header interface with color coding"""
    st.markdown("### ✏️ Edit Column Headers")
    st.markdown("**Instructions:** Edit header names to match required format. Valid headers appear in green, invalid in red, extra in yellow.")

    # Select the appropriate expected headers based on page type
    if page_type == "training":
        expected_headers = EXPECTED_HEADERS_TRAINING
    else:
        expected_headers = EXPECTED_HEADERS_INFERENCE

    # Initialize session state for headers if not exists
    if 'edited_headers' not in st.session_state:
        st.session_state.edited_headers = df.columns.tolist()

    # Create columns for better layout
    num_cols = 3
    cols_per_row = st.columns(num_cols)

    edited_headers = []
    header_status = []

    for idx, original_header in enumerate(df.columns):
        col_idx = idx % num_cols
        with cols_per_row[col_idx]:
            # Get current value
            current_value = st.session_state.edited_headers[idx] if idx < len(st.session_state.edited_headers) else original_header

            # Determine header status
            is_valid = current_value in expected_headers
            is_required = current_value in expected_headers

            # Color code the label
            if is_valid:
                label_html = f'<div class="header-valid">✓ {current_value}</div>'
                status = 'valid'
            elif current_value in edited_headers:
                label_html = f'<div class="header-invalid">✗ {current_value} (duplicate)</div>'
                status = 'duplicate'
            else:
                # Check if it could be extra
                if original_header not in expected_headers and current_value not in expected_headers:
                    label_html = f'<div class="header-extra">⚠ {current_value} (extra)</div>'
                    status = 'extra'
                else:
                    label_html = f'<div class="header-invalid">✗ {current_value}</div>'
                    status = 'invalid'

            st.markdown(label_html, unsafe_allow_html=True)

            # Text input for editing
            new_value = st.text_input(
                f"Column {idx + 1}",
                value=current_value,
                key=f"header_{idx}",
                label_visibility="collapsed",
                help=f"Original: {original_header}"
            )

            edited_headers.append(new_value)
            header_status.append(status)

    # Update session state
    st.session_state.edited_headers = edited_headers

    # Check if all required headers are present
    missing_required = [h for h in expected_headers if h not in edited_headers]
    extra_headers = [h for h in edited_headers if h not in expected_headers]
    duplicate_headers = [h for h in edited_headers if edited_headers.count(h) > 1]

    st.markdown("---")

    # Display status summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        valid_count = sum(1 for h in edited_headers if h in expected_headers)
        st.metric("✅ Valid Headers", f"{valid_count}/{len(expected_headers)}")
    with col2:
        st.metric("❌ Missing Required", len(missing_required))
    with col3:
        st.metric("⚠️ Extra Headers", len(extra_headers))
    with col4:
        st.metric("🔄 Duplicates", len(set(duplicate_headers)))

    # Show details
    if missing_required:
        st.markdown("**❌ Missing Required Headers:**")
        cols = st.columns(3)
        for idx, header in enumerate(missing_required):
            with cols[idx % 3]:
                st.code(f"{header}\n{HEADER_DESCRIPTIONS.get(header, '')}", language=None)

    # Extra headers are allowed - no need to show warning
    # if extra_headers:
    #     st.markdown("**⚠️ Extra Headers (will be included but not used for classification):**")
    #     st.info(", ".join(extra_headers))

    if duplicate_headers:
        st.error(f"**🔄 Duplicate Headers Found:** {', '.join(set(duplicate_headers))}")

    # Return validation status and edited headers
    can_classify = len(missing_required) == 0 and len(duplicate_headers) == 0

    return {
        'can_classify': can_classify,
        'edited_headers': edited_headers,
        'missing': missing_required,
        'extra': extra_headers,
        'duplicates': duplicate_headers
    }

# ---------------------------
# Navigation
# ---------------------------
# Initialize session state for current page
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Inference'

# Sidebar logo/banner
st.sidebar.markdown("""
<div style="text-align: center; padding: 0.8rem 0 0.5rem 0;">
    <div style="font-size: 2.5rem; margin-bottom: 0.3rem;">
        🪐
    </div>
    <div style="font-size: 1.2rem; font-weight: 700;
                background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.2rem;">
        Exoplanet AI
    </div>
    <div style="font-size: 0.7rem; color: #6c757d; letter-spacing: 1px;">
        CLASSIFICATION SYSTEM
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Sidebar navigation
st.sidebar.markdown("## 🧭 Navigation")
st.sidebar.markdown("")  # Add spacing

# Create navigation buttons with custom styling - vertical layout
inference_active = st.session_state.current_page == 'Inference'
if st.sidebar.button(
    "🔮 Inference",
    use_container_width=True,
    key="nav_inference_btn",
    type="primary" if inference_active else "secondary"
):
    if not inference_active:
        st.session_state.current_page = 'Inference'
        st.rerun()

training_active = st.session_state.current_page == 'Training'
if st.sidebar.button(
    "🎓 Training",
    use_container_width=True,
    key="nav_training_btn",
    type="primary" if training_active else "secondary"
):
    if not training_active:
        st.session_state.current_page = 'Training'
        st.rerun()

models_active = st.session_state.current_page == 'Models'
if st.sidebar.button(
    "🤖 Models",
    use_container_width=True,
    key="nav_models_btn",
    type="primary" if models_active else "secondary"
):
    if not models_active:
        st.session_state.current_page = 'Models'
        st.rerun()

dataset_active = st.session_state.current_page == 'Dataset'
if st.sidebar.button(
    "📊 Dataset",
    use_container_width=True,
    key="nav_dataset_btn",
    type="primary" if dataset_active else "secondary"
):
    if not dataset_active:
        st.session_state.current_page = 'Dataset'
        st.rerun()

help_active = st.session_state.current_page == 'Help'
if st.sidebar.button(
    "❓ Help",
    use_container_width=True,
    key="nav_help_btn",
    type="primary" if help_active else "secondary"
):
    if not help_active:
        st.session_state.current_page = 'Help'
        st.rerun()

st.sidebar.markdown("")  # Add spacing
st.sidebar.markdown("---")

# ---------------------------
# Main UI
# ---------------------------
st.markdown('<h1 class="main-header">🪐 Exoplanet Classification System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced machine learning classification for exoplanet candidate analysis</p>', unsafe_allow_html=True)

# Sidebar info
st.sidebar.markdown("## 📊 About This Tool")
st.sidebar.markdown("""
This application uses machine learning to classify exoplanet candidates into three categories:

- **✅ CONFIRMED** - Verified exoplanet
- **❌ FALSE POSITIVE** - Not an exoplanet
- **🔍 CANDIDATE** - Requires additional data

Built for astronomical researchers and data scientists.
""")

st.sidebar.markdown("---")

# Show expected headers based on current page
with st.sidebar.expander("📋 Required CSV Headers", expanded=False):
    if st.session_state.current_page == 'Training':
        st.markdown("**Training CSV must contain these columns:**")
        headers_to_show = EXPECTED_HEADERS_TRAINING
    else:
        st.markdown("**Inference CSV must contain these columns:**")
        headers_to_show = EXPECTED_HEADERS_INFERENCE
    
    for i, header in enumerate(headers_to_show, 1):
        description = HEADER_DESCRIPTIONS.get(header, "")
        st.markdown(f"**{i}. `{header}`**")
        st.caption(description)

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Tips:**")
st.sidebar.info("""
- Headers are case-sensitive
- Extra columns are allowed
- Missing values will be handled automatically
- Edit headers inline if they don't match
""")

# ---------------------------
# Page Content
# ---------------------------
if st.session_state.current_page == 'Inference':
    # ---------------------------
    # Inference Page
    # ---------------------------
    st.markdown("### 🤖 Model Selection")

    # Load models for dropdown
    col1, col2 = st.columns([4, 1])
    with col1:
        # Get available models
        if 'models_list' not in st.session_state:
            st.info("💡 Click 'Load Models' to fetch available models from the backend.")
        else:
            models_response = st.session_state['models_list']
            if isinstance(models_response, dict) and 'data' in models_response:
                models_data = models_response['data']
                model_names = [model.get('name', f'Model {idx}') for idx, model in models_data.items()]

                if model_names:
                    selected_model = st.selectbox(
                        "Select Model",
                        options=model_names,
                        help="Choose which model to use for classification"
                    )
                    st.session_state['selected_model'] = selected_model
                else:
                    st.warning("⚠️ No models available. Please train a model first.")
            else:
                st.warning("⚠️ Failed to load models. Click 'Load Models' to retry.")

    with col2:
        if st.button("🔄 Load Models", use_container_width=True, key="inference_load_models"):
            with st.spinner("🔄 Loading models..."):
                models = load_models(API_URL)
                if isinstance(models, dict) and "error" in models:
                    st.error(f"❌ {models['error']}")
                else:
                    st.session_state['models_list'] = models
                    st.success("✅ Models loaded!")
                    st.rerun()
    st.markdown("---")

    st.markdown("### 📊 Dataset Selection")

    # Load datasets for dropdown
    col1, col2 = st.columns([4, 1])
    with col1:
        # Get available datasets
        if 'datasets_list' not in st.session_state:
            st.info("💡 Click 'Load Datasets' to fetch available datasets from the backend.")
        else:
            datasets_response = st.session_state['datasets_list']
            if isinstance(datasets_response, dict) and 'data' in datasets_response:
                datasets_data = datasets_response['data']
                dataset_names = [dataset.get('name', f'Dataset {idx}') for idx, dataset in datasets_data.items()]

                if dataset_names:
                    selected_dataset = st.selectbox(
                        "Select Dataset",
                        options=dataset_names,
                        help="Choose which dataset to use for classification"
                    )
                    st.session_state['selected_dataset'] = selected_dataset
                else:
                    st.warning("⚠️ No datasets available. Please upload a dataset first.")
            else:
                st.warning("⚠️ Failed to load datasets. Click 'Load Datasets' to retry.")

    with col2:
        if st.button("🔄 Load Datasets", use_container_width=True, key="inference_load_datasets"):
            with st.spinner("🔄 Loading datasets..."):
                datasets = load_datasets(API_URL)
                if isinstance(datasets, dict) and "error" in datasets:
                    st.error(f"❌ {datasets['error']}")
                else:
                    st.session_state['datasets_list'] = datasets
                    st.success("✅ Datasets loaded!")
                    st.rerun()

    st.markdown("---")

    st.markdown("### 📤 Upload CSV File")

    uploaded_file = st.file_uploader(
        "Choose a CSV file containing exoplanet candidate measurements",
        type=["csv"],
        help="Upload a CSV file with exoplanet transit and stellar parameters"
    )

    if uploaded_file is not None:
        # Read file content
        file_content = uploaded_file.read()

        # Preview original CSV
        st.markdown("### 📊 Data Preview")
        try:
            df_original = pd.read_csv(io.BytesIO(file_content))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📝 Rows", f"{len(df_original):,}")
            with col2:
                st.metric("📋 Columns", len(df_original.columns))
            with col3:
                st.metric("💾 File Size", f"{len(file_content) / 1024:.1f} KB")

            # Show preview with option to expand
            with st.expander("🔍 View Data Sample", expanded=True):
                st.dataframe(df_original.head(10), use_container_width=True)

            st.markdown("---")

            # Editable headers section
            validation_result = render_editable_headers(df_original, "inference")

            # Classify button
            st.markdown("---")

            if validation_result['can_classify']:
                st.markdown('<div class="success-box"><strong>✅ Ready to classify!</strong> All required headers are present.</div>', unsafe_allow_html=True)

                if validation_result['extra']:
                    st.markdown('<div class="warning-box"><strong>⚠️ Note:</strong> Extra columns will be included in the output but not used for classification.</div>', unsafe_allow_html=True)

                if st.button("🚀 Classify Exoplanets", type="primary", use_container_width=True):
                    # Apply header renaming and convert to lowercase
                    df_renamed = df_original.copy()
                    df_renamed.columns = [h.lower() for h in validation_result['edited_headers']]

                    # Filter to only keep EXPECTED_HEADERS columns (lowercase)
                    required_headers_lower = [h.lower() for h in EXPECTED_HEADERS]
                    # Only keep columns that are in EXPECTED_HEADERS
                    df_renamed = df_renamed[[col for col in required_headers_lower if col in df_renamed.columns]]

                    # Convert to CSV
                    csv_buffer = io.StringIO()
                    df_renamed.to_csv(csv_buffer, index=False)
                    renamed_content = csv_buffer.getvalue().encode()

                    with st.spinner("🔄 Classifying exoplanets... This may take a moment."):
                        result = classify_exoplanets(renamed_content, uploaded_file.name, API_URL)

                        if isinstance(result, dict) and "error" in result:
                            st.error(f"❌ {result['error']}")
                        else:
                            st.session_state['classified_df'] = result
                            st.success("✅ Classification complete!")
                            st.rerun()
            else:
                st.markdown('<div class="warning-box"><strong>⚠️ Cannot classify yet</strong></div>', unsafe_allow_html=True)
                if validation_result['missing']:
                    st.error("❌ Please ensure all required headers are present and correctly named.")
                if validation_result['duplicates']:
                    st.error("❌ Duplicate header names detected. Each column must have a unique name.")

        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("💡 Please ensure your file is a valid CSV format.")

    # ---------------------------
    # Display Results
    # ---------------------------
    if 'classified_df' in st.session_state:
        st.markdown("---")
        st.markdown("### 🎯 Classification Results")

        df_result = st.session_state['classified_df']

        # Check if result is a valid DataFrame
        if isinstance(df_result, dict):
            st.error(f"❌ Unexpected response from backend: {df_result}")
            if st.button("Clear and Try Again"):
                del st.session_state['classified_df']
                st.rerun()
        else:
            # Statistics
            if 'exoplanet_status' in df_result.columns:
                st.markdown("#### 📈 Classification Summary")

                status_counts = df_result['exoplanet_status'].value_counts()

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("🌍 Total Candidates", f"{len(df_result):,}")
                with col2:
                    confirmed = status_counts.get('CONFIRMED', 0)
                    pct = (confirmed / len(df_result) * 100) if len(df_result) > 0 else 0
                    st.metric("✅ Confirmed", f"{confirmed:,}", f"{pct:.1f}%")
                with col3:
                    false_pos = status_counts.get('FALSE POSITIVE', 0)
                    pct = (false_pos / len(df_result) * 100) if len(df_result) > 0 else 0
                    st.metric("❌ False Positive", f"{false_pos:,}", f"{pct:.1f}%")
                with col4:
                    candidate = status_counts.get('CANDIDATE', 0)
                    pct = (candidate / len(df_result) * 100) if len(df_result) > 0 else 0
                    st.metric("🔍 Candidate", f"{candidate:,}", f"{pct:.1f}%")

            # Display results table
            st.markdown("#### 📊 Detailed Results")
            st.markdown(f"**Total columns:** {len(df_result.columns)} | **Rows:** {len(df_result):,}")
            display_dataframe_styled(df_result)

            # Download button
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                csv = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="exoplanet_classification_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:
                if st.button("🔄 Process New File", use_container_width=True):
                    # Clear session state
                    if 'classified_df' in st.session_state:
                        del st.session_state['classified_df']
                    if 'edited_headers' in st.session_state:
                        del st.session_state['edited_headers']
                    st.rerun()

elif st.session_state.current_page == 'Training':
    # ---------------------------
    # Training Page
    # ---------------------------
    st.markdown("### 🎓 Model Training")
    
    st.markdown("### 📤 Upload Training CSV File")
    
    uploaded_file_training = st.file_uploader(
        "Choose a CSV file containing exoplanet training data",
        type=["csv"],
        help="Upload a CSV file with exoplanet transit and stellar parameters for model training",
        key="training_file_uploader"
    )
    
    if uploaded_file_training is not None:
        # Read file content
        file_content_training = uploaded_file_training.read()
        
        # Preview original CSV
        st.markdown("### 📊 Training Data Preview")
        try:
            df_original_training = pd.read_csv(io.BytesIO(file_content_training))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📝 Rows", f"{len(df_original_training):,}")
            with col2:
                st.metric("📋 Columns", len(df_original_training.columns))
            with col3:
                st.metric("💾 File Size", f"{len(file_content_training) / 1024:.1f} KB")
            
            # Show preview with option to expand
            with st.expander("🔍 View Training Data Sample", expanded=True):
                st.dataframe(df_original_training.head(10), use_container_width=True)
            
            st.markdown("---")
            
            # Editable headers section
            validation_result_training = render_editable_headers(df_original_training, "training")
            
            # Training button
            st.markdown("---")
            
            if validation_result_training['can_classify']:
                st.markdown('<div class="success-box"><strong>✅ Ready for training!</strong> All required headers are present.</div>', unsafe_allow_html=True)
                
                if validation_result_training['extra']:
                    st.markdown('<div class="warning-box"><strong>⚠️ Note:</strong> Extra columns will be included in the output but not used for training.</div>', unsafe_allow_html=True)

                # Hyperparameter configuration section
                st.markdown("### ⚙️ Hyperparameter Configuration")

                # Default hyperparameters
                default_hyperparameters = {
                    'lgbm': {
                        'n_estimators': [200, 400],
                        'learning_rate': [0.05, 0.1],
                        'num_leaves': [31, 63],
                        'max_depth': [5, 10],
                        'subsample': [0.8, 1.0],
                        'colsample_bytree': [0.8, 1.0],
                    },
                    'gb': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.05, 0.1],
                        'max_depth': [3, 5],
                        'subsample': [0.8, 1.0]
                    }
                }

                # Initialize session state for hyperparameters
                import json
                if 'hyperparameters_json' not in st.session_state:
                    st.session_state.hyperparameters_json = json.dumps(default_hyperparameters, indent=2)

                # Expert mode toggle and reset button
                col1, col2 = st.columns([3, 1])
                with col1:
                    expert_mode = st.checkbox("🔬 Expert Mode (Custom JSON)", value=False, help="Enable to input hyperparameters as JSON")
                with col2:
                    if st.button("🔄 Reset", help="Reset hyperparameters to default values", use_container_width=True):
                        # Reset to default hyperparameters
                        st.session_state.hyperparameters_json = json.dumps(default_hyperparameters, indent=2)
                        st.success("✅ Reset to defaults!")
                        st.rerun()

                if expert_mode:
                    # Expert mode: JSON input
                    st.markdown("**📝 Edit Hyperparameters as JSON**")
                    st.caption("Modify the JSON below to customize hyperparameters for both LightGBM and Gradient Boosting models.")

                    hyperparameters_json = st.text_area(
                        "Hyperparameters JSON",
                        value=st.session_state.hyperparameters_json,
                        height=300,
                        help="Enter hyperparameters in JSON format",
                        label_visibility="collapsed",
                        key="hyperparams_text_area"
                    )

                    # Update session state
                    st.session_state.hyperparameters_json = hyperparameters_json

                    # Validate JSON
                    try:
                        hyperparameters = json.loads(hyperparameters_json)
                        st.success("✅ Valid JSON format")
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Invalid JSON: {str(e)}")
                        hyperparameters = default_hyperparameters
                else:
                    # Default mode: Display locked JSON
                    st.markdown("**📋 Default Hyperparameters**")
                    st.caption("Using default hyperparameter configuration. Enable Expert Mode to customize.")

                    st.text_area(
                        "Default Hyperparameters",
                        value=st.session_state.hyperparameters_json,
                        height=300,
                        disabled=True,
                        label_visibility="collapsed"
                    )

                    try:
                        hyperparameters = json.loads(st.session_state.hyperparameters_json)
                    except json.JSONDecodeError:
                        hyperparameters = default_hyperparameters

                st.markdown("---")

                if st.button("🚀 Start Training", type="primary", use_container_width=True):
                    # Apply header renaming and convert to lowercase
                    df_renamed_training = df_original_training.copy()
                    df_renamed_training.columns = [h.lower() for h in validation_result_training['edited_headers']]

                    # Filter to only keep EXPECTED_HEADERS_TRAINING columns (lowercase)
                    required_headers_training_lower = [h.lower() for h in EXPECTED_HEADERS_TRAINING]
                    # Only keep columns that are in EXPECTED_HEADERS_TRAINING
                    df_renamed_training = df_renamed_training[[col for col in required_headers_training_lower if col in df_renamed_training.columns]]

                    # Convert to CSV
                    csv_buffer_training = io.StringIO()
                    df_renamed_training.to_csv(csv_buffer_training, index=False)
                    renamed_content_training = csv_buffer_training.getvalue().encode()

                    with st.spinner("🔄 Training model... This may take several minutes."):
                        result = train_model(renamed_content_training, uploaded_file_training.name, API_URL, hyperparameters)

                        if isinstance(result, dict) and "error" in result:
                            st.error(f"❌ {result['error']}")
                        else:
                            st.session_state['training_result'] = result
                            st.success("✅ Training complete!")
                            st.rerun()
            else:
                st.markdown('<div class="warning-box"><strong>⚠️ Cannot train yet</strong></div>', unsafe_allow_html=True)
                if validation_result_training['missing']:
                    st.error("❌ Please ensure all required headers are present and correctly named.")
                if validation_result_training['duplicates']:
                    st.error("❌ Duplicate header names detected. Each column must have a unique name.")
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("💡 Please ensure your file is a valid CSV format.")

    # ---------------------------
    # Display Training Results
    # ---------------------------
    if 'training_result' in st.session_state:
        st.markdown("---")
        st.markdown("### 🎯 Training Results")

        training_result = st.session_state['training_result']

        # Check if result is a valid training result
        if isinstance(training_result, dict) and "error" in training_result:
            st.error(f"❌ Unexpected response from backend: {training_result}")
            if st.button("Clear and Try Again", key="clear_training"):
                del st.session_state['training_result']
                st.rerun()
        else:
            # Display only training metrics
            if isinstance(training_result, dict):
                # Display training data
                if 'data' in training_result and isinstance(training_result['data'], list) and len(training_result['data']) > 0:
                    training_data = training_result['data'][0]
                    
                    st.markdown("#### 🎯 Training Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'accuracy' in training_data:
                            accuracy_value = training_data['accuracy']
                            st.metric("🎯 Accuracy", f"{accuracy_value:.3f}")
                    
                    with col2:
                        if 'roc_auc' in training_data:
                            roc_auc_value = training_data['roc_auc']
                            st.metric("📊 ROC AUC", f"{roc_auc_value:.3f}")
                    
                    with col3:
                        if 'f1_score' in training_data:
                            f1_value = training_data['f1_score']
                            st.metric("⚖️ F1 Score", f"{f1_value:.3f}")
            
            # Download button for training results
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                import json
                json_data = json.dumps(training_result, indent=2).encode('utf-8')
                st.download_button(
                    label="📥 Download Training Results (JSON)",
                    data=json_data,
                    file_name="training_results.json",
                    mime="application/json",
                    use_container_width=True
                )

            with col2:
                if st.button("🔄 Train New Model", use_container_width=True, key="new_training"):
                    # Clear session state
                    if 'training_result' in st.session_state:
                        del st.session_state['training_result']
                    if 'edited_headers' in st.session_state:
                        del st.session_state['edited_headers']
                    st.rerun()
    
    else:
        # Show training data requirements when no file is uploaded
        st.markdown("### 📊 Training Data Requirements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Required Columns:**
            - All inference features
            - `exoplanet_status` (target variable)
            - `confidence_score` (optional)
            """)
        
        with col2:
            st.markdown("""
            **Data Quality:**
            - Minimum 1000 samples
            - Balanced class distribution
            - Clean, validated data
            """)
        
        st.markdown("### 🔧 Training Configuration")
        
        with st.expander("📋 Model Parameters", expanded=False):
            st.markdown("**Coming Soon:** Advanced training configuration options")
            st.info("This section will include hyperparameter tuning, cross-validation settings, and model architecture options.")

elif st.session_state.current_page == 'Models':
    # ---------------------------
    # Models Page
    # ---------------------------
    st.markdown("### 🤖 Model Management")

    # Refresh button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("View and manage available models from the backend.")
    with col2:
        if st.button("🔄 Refresh Models", use_container_width=True):
            if 'models_list' in st.session_state:
                del st.session_state['models_list']
            st.rerun()

    st.markdown("---")

    # Load models button
    if st.button("📥 Load Models", type="primary", use_container_width=True):
        with st.spinner("🔄 Loading models from backend..."):
            models = load_models(API_URL)

            if isinstance(models, dict) and "error" in models:
                st.error(f"❌ {models['error']}")
            else:
                st.session_state['models_list'] = models
                # Extract model count from the response
                model_count = len(models.get('data', {})) if isinstance(models, dict) and 'data' in models else 0
                st.success(f"✅ Loaded {model_count} model(s)")
                st.rerun()

    # Display models if loaded
    if 'models_list' in st.session_state:
        st.markdown("---")
        st.markdown("### 📋 Available Models")

        models_response = st.session_state['models_list']

        # Handle the new JSON format with 'data' key
        if isinstance(models_response, dict) and 'data' in models_response:
            models_data = models_response['data']

            if len(models_data) > 0:
                # Display model count
                st.markdown(f"**Total Models:** {len(models_data)}")

                # Display each model
                for idx, model in models_data.items():
                    with st.expander(f"🔹 Model {int(idx) + 1}: {model.get('name', 'Unknown')}"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Model Information:**")
                            st.write(f"**Name:** {model.get('name', 'N/A')}")
                            st.write(f"**Path:** {model.get('path', 'N/A')}")

                        with col2:
                            st.markdown("**Additional Details:**")
                            if 'metrics' in model:
                                metrics = model['metrics']
                                st.write(f"**Accuracy:** {metrics.get('accuracy', 'N/A')}")
                                st.write(f"**F1 Score:** {metrics.get('f1_score', 'N/A')}")
                                st.write(f"**ROC AUC:** {metrics.get('roc_auc', 'N/A')}")
                            else:
                                st.write("No metrics available")

                        # Display full model data as JSON
                        with st.expander("📄 View Full Model Data (JSON)"):
                            import json
                            st.json(model)
            else:
                st.info("📭 No models found in the backend.")
        else:
            st.warning("⚠️ Unexpected response format from backend.")

elif st.session_state.current_page == 'Dataset':
    # ---------------------------
    # Dataset Page
    # ---------------------------
    st.markdown("### 📊 Dataset Management")

    # Refresh and Load buttons
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("View, manage, and upload datasets stored on the server.")
    with col2:
        if st.button("🔄 Refresh Datasets", use_container_width=True):
            if 'datasets_list' in st.session_state:
                del st.session_state['datasets_list']
            st.rerun()

    st.markdown("---")

    # Load datasets button
    if st.button("📥 Load Datasets", type="primary", use_container_width=True):
        with st.spinner("🔄 Loading datasets from server..."):
            datasets = load_datasets(API_URL)

            if isinstance(datasets, dict) and "error" in datasets:
                st.error(f"❌ {datasets['error']}")
            else:
                st.session_state['datasets_list'] = datasets
                # Extract dataset count from the response
                dataset_count = len(datasets.get('data', {})) if isinstance(datasets, dict) and 'data' in datasets else 0
                st.success(f"✅ Loaded {dataset_count} dataset(s)")
                st.rerun()

    # Display datasets if loaded
    if 'datasets_list' in st.session_state:
        st.markdown("---")
        st.markdown("### 📋 Available Datasets")

        datasets_response = st.session_state['datasets_list']

        # Handle the JSON format with 'data' key
        if isinstance(datasets_response, dict) and 'data' in datasets_response:
            datasets_data = datasets_response['data']

            if len(datasets_data) > 0:
                # Display dataset count
                st.markdown(f"**Total Datasets:** {len(datasets_data)}")

                # Create table data
                table_data = []
                for idx, dataset in datasets_data.items():
                    row = {
                        "#": int(idx) + 1,
                        "Dataset Name": dataset.get('name', 'N/A'),
                        "Path": dataset.get('path', 'N/A'),
                        "Size": dataset.get('size', 'N/A'),
                        "Upload Date": dataset.get('upload_date', 'N/A'),
                    }
                    table_data.append(row)

                # Display as DataFrame table
                df_datasets = pd.DataFrame(table_data)
                st.dataframe(df_datasets, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.markdown("### 📄 Dataset Details")

                # Display each dataset in expandable sections
                for idx, dataset in datasets_data.items():
                    with st.expander(f"🔹 Dataset {int(idx) + 1}: {dataset.get('name', 'Unknown')}"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Dataset Information:**")
                            st.write(f"**Name:** {dataset.get('name', 'N/A')}")
                            st.write(f"**Path:** {dataset.get('path', 'N/A')}")
                            st.write(f"**Size:** {dataset.get('size', 'N/A')}")

                        with col2:
                            st.markdown("**Additional Details:**")
                            st.write(f"**Upload Date:** {dataset.get('upload_date', 'N/A')}")
                            st.write(f"**Rows:** {dataset.get('rows', 'N/A')}")
                            st.write(f"**Columns:** {dataset.get('columns', 'N/A')}")

                        # Display full dataset data as JSON
                        with st.expander("📄 View Full Dataset Data (JSON)"):
                            import json
                            st.json(dataset)
            else:
                st.info("📭 No datasets found on the server.")
        else:
            st.warning("⚠️ Unexpected response format from backend.")

    st.markdown("---")
    st.markdown("### 📤 Upload New Dataset")

    # File uploader
    uploaded_dataset = st.file_uploader(
        "Choose a CSV file to upload to the server",
        type=["csv"],
        help="Upload a CSV file to be stored on the server",
        key="dataset_uploader"
    )

    if uploaded_dataset is not None:
        # Read file content
        dataset_content = uploaded_dataset.read()

        # Preview uploaded file
        st.markdown("### 📊 Dataset Preview")
        try:
            df_dataset = pd.read_csv(io.BytesIO(dataset_content))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📝 Rows", f"{len(df_dataset):,}")
            with col2:
                st.metric("📋 Columns", len(df_dataset.columns))
            with col3:
                st.metric("💾 File Size", f"{len(dataset_content) / 1024:.1f} KB")

            # Show preview
            with st.expander("🔍 View Dataset Sample", expanded=True):
                st.dataframe(df_dataset.head(10), use_container_width=True)

            st.markdown("---")

            # Upload button
            if st.button("📤 Upload to Server", type="primary", use_container_width=True):
                with st.spinner("📤 Uploading dataset to server..."):
                    result = upload_dataset(dataset_content, uploaded_dataset.name, API_URL)

                    if isinstance(result, dict) and "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.success(f"✅ Dataset uploaded successfully!")
                        if isinstance(result, dict):
                            st.json(result)

        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("💡 Please ensure your file is a valid CSV format.")
    else:
        # Show upload instructions when no file is uploaded
        st.markdown("### 📋 Upload Instructions")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **What you can upload:**
            - Training datasets with labels
            - Inference datasets
            - Any CSV file for later use
            """)

        with col2:
            st.markdown("""
            **Benefits:**
            - Store datasets on the server
            - Reuse datasets across sessions
            - Share datasets with team members
            """)

        st.markdown("---")
        st.info("💡 **Tip:** Upload your CSV file using the file uploader above. The file will be saved on the server and can be accessed later for training or inference.")

elif st.session_state.current_page == 'Help':
    # ---------------------------
    # Help Page
    # ---------------------------
    st.markdown("### ❓ Help & Documentation")
    st.markdown("Comprehensive guide to using the Exoplanet Classification System")

    st.markdown("---")

    # Table of Contents
    st.markdown("## 📑 Table of Contents")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - [Machine Learning Models](#machine-learning-models)
        - [Classification Categories](#classification-categories)
        - [Using the System](#using-the-system)
        """)
    with col2:
        st.markdown("""
        - [Data Requirements](#data-requirements)
        - [API Information](#api-information)
        - [Troubleshooting](#troubleshooting)
        """)

    st.markdown("---")

    # Machine Learning Models
    st.markdown("## 🤖 Machine Learning Models")
    st.markdown("Our system uses an ensemble of gradient boosting models for accurate exoplanet classification.")

    with st.expander("🌟 **LightGBM (Light Gradient Boosting Machine)**", expanded=True):
        st.markdown("""
        **Description:**
        LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It's designed for distributed and efficient training.

        **Key Features:**
        - **Faster training speed** and higher efficiency
        - **Lower memory usage**
        - Better accuracy with large datasets
        - Support for parallel and GPU learning

        **Default Hyperparameters:**
        - `n_estimators`: [200, 400] - Number of boosting iterations
        - `learning_rate`: [0.05, 0.1] - Step size shrinkage
        - `num_leaves`: [31, 63] - Maximum tree leaves
        - `max_depth`: [5, 10] - Maximum tree depth
        - `subsample`: [0.8, 1.0] - Training data sampling ratio
        - `colsample_bytree`: [0.8, 1.0] - Feature sampling ratio

        **Best For:** Large datasets with complex patterns
        """)

    with st.expander("🌲 **Gradient Boosting Classifier**"):
        st.markdown("""
        **Description:**
        Gradient Boosting builds an ensemble of weak prediction models (typically decision trees) in a stage-wise fashion.

        **Key Features:**
        - **High predictive accuracy**
        - Handles both numerical and categorical features
        - Robust to outliers
        - Provides feature importance rankings

        **Default Hyperparameters:**
        - `n_estimators`: [100, 200] - Number of boosting stages
        - `learning_rate`: [0.05, 0.1] - Learning rate
        - `max_depth`: [3, 5] - Maximum tree depth
        - `subsample`: [0.8, 1.0] - Training data sampling ratio

        **Best For:** Medium-sized datasets with mixed feature types
        """)

    with st.expander("📊 **Model Ensemble & Stacking**"):
        st.markdown("""
        **Description:**
        Our system combines multiple models using a stacking ensemble approach for optimal performance.

        **How it Works:**
        1. **Base Models** (Level 0): LightGBM and Gradient Boosting independently make predictions
        2. **Meta Model** (Level 1): Combines base model predictions using a meta-learner
        3. **Final Prediction**: Weighted average or voting mechanism produces final classification

        **Benefits:**
        - **Higher accuracy** than individual models
        - **Reduced overfitting** through model diversity
        - **Robust predictions** across different data distributions
        """)

    st.markdown("---")

    # Classification Categories
    st.markdown("## 🏷️ Classification Categories")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="success-box">
            <h4>✅ CONFIRMED</h4>
            <p><strong>Definition:</strong> Verified exoplanet with strong evidence</p>
            <p><strong>Confidence:</strong> >85%</p>
            <p><strong>Characteristics:</strong></p>
            <ul>
                <li>Clear transit signal</li>
                <li>Consistent orbital period</li>
                <li>Ruled out false positives</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>❌ FALSE POSITIVE</h4>
            <p><strong>Definition:</strong> Not an exoplanet</p>
            <p><strong>Confidence:</strong> >80%</p>
            <p><strong>Common Causes:</strong></p>
            <ul>
                <li>Binary star systems</li>
                <li>Stellar variability</li>
                <li>Instrumental artifacts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-box">
            <h4>🔍 CANDIDATE</h4>
            <p><strong>Definition:</strong> Requires additional data</p>
            <p><strong>Confidence:</strong> 50-85%</p>
            <p><strong>Next Steps:</strong></p>
            <ul>
                <li>Additional observations</li>
                <li>Follow-up analysis</li>
                <li>Radial velocity confirmation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Using the System
    st.markdown("## 🚀 Using the System")

    with st.expander("**1️⃣ Inference Workflow**", expanded=True):
        st.markdown("""
        **Step 1: Load Models**
        Click "Load Models" to fetch available models from the backend

        **Step 2: Select Model**
        Choose the model you want to use for classification from the dropdown

        **Step 3: Upload Data**
        Upload a CSV file with exoplanet candidate measurements

        **Step 4: Validate Headers**
        Edit column names if needed to match required format

        **Step 5: Classify**
        Click "Classify Exoplanets" to get predictions

        **Step 6: Download Results**
        Download the classification results as CSV
        """)

    with st.expander("**2️⃣ Training Workflow**"):
        st.markdown("""
        **Step 1: Prepare Training Data**
        Ensure your CSV includes all required features + `exoplanet_status` label

        **Step 2: Upload Training CSV**
        Upload your labeled training dataset

        **Step 3: Configure Hyperparameters**
        Use default hyperparameters or enable Expert Mode for custom JSON

        **Step 4: Start Training**
        Click "Start Training" and wait for the process to complete

        **Step 5: Review Metrics**
        Examine accuracy, ROC AUC, and F1 score

        **Step 6: Download Results**
        Save training results as JSON for future reference
        """)

    st.markdown("---")

    # Data Requirements
    st.markdown("## 📊 Data Requirements")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Inference Features (11)")
        for i, header in enumerate(EXPECTED_HEADERS_INFERENCE, 1):
            st.markdown(f"**{i}. `{header}`**")
            st.caption(HEADER_DESCRIPTIONS.get(header, ""))

    with col2:
        st.markdown("### Training Features (17)")
        st.info("All inference features plus additional parameters and target variable")
        st.markdown("""
        **Additional Required:**
        - `st_pmdec`, `st_pmra` (Proper motion)
        - `dec`, `ra` (Coordinates)
        - `pl_tranmid` (Transit midpoint)
        - `exoplanet_status` (Target label)
        """)

    st.markdown("---")

    # API Information
    st.markdown("## 🔌 API Information")

    with st.expander("**API Endpoints**"):
        st.markdown(f"""
        **Backend URL:** `{API_URL}`

        **Available Endpoints:**
        - `POST /inference/classify-csv` - Classify exoplanet candidates
        - `GET /inference/get-models` - List available models
        - `POST /model/train` - Train new model
        - `POST /dataset/upload` - Upload dataset to server

        **Request Format:**
        Multipart form data with CSV file attachment

        **Response Format:**
        - Inference: CSV file with predictions
        - Training: JSON with metrics and model info
        """)

    st.markdown("---")

    # Troubleshooting
    st.markdown("## 🔧 Troubleshooting")

    with st.expander("**❌ Common Issues & Solutions**"):
        st.markdown("""
        **Problem:** "Missing Required Headers" error
        **Solution:** Use the header editor to rename columns to match expected format

        ---

        **Problem:** "Connection Error" message
        **Solution:** Check that the backend server is running and accessible

        ---

        **Problem:** Model training fails
        **Solution:** Ensure dataset has minimum 1000 samples and includes `exoplanet_status` column

        ---

        **Problem:** Invalid JSON in Expert Mode
        **Solution:** Verify JSON syntax - use JSON validator or reset to defaults

        ---

        **Problem:** Low classification confidence
        **Solution:** Consider:
        - Using more training data
        - Adjusting hyperparameters
        - Checking data quality
        """)

    st.markdown("---")

    # Contact & Support
    st.markdown("## 💬 Contact & Support")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **For Technical Support:**
        - Check the troubleshooting section above
        - Review the API documentation
        - Verify data format requirements
        """)

    with col2:
        st.markdown("""
        **System Information:**
        - **Backend:** FastAPI
        - **Frontend:** Streamlit
        - **Models:** LightGBM + Gradient Boosting
        - **Version:** 1.0.0
        """)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6c757d; padding: 2rem 0;'>
        <p style='margin: 0; font-size: 0.9rem;'>
            <strong>Exoplanet Classification System</strong> | Built with Streamlit + FastAPI
        </p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem;'>
            For research and educational purposes
        </p>
    </div>
    """,
    unsafe_allow_html=True
)