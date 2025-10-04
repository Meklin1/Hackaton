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
</style>
""", unsafe_allow_html=True)

# FastAPI backend URL
API_URL = st.sidebar.text_input("🔗 FastAPI Backend URL", value="http://16.16.128.44:8000")

# Expected headers (fetch from backend or hardcode)
EXPECTED_HEADERS = [
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

# Header descriptions for tooltips
HEADER_DESCRIPTIONS = {
    "pl_rade": "Planet Radius (Earth radii)",
    "pl_trandep": "Transit Depth (%)",
    "pl_orbper": "Orbital Period (days)",
    "pl_trandurh": "Transit Duration (hours)",
    "pl_insol": "Insolation Flux (Earth flux)",
    "pl_eqt": "Equilibrium Temperature (K)",
    "st_rad": "Stellar Radius (Solar radii)",
    "st_logg": "Stellar Surface Gravity (log10(cm/s²))",
    "st_teff": "Stellar Effective Temperature (K)",
    "st_tmag": "TESS Magnitude",
    "st_dist": "Distance to Star (parsecs)"
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

def render_editable_headers(df: pd.DataFrame):
    """Render editable header interface with color coding"""
    st.markdown("### ✏️ Edit Column Headers")
    st.markdown("**Instructions:** Edit header names to match required format. Valid headers appear in green, invalid in red, extra in yellow.")

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
            is_valid = current_value in EXPECTED_HEADERS
            is_required = current_value in EXPECTED_HEADERS

            # Color code the label
            if is_valid:
                label_html = f'<div class="header-valid">✓ {current_value}</div>'
                status = 'valid'
            elif current_value in edited_headers:
                label_html = f'<div class="header-invalid">✗ {current_value} (duplicate)</div>'
                status = 'duplicate'
            else:
                # Check if it could be extra
                if original_header not in EXPECTED_HEADERS and current_value not in EXPECTED_HEADERS:
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
    missing_required = [h for h in EXPECTED_HEADERS if h not in edited_headers]
    extra_headers = [h for h in edited_headers if h not in EXPECTED_HEADERS]
    duplicate_headers = [h for h in edited_headers if edited_headers.count(h) > 1]

    st.markdown("---")

    # Display status summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        valid_count = sum(1 for h in edited_headers if h in EXPECTED_HEADERS)
        st.metric("✅ Valid Headers", f"{valid_count}/{len(EXPECTED_HEADERS)}")
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

# Show expected headers
with st.sidebar.expander("📋 Required CSV Headers", expanded=False):
    st.markdown("**Your CSV must contain these columns:**")
    for i, header in enumerate(EXPECTED_HEADERS, 1):
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
# File Upload Section
# ---------------------------
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
        validation_result = render_editable_headers(df_original)

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