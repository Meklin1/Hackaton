# Exoplanet Classification App 🚀

An AI/ML-powered Streamlit web application for classifying exoplanets using NASA's TESS (Transiting Exoplanet Survey Satellite) data. This project provides an intuitive interface for uploading astronomical data and getting real-time exoplanet classifications.

## 🌟 Features

- **Interactive Web Interface**: User-friendly Streamlit dashboard for data upload and analysis
- **Machine Learning Pipeline**: Pre-trained stacking ensemble model for exoplanet classification
- **CSV Upload Support**: Upload astronomical data for batch classification
- **Real-time Results**: Instant visualization of classification results with confidence scores
- **Model Management**: Train new models and manage existing ones
- **Dataset Management**: Upload and manage datasets on the server
- **Header Validation**: Smart column header validation and editing
- **Docker Support**: Containerized deployment with Docker and Docker Compose

## 🏗️ Architecture

The project uses a client-server architecture with the following components:

- **Streamlit Frontend** (`app.py`): Interactive web interface for users
- **FastAPI Backend** (`src/main.py`): REST API server for ML operations
- **Inference Router** (`src/api/routers/inference.py`): Handles classification endpoints
- **TOI Model** (`src/models/toi.py`): Manages the pre-trained machine learning model
- **Docker Configuration**: Multi-stage builds with optimized Python 3.12 image

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Docker (optional)
- UV package manager (recommended)

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd exoplanet
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Start the Streamlit app**
   ```bash
   streamlit run app.py --server.port 8501
   ```

4. **Access the application**
   - Streamlit App: http://localhost:8501
   - Backend API: http://localhost:8000 (if running separately)

### Docker Deployment

1. **Using Docker Compose (Recommended)**
   ```bash
   docker-compose up --build
   ```

2. **Using Docker directly**
   ```bash
   docker build -f dockerfiles/server.dockerfile -t exoplanet-api .
   docker run -p 8000:8000 exoplanet-api
   ```

## 🖥️ Application Features

### Main Pages
- **🔮 Inference**: Upload CSV files and classify exoplanet candidates
- **🎓 Training**: Train new machine learning models with custom datasets
- **🤖 Models**: View and manage available models
- **📊 Dataset**: Upload and manage datasets on the server
- **❓ Help**: Comprehensive documentation and troubleshooting guide

### Key Capabilities
- **Smart Header Validation**: Automatically detects and allows editing of CSV column headers
- **Real-time Classification**: Instant results with confidence scores and visualizations
- **Model Management**: Train, load, and switch between different models
- **Data Visualization**: Interactive tables with color-coded classification results
- **Export Results**: Download classification results as CSV files

## 🔬 Model Details

The classification model is a **stacking ensemble** trained on NASA's TESS data:

- **Model Type**: Pre-trained scikit-learn pipeline with LightGBM
- **Model File**: `models/toi_stacking_pipeline.joblib`
- **Training Data**: TESS Object of Interest (TOI) features
- **Classification**: Exoplanet vs. False Positive
- **Features**: Astronomical parameters (orbital period, transit duration, planetary radius, etc.)

## 📊 Usage Examples

### Using the Streamlit Interface

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Navigate to Inference page**
   - Click on "🔮 Inference" in the sidebar
   - Load available models from the backend
   - Upload your CSV file with exoplanet data

3. **Validate and classify**
   - Edit column headers if needed
   - Click "🚀 Classify Exoplanets"
   - View results with confidence scores
   - Download results as CSV

### Direct API Usage (Backend)

```bash
# Upload CSV for classification
curl -X POST "http://localhost:8000/inference/classify-csv" \
     -H "Content-Type: multipart/form-data" \
     -F "csv_file=@your_data.csv"

# Check model status
curl -X GET "http://localhost:8000/inference/model-status"
```

## 🛠️ Development

### Project Structure
```
exoplanet/
├── app.py                   # Streamlit frontend application
├── src/
│   ├── api/
│   │   ├── models/          # Request/Response models
│   │   └── routers/         # API route handlers
│   ├── models/              # ML model classes
│   └── main.py              # FastAPI backend application
├── models/                  # Pre-trained model files
├── data/                    # Training datasets
├── dockerfiles/             # Docker configurations
├── compose.yml              # Docker Compose setup
└── pyproject.toml           # Project dependencies
```

### Dependencies

- **Streamlit**: Frontend web interface
- **FastAPI**: Backend web framework
- **scikit-learn**: Machine learning
- **LightGBM**: Gradient boosting
- **pandas**: Data manipulation
- **joblib**: Model serialization
- **requests**: HTTP client for API calls

## 🐳 Docker Configuration

The project includes optimized Docker configurations:

- **Base Image**: Python 3.12-slim
- **Package Manager**: UV for fast dependency resolution
- **System Dependencies**: LightGBM runtime libraries
- **Ports**: 8501 (Streamlit), 8000 (FastAPI backend)
- **Volume Mounting**: Source code for development

## 📈 Performance

- **Interactive Interface**: Real-time data visualization and editing
- **Model Loading**: Automatic on backend startup
- **Prediction Speed**: Optimized for batch processing
- **Memory Usage**: Efficient with large datasets
- **User Experience**: Intuitive drag-and-drop file uploads

## 🔧 Configuration

### Environment Variables
- `PORT`: Backend server port (default: 8000)
- `HOST`: Backend server host (default: 0.0.0.0)
- `API_URL`: Backend API URL (configured in app.py)

### Model Configuration
- Model path: `models/toi_stacking_pipeline.joblib`
- Auto-loading: Enabled on backend startup
- Error handling: Graceful fallback with user-friendly messages

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of a NASA exoplanet classification challenge. Please refer to the original challenge guidelines for usage terms.

## 🙏 Acknowledgments

- NASA for providing the TESS exoplanet datasets
- The astronomical community for open-source tools and libraries
- Streamlit, FastAPI, and scikit-learn communities for excellent documentation

---

**Status**: ✅ Production Ready | **Version**: 1.0.0 | **Python**: 3.12+ | **Frontend**: Streamlit | **Backend**: FastAPI






