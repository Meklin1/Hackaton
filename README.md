# Exoplanet Classification API 🚀

An AI/ML-powered FastAPI server for classifying exoplanets using NASA's TESS (Transiting Exoplanet Survey Satellite) data. This project implements a machine learning pipeline that analyzes astronomical data to identify potential exoplanets, planetary candidates, or false positives.

## 🌟 Features

- **Machine Learning Pipeline**: Pre-trained stacking ensemble model for exoplanet classification
- **RESTful API**: FastAPI-based web service with automatic documentation
- **CSV Upload Support**: Upload astronomical data for batch classification
- **Confidence Scoring**: Provides prediction confidence levels for each classification
- **Docker Support**: Containerized deployment with Docker and Docker Compose
- **Health Monitoring**: Built-in health checks and model status endpoints
- **CORS Enabled**: Cross-origin resource sharing for web integration

## 🏗️ Architecture

The project uses a modular architecture with the following components:

- **FastAPI Server** (`src/main.py`): Main application with CORS middleware
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

3. **Start the server**
   ```bash
   ./start_server.sh
   ```
   Or manually:
   ```bash
   source .venv/bin/activate
   uvicorn src.main:app --reload --port 8081
   ```

4. **Access the API**
   - API Documentation: http://localhost:8081/docs
   - Health Check: http://localhost:8081/health
   - Interactive API: http://localhost:8081/redoc

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

## 📡 API Endpoints

### Health & Status
- `GET /` - Basic health check
- `GET /health` - Detailed health status
- `GET /inference/model-status` - Model loading status

### Classification
- `POST /inference/classify` - Classify exoplanets (returns JSON)
- `POST /inference/classify-csv` - Classify exoplanets (returns CSV download)

### API Documentation
- `GET /docs` - Interactive Swagger UI
- `GET /redoc` - ReDoc documentation

## 🔬 Model Details

The classification model is a **stacking ensemble** trained on NASA's TESS data:

- **Model Type**: Pre-trained scikit-learn pipeline with LightGBM
- **Model File**: `models/toi_stacking_pipeline.joblib`
- **Training Data**: TESS Object of Interest (TOI) features
- **Classification**: Exoplanet vs. False Positive
- **Features**: Astronomical parameters (orbital period, transit duration, planetary radius, etc.)

## 📊 Usage Examples

### Upload CSV for Classification

```bash
curl -X POST "http://localhost:8081/inference/classify" \
     -H "Content-Type: multipart/form-data" \
     -F "csv_file=@your_data.csv"
```

### Check Model Status

```bash
curl -X GET "http://localhost:8081/inference/model-status"
```

## 🛠️ Development

### Project Structure
```
exoplanet/
├── src/
│   ├── api/
│   │   ├── models/          # Request/Response models
│   │   └── routers/         # API route handlers
│   ├── models/              # ML model classes
│   └── main.py              # FastAPI application
├── models/                  # Pre-trained model files
├── data/                    # Training datasets
├── dockerfiles/             # Docker configurations
├── compose.yml              # Docker Compose setup
└── pyproject.toml           # Project dependencies
```

### Dependencies

- **FastAPI**: Web framework
- **scikit-learn**: Machine learning
- **LightGBM**: Gradient boosting
- **pandas**: Data manipulation
- **joblib**: Model serialization
- **uvicorn**: ASGI server

## 🐳 Docker Configuration

The project includes optimized Docker configurations:

- **Base Image**: Python 3.12-slim
- **Package Manager**: UV for fast dependency resolution
- **System Dependencies**: LightGBM runtime libraries
- **Port**: 8000 (configurable)
- **Volume Mounting**: Source code for development

## 📈 Performance

- **Model Loading**: Automatic on startup
- **Prediction Speed**: Optimized for batch processing
- **Memory Usage**: Efficient with large datasets
- **Concurrent Requests**: ASGI-based async handling

## 🔧 Configuration

### Environment Variables
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)

### Model Configuration
- Model path: `models/toi_stacking_pipeline.joblib`
- Auto-loading: Enabled on startup
- Error handling: Graceful fallback

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
- FastAPI and scikit-learn communities for excellent documentation

---

**Status**: ✅ Production Ready | **Version**: 1.0.0 | **Python**: 3.12+






