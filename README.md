# Objectives
Your challenge is to create an artificial intelligence/machine learning model that is trained on one or more of NASA’s open-source exoplanet datasets, and not only analyzes data to identify new exoplanets, but includes a web interface to facilitate user interaction. A number of exoplanet datasets from NASA’s Kepler, K2, and TESS missions are available (see Resources tab). Feel free to utilize any open-source programming language, machine learning libraries, or software solutions that you think would fit into this project well. Think about the different ways that each data variable (e.g., orbital period, transit duration, planetary radius, etc.) might impact the final decision to classify the data point as a confirmed exoplanet, planetary candidate, or false positive. Processing, removing, or incorporating specific data in different ways could mean the difference between higher-accuracy and lower-accuracy models. Think about how scientists and researchers may interact with the project you create. Will you allow users to upload new data or manually enter data via the user interface? Will you utilize the data users provide to update your model? The choices are endless!

## Potential Considerations
You may (but are not required to) consider the following:

Your project could be aimed at researchers wanting to classify new data or novices in the field who want to interact with exoplanet data and do not know where to start.
Your interface could enable your tool to ingest new data and train the models as it does so.
Your interface could show statistics about the accuracy of the current model.
Your model could allow hyperparameter tweaking from the interface.
For data and resources related to this challenge, refer to the Resources tab at the top of the page.


# Implementation Status ✅

## FastAPI Server Features


## API Endpoints

### Inference
- `POST /inference/classify` - Get exoplanet predictions based on CSV

### Data Management
- `POST /data/upload-csv` - Upload CSV files
- `GET /data/models` - List available models
- `GET /data/models/{model_name}` - Get model details
- `POST /data/select-model` - Select a model
- `POST /data/stats` - Get data statistics
- `POST /data/preprocess` - Preprocess data





