# Mineral Exploration with Machine Learning

This project uses machine learning to predict mineral deposit likelihood using open-source geospatial data. It serves as a demonstration of a complete geospatial ML pipeline, from data ingestion and feature engineering to model training and visualization.

## Features

- **Data Ingestion**: Programmatically downloads and processes mineral occurrence data from the USGS, along with other geospatial datasets (geology, gravity, magnetics).
- **Feature Engineering**: Creates a comprehensive feature set for modeling by integrating various data sources onto a single grid.
- **Machine Learning Models**: Implements several models to predict mineralization, including Random Forest and Bayesian Logistic Regression.
- **Interactive Web App**: A Streamlit application (`app.py`) for visualizing model predictions and uncertainty.
- **Reproducibility**: The entire workflow is captured in a series of Jupyter notebooks and Python scripts.

## Web App

The project includes a Streamlit web application for interactive visualization of the model outputs.

To run the app:

```bash
streamlit run app.py
```

This will start a local web server and open the application in your browser. You can use the app to explore the predicted ore deposit likelihood and associated uncertainty.

## Quickstart

1. **Clone the repository:**

   ```bash
   git clone https://github.com/acfdavis/ore-exploration-ml.git
   cd ore-exploration-ml
   ```

2. **Set up the environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Run the notebooks:**
   Execute the Jupyter notebooks in the `notebooks/` directory in numerical order to download data, engineer features, and train the models.

4. **Run the web app:**

   ```bash
   streamlit run app.py
   ```

## Repo Layout

```text
ore-exploration-ml/
├── app.py                # Streamlit web application
├── data/
│   ├── raw/              # Raw downloaded data
│   └── processed/        # Processed data and features
├── figures/              # Saved figures and maps
├── models/               # Trained model files
├── notebooks/            # Jupyter notebooks for the analysis
├── src/                  # Python source code
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
├── LICENSE
└── README.md
```

## Data Sources

- **USGS Mineral Resource Data System (MRDS)**: Primary source for mineral occurrences.
- **State Geologic Map Compilation (SGMC)**: For bedrock geology features.
- **Gravity and Magnetic Data**: From various public sources.

## Example Outputs

The analysis generates several outputs, including:

- **Probability Maps**: Heatmaps showing the likelihood of mineral deposits.
- **Uncertainty Maps**: Visualizations of the model's uncertainty in its predictions.
- **Feature Importance Plots**: Insights into which features are most predictive.

These outputs are saved in the `figures/` directory and can be explored interactively in the web app.


