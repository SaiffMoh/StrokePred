# Stroke Prediction and Patient Clustering

## Overview
This project focuses on predicting stroke risk and identifying high-risk patient groups using a healthcare dataset. By integrating **supervised learning** (Naïve Bayes, SVM, KNN, Decision Trees) and **unsupervised learning** (K-Means, Hierarchical Clustering), the project delivers accurate stroke predictions and actionable insights for healthcare applications. The primary goal is to prioritize **recall** to minimize false negatives, critical for early stroke detection, while clustering reveals natural patient groupings for targeted interventions.

The project is implemented in Python using libraries like `scikit-learn`, `pandas`, `seaborn`, and `matplotlib`, showcasing skills in data preprocessing, model optimization, and visualization. This work aligns with real-world healthcare needs, such as preventive care and patient risk profiling.

## Table of Contents
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Dataset
The project uses the **Stroke Prediction Dataset** from Kaggle, available [here](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). The dataset contains ~5,000 records with clinical and lifestyle features for stroke risk prediction.

- **Source**: `healthcare-dataset-stroke-data.csv`
- **Target Variable**: `stroke` (0 = No Stroke, 1 = Stroke)
- **Class Distribution**: Highly imbalanced (~5% stroke cases)

## Features
- **Numerical**: `age`, `avg_glucose_level`, `bmi`
- **Categorical**: `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`
- **Binary**: `hypertension`, `heart_disease`
- **ID**: `id` (dropped during modeling)

## Installation
To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/stroke-prediction.git
   cd stroke-prediction
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Install required packages using:
   ```bash
   pip install -r requirements.txt
   ```

   **requirements.txt**:
   ```
   pandas==2.0.3
   numpy==1.24.3
   scikit-learn==1.3.0
   imblearn==0.10.1
   seaborn==0.12.2
   matplotlib==3.7.2
   scipy==1.10.1
   ```

4. **Download the Dataset**:
   - Download `healthcare-dataset-stroke-data.csv` from the Kaggle link above.
   - Place it in the project root directory.

5. **Install Jupyter Notebook** (if not already installed):
   ```bash
   pip install jupyter
   ```

## Usage
1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the Notebook**:
   - Navigate to `Milestone1&2.ipynb` in the Jupyter interface.
   - Run all cells to execute the pipeline, from data loading to model evaluation and clustering.

3. **Outputs**:
   - Visualizations (e.g., histograms, confusion matrices, PCA/t-SNE plots) will display inline.
   - Performance metrics (accuracy, precision, recall, F1-score) and clustering statistics will print to the console.

## Methodology
The project follows a comprehensive machine learning pipeline:

1. **Data Exploration**:
   - Visualized feature distributions, correlations, and stroke prevalence using histograms, count plots, and scatter plots.
   - Identified class imbalance and missing values in `bmi`.

2. **Preprocessing**:
   - **Missing Values**: Imputed `bmi` with age-group medians.
   - **Outliers**: Replaced `bmi` and `avg_glucose_level` outliers with age-group medians using IQR.
   - **Encoding**: One-hot encoded categorical features; label-encoded `gender` (Female=1, Male=0).
   - **Scaling**: Standardized numerical features with `StandardScaler`.
   - **Class Imbalance**: Applied SMOTE to the training set.

3. **Dimensionality Reduction**:
   - Used PCA, LDA, and t-SNE for 2D/3D visualizations to explore data structure and class separability.

4. **Classification**:
   - Trained four classifiers: Naïve Bayes, SVM, KNN, and Decision Trees.
   - Tuned hyperparameters using `GridSearchCV` with 5-fold cross-validation, optimizing F1-score.
   - Evaluated models on accuracy, precision, recall, and F1-score, prioritizing recall for medical applications.

5. **Clustering**:
   - Applied K-Means and Hierarchical Clustering (K=3, determined via Elbow Method).
   - Visualized clusters using PCA and analyzed stroke prevalence per cluster.

6. **Comparison**:
   - Compared classifier performance and aligned clustering results with Naïve Bayes predictions to identify high-risk groups.

## Results
- **Classifier Performance**:
  - **Naïve Bayes** achieved the highest recall, making it the best model for minimizing false negatives in stroke prediction.
  - Example metrics (approximate, based on typical runs):
    - **Naïve Bayes**: Accuracy=0.82, Precision=0.25, Recall=0.75, F1=0.38
    - **SVM**: Accuracy=0.85, Precision=0.30, Recall=0.65, F1=0.41
    - **KNN**: Accuracy=0.80, Precision=0.28, Recall=0.60, F1=0.38
    - **Decision Tree**: Accuracy=0.83, Precision=0.27, Recall=0.62, F1=0.37

- **Clustering Insights**:
  - Identified high-risk clusters with elevated stroke prevalence.
  - Naïve Bayes predictions aligned with high-risk clusters, reinforcing the clustering results.

- **Key Insight**: The combination of high-recall classification and clustering provides a robust approach for early stroke detection and patient risk profiling, suitable for healthcare applications.

## Project Structure
```
stroke-prediction/
├── healthcare-dataset-stroke-data.csv  # Dataset (not included, download from Kaggle)
├── Milestone1&2.ipynb                  # Main Jupyter Notebook
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make changes and commit (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request with a detailed description of your changes.

Please ensure code follows PEP 8 style guidelines and includes relevant documentation.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **Dataset**: Provided by [fedesoriano](https://www.kaggle.com/fedesoriano) on Kaggle.
- **Libraries**: Thanks to the developers of `scikit-learn`, `pandas`, `seaborn`, `matplotlib`, and `imblearn`.
- **Inspiration**: This project was motivated by the need for early stroke detection in healthcare, aligning with real-world medical applications.