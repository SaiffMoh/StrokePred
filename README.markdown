# Stroke Prediction and Patient Clustering

## Overview
This project predicts stroke risk and identifies high-risk patient groups using a healthcare dataset. By integrating **supervised learning** (Naïve Bayes, SVM, KNN, Decision Trees) and **unsupervised learning** (K-Means, Hierarchical Clustering), it delivers accurate stroke predictions and actionable insights for healthcare applications. The project prioritizes **recall** to minimize false negatives, crucial for early stroke detection, while clustering reveals natural patient groupings for targeted interventions.

Implemented in Python with `scikit-learn`, `pandas`, `seaborn`, and `matplotlib`, the project showcases expertise in data preprocessing, model optimization, visualization, and analysis. It addresses real-world healthcare needs, such as preventive care and patient risk profiling, making it highly relevant for medical applications.

## Table of Contents
- [Dataset](#dataset)
- [Features](#features)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Dataset
The project uses the **Stroke Prediction Dataset** from Kaggle, available [here](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). It contains approximately 5,000 records with clinical and lifestyle features for stroke risk prediction.

- **Source**: `healthcare-dataset-stroke-data.csv`
- **Target Variable**: `stroke` (0 = No Stroke, 1 = Stroke)
- **Class Distribution**: Highly imbalanced (~5% stroke cases)

## Features
- **Numerical**: `age`, `avg_glucose_level`, `bmi`
- **Categorical**: `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`
- **Binary**: `hypertension`, `heart_disease`
- **ID**: `id` (dropped during modeling)

## Methodology
The project follows a comprehensive machine learning pipeline:

1. **Data Exploration**:
   - Visualized feature distributions, correlations, and stroke prevalence using histograms, count plots, scatter plots, and correlation heatmaps.
   - Identified class imbalance and missing values in `bmi`.

2. **Preprocessing**:
   - **Missing Values**: Imputed `bmi` with medians within age groups (0-18, 19-40, 41-60, 61-100).
   - **Outliers**: Replaced `bmi` and `avg_glucose_level` outliers with age-group medians using the IQR method.
   - **Encoding**: One-hot encoded categorical features (`ever_married`, `work_type`, `Residence_type`, `smoking_status`); label-encoded `gender` (Female=1, Male=0, reflecting higher stroke prevalence in females).
   - **Scaling**: Standardized numerical features (`age`, `avg_glucose_level`, `bmi`) with `StandardScaler`.
   - **Class Imbalance**: Applied SMOTE to the training set to balance stroke and no-stroke classes.

3. **Dimensionality Reduction**:
   - Applied PCA, LDA, and t-SNE for 2D and 3D visualizations to explore data structure and class separability.
   - Visualized patient distributions to assess feature relationships and clustering potential.

4. **Classification**:
   - Trained four classifiers: Naïve Bayes, SVM, KNN, and Decision Trees.
   - Tuned hyperparameters using `GridSearchCV` with 5-fold cross-validation, optimizing for F1-score to balance precision and recall.
   - Evaluated models on accuracy, precision, recall, and F1-score, prioritizing recall to minimize false negatives in medical applications.

5. **Clustering**:
   - Performed K-Means and Hierarchical Clustering with 3 clusters, selected via the Elbow Method.
   - Visualized clusters using PCA scatter plots and a dendrogram for hierarchical clustering.
   - Analyzed stroke prevalence per cluster to identify high-risk groups.

6. **Comparison**:
   - Compared classifier performance using bar charts of accuracy, precision, recall, and F1-score.
   - Aligned clustering results with Naïve Bayes predictions to validate high-risk group identification.

## Results
- **Classifier Performance**:
  - **Naïve Bayes** achieved the highest recall, making it the best model for stroke prediction by minimizing false negatives, critical in medical contexts.
  - Approximate metrics (based on typical runs):
    - **Naïve Bayes**: Accuracy=0.82, Precision=0.25, Recall=0.75, F1=0.38
    - **SVM**: Accuracy=0.85, Precision=0.30, Recall=0.65, F1=0.41
    - **KNN**: Accuracy=0.80, Precision=0.28, Recall=0.60, F1=0.38
    - **Decision Tree**: Accuracy=0.83, Precision=0.27, Recall=0.62, F1=0.37

- **Clustering Insights**:
  - Identified clusters with elevated stroke prevalence, indicating high-risk patient groups.
  - Naïve Bayes predictions showed strong alignment with high-risk clusters, reinforcing the clustering results.

- **Key Insight**: The integration of high-recall classification and clustering provides a robust framework for early stroke detection and patient risk profiling, suitable for healthcare applications like preventive care.

## Project Structure
```
stroke-prediction/
├── healthcare-dataset-stroke-data.csv  # Dataset (not included, download from Kaggle)
├── StrokePred.ipynb                    # Main Jupyter Notebook
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
- **Inspiration**: Motivated by the need for early stroke detection in healthcare, this project aims to contribute to real-world medical applications.