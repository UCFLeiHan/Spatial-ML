# Spatial Machine Learning for Intersection Safety Analysis

This repository presents a novel spatial Machine Learning (ML) framework that integrates multiple ML models—such as Random Forest, XGBoost, and LightGBM—with geographically weighted regression to account for spatial heterogeneity. The framework has been successfully applied to intersection crash frequency modeling, achieving excellent performance.

Inspired by previous Geographical Random Forest (GRF) studies:
1. Georganos, S., Grippa, T., Niang Gadiaga, A., et al. (2021). *Geographical random forests: a spatial extension of the random forest algorithm to address spatial heterogeneity in remote sensing and population modelling*. Geocarto International, 36(2), 121-136.
2. Sun, K., Zhou, R. Z., Kim, J., & Hu, Y. (2024). *PyGRF: An improved Python Geographical Random Forest model and case studies in public health and natural disasters*. Transactions in GIS.

We further extend geographically weighted modeling by integrating boosted algorithms (XGBoost and LightGBM) as local models. Our experiments demonstrate that spatial XGBoost/LightGBM achieves comparable prediction results with significantly faster training speeds.

![Model training structure](https://github.com/user-attachments/assets/288bd1fb-e946-4da2-9c41-19f9528ac54c)

## Repository Structure

- **SpatialML_codes/**
  - `PyGML.py`: Contains the spatial ML functions.
  - `Intersection_crash_modeling.ipynb`: An example notebook demonstrating how to use the framework for intersection crash modeling.

## Quick Start

### 1. Training the Models
Use the provided notebook to train the models on your dataset. For example:

```python
model = PyGML.PyGBoostBuilder(
    model_type="lightgbm", 
    n_estimators=50, 
    band_width=105, 
    learning_rate=0.2, 
    random_state=42
) 
model.fit(X_train, y_train, X_Coord_train)
```

### 2. Testing
Evaluate model performance on a test dataset:

```python
y_pred, predict_global, predict_local = model.predict(X_test, X_Coord_test, local_weight=0.11)
```

### 3. Model Interpretation
Utilize SHAP for model interpretation to understand feature contributions.

#### Global Level Interpretation
```python
explainer = shap.TreeExplainer(model.global_model)
shap_values = explainer.shap_values(X) 
shap.summary_plot(shap_values, X)
```

#### Local Level Interpretation
```python
importance = model.get_local_feature_importance()
columns_to_normalize = importance.columns.difference(['model_index'])
importance[columns_to_normalize] = (
    importance[columns_to_normalize]
    .div(importance[columns_to_normalize].sum(axis=1), axis=0)
)
```

If you have any questions or encounter any issues, please feel free to contact the authors.

Happy modeling!
