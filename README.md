# Spatial Machine Learning For Intersection Safety Analysis 
This is a novel spatial Machine Learning (ML) framework that integrates multiple ML models (e.g., RF, XGBoost and LightGBM) with geographically weighted regression to account for spatial heterogeneity. I have utilized it in intersection crash frequency modeling and achieve good performances.
Inspired by the previous Geographical Random Forest (GRF) studeis: 
1) Georganos S, Grippa T, Niang Gadiaga A, et al. Geographical random forests: a spatial extension of the random forest algorithm to address spatial heterogeneity in remote sensing and population modelling[J]. Geocarto International, 2021, 36(2): 121-136.
2) Kai Sun, Ryan Zhenqi Zhou, Jiyeon Kim, and Yingjie Hu. 2024. PyGRF: An improved Python Geographical Random Forest model and case studies in public health and natural disasters. Transactions in GIS.

We furthur integrate the XGBoost and LightGBM into the geographically weighted modeling, i.e., replace the local models with XGBoost and LightGBM. Our experiments show that spatial XGBoost/LightGBM shows comparible prediction results but much faster training speed.

![Model training structure](https://github.com/user-attachments/assets/288bd1fb-e946-4da2-9c41-19f9528ac54c)
