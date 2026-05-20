# House Price Prediction

A machine learning project that predicts housing prices based on various features such as area, number of bedrooms, bathrooms, and other property attributes.

## Overview

This project implements a regression-based approach to estimate house prices using real-world data. It demonstrates a comprehensive machine learning workflow, including data preprocessing, feature engineering, feature scaling, model training, and the evaluation of different algorithms.

## Features

- **Data Preprocessing:** Handled categorical variables using one-hot encoding via Pandas.
- **Feature Engineering:** Created derived attributes, such as total room counts and area per room, to improve model accuracy.
- **Feature Scaling:** Applied standardization to numerical inputs to optimize the performance of distance-sensitive algorithms.
- **Data Visualization:** Generated scatter plots with ideal fit baselines using Matplotlib to visually evaluate predictions.
- **Model Training:** - Linear Regression
  - Random Forest Regressor (with hyperparameter tuning)
- **Model Evaluation:** Utilized Mean Absolute Error (MAE) and R² Score for performance analysis.

## Key Insights

- The dataset exhibits a strong positive correlation between property area and final price.
- The enhanced Linear Regression model outperformed the tuned Random Forest Regressor on this specific dataset.
- The results demonstrate that simpler, properly scaled linear models can often outperform more complex, tree-based models when the underlying data relationships are highly linear.

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Project Structure

```text
house-price-prediction/
├── house_price_prediction.ipynb
├── house_price_prediction.py
├── housing.csv
└── README.md
```

## How to Run

1. Clone the repository and navigate into the directory:
```bash
git clone [https://github.com/Chronos2110/Housing-Price-Prediction.git](https://github.com/Chronos2110/Housing-Price-Prediction.git)
cd Housing-Price-Prediction
```
2. Install the required dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn
```
3. Execute the python script:
```bash
python house_price_prediction.py
```
