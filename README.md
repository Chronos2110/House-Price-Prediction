# House Price Prediction 

A Machine Learning project that predicts housing prices based on various features such as area, number of bedrooms, bathrooms, and other property attributes.

---

## Overview

This project implements a regression-based approach to estimate house prices using real-world data. It demonstrates the complete ML workflow including data preprocessing, visualization, model training, evaluation, and comparison of different algorithms.

---

## Features

- Data preprocessing using Pandas  
- Handling categorical variables with encoding  
- Data visualization using Matplotlib  
- Model training using:
  - Linear Regression  
  - Random Forest Regressor  
- Model evaluation using:
  - Mean Absolute Error (MAE)  
  - R² Score  
- Model comparison and performance analysis  

---

## Key Insights

- Observed a **positive correlation** between area and price  
- Linear Regression outperformed Random Forest on this dataset  
- Demonstrated that **simpler models can outperform complex ones** when data relationships are linear  

---

## Tech Stack

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  

---

## Project Structure
house-price-prediction/
│── house_price_prediction.ipynb
│── house_price_prediction.py
│── housing.csv
│── README.md


---

## How to Run

1. Clone the repository:
```bash
1. git clone https://github.com/Chronos2110/Housing-Price-Prediction.git
2. cd Housing-Price-Prediction
3. pip install pandas numpy matplotlib scikit-learn
4. python house_price_prediction.py
