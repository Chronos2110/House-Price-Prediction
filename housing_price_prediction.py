import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# Load dataset
# Replace the relative path with your exact machine path
data = pd.read_csv(r"C:\Python\Practicing python\Projects\Project 2\housing.csv")

# ==========================================
# 1. FEATURE ENGINEERING
# Creating new, smarter attributes from existing columns
# ==========================================
data['total_rooms'] = data['bedrooms'] + data['bathrooms']
data['area_per_room'] = data['area'] / data['total_rooms']

# Preprocessing: Convert text variables (yes/no, furnished) into 1s and 0s
data = pd.get_dummies(data, drop_first=True)

X = data.drop('price', axis=1)
y = data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 2. FEATURE SCALING
# Standardizing the numbers so 'area' doesn't overpower 'bedrooms'
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Only transform the test set to prevent data leakage

# Linear Regression (Trained on SCALED data)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_predictions = lr.predict(X_test_scaled)

print("----- Enhanced Linear Regression -----")
print(f"MAE: {metrics.mean_absolute_error(y_test, lr_predictions):.2f}")
print(f"R2 Score: {metrics.r2_score(y_test, lr_predictions):.4f}\n")

# ==========================================
# 3. HYPERPARAMETER TUNING
# Customizing the Random Forest to prevent overfitting
# ==========================================
# (Note: Tree-based models like RF don't actually need scaled data)
rf = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=5, random_state=42)
rf.fit(X_train, y_train) 
rf_predictions = rf.predict(X_test)

print("----- Enhanced Random Forest -----")
print(f"MAE: {metrics.mean_absolute_error(y_test, rf_predictions):.2f}")
print(f"R2 Score: {metrics.r2_score(y_test, rf_predictions):.4f}\n")

# Plotting the Enhanced Linear Regression Model
plt.figure(figsize=(8, 5))
plt.scatter(y_test, lr_predictions, alpha=0.7, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Enhanced Linear Regression: Actual vs Predicted Price")
plt.show()