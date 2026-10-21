import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC

# ==========================================
# 1. Use cross_val_score to test accuracy
# ==========================================
# Load sample dataset
data = load_iris()
X, y = data.data, data.target

# Initialize model
rf_model = RandomForestClassifier(random_state=42)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')

print("--- 1. Cross-Validation Results ---")
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Accuracy: {cv_scores.mean():.4f}\n")


# ==========================================
# 2. Implement GridSearchCV for SVM
# ==========================================
# Define SVM model and hyperparameter grid
svm_model = SVC(random_state=42)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

print("--- 2. GridSearchCV Results ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}\n")


# ==========================================
# 3. Save trained model using joblib
# ==========================================
# Extract the best trained model
best_svm_model = grid_search.best_estimator_

# Save the model to a file
joblib.dump(best_svm_model, 'svm_model.pkl')
print("--- 3. Model Saving ---")
print("Model successfully saved as 'svm_model.pkl'\n")


# ==========================================
# 4. Load model and predict on new data
# ==========================================
# Load the saved model from disk
loaded_model = joblib.load('svm_model.pkl')

# New input sample data for prediction
new_data = [
    [5.1, 3.5, 1.4, 0.2],  # Sample 1
    [6.7, 3.0, 5.2, 2.3]   # Sample 2
]

# Make predictions
predictions = loaded_model.predict(new_data)

print("--- 4. Predictions on New Data ---")
print(f"Predictions: {predictions}")
