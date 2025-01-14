# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt

# Load datasets
train_path = "path_to_your_training_file.csv"  # Replace with your training dataset path
test_path = "path_to_your_testing_file.csv"    # Replace with your test dataset path

training_data = pd.read_csv(train_path)
testing_data = pd.read_csv(test_path)

# Replace missing values (-999.0) with NaN
training_data.replace(-999.0, np.nan, inplace=True)
testing_data.replace(-999.0, np.nan, inplace=True)

# Separate features and target in training data
X = training_data.drop(columns=["EventId", "Weight", "Label"])
y = training_data["Label"].map({"s": 1, "b": 0})  # Encode target as 1 (signal) and 0 (background)

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
testing_data_imputed = imputer.transform(testing_data.drop(columns=["EventId"]))

# Scale features for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
testing_data_scaled = scaler.transform(testing_data_imputed)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Validate the model
y_pred = rf_model.predict(X_val)
y_pred_prob = rf_model.predict_proba(X_val)[:, 1]

# Evaluation metrics
accuracy = accuracy_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_pred_prob)
classification_rep = classification_report(y_val, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("Classification Report:\n", classification_rep)

# Feature Importance
feature_importances = rf_model.feature_importances_
feature_names = X.columns

# Plot the top 10 features
sorted_idx = np.argsort(feature_importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(10), feature_importances[sorted_idx][:10], align="center")
plt.xticks(range(10), feature_names[sorted_idx][:10], rotation=45)
plt.title("Top 10 Feature Importances")
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.show()

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform RandomizedSearchCV
rf_random = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

rf_random.fit(X_train, y_train)

# Best parameters and evaluation
best_params = rf_random.best_params_
print(f"Best Hyperparameters: {best_params}")
best_rf_model = rf_random.best_estimator_

# Test set predictions
test_predictions = best_rf_model.predict_proba(testing_data_scaled)[:, 1]

# Prepare submission file
submission = pd.DataFrame({
    "EventId": testing_data["EventId"],
    "RankOrder": test_predictions.argsort().argsort() + 1,
    "Class": np.where(test_predictions > 0.5, "s", "b")
})

submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created successfully.")