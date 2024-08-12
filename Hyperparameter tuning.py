import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("emails.csv")

# Data info and head
df.info()
df.head()

# Define feature matrix X and target vector y
X = df.iloc[:, :-1]  # Assuming the last column is 'Prediction'
y = df['Prediction']

# Preprocess the dataset
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that combines the preprocessor with the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

# Define the hyperparameters for Grid Search
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': [1, 0.1, 0.01, 0.001],
    'classifier__kernel': ['linear', 'rbf']
}

# Implement Grid Search
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters and model from Grid Search
best_grid_params = grid_search.best_params_
best_grid_model = grid_search.best_estimator_

# Evaluate the Grid Search model
y_pred_grid = best_grid_model.predict(X_test)
print("Grid Search - Best Parameters:", best_grid_params)
print("Grid Search - Classification Report:\n", classification_report(y_test, y_pred_grid))
print("Grid Search - Accuracy:", accuracy_score(y_test, y_pred_grid))

# Define the hyperparameters for Random Search
param_dist = {
    'classifier__C': np.logspace(-3, 3, 10),
    'classifier__gamma': np.logspace(-3, 3, 10),
    'classifier__kernel': ['linear', 'rbf']
}

# Implement Random Search
random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train, y_train)

# Get the best parameters and model from Random Search
best_random_params = random_search.best_params_
best_random_model = random_search.best_estimator_

# Evaluate the Random Search model
y_pred_random = best_random_model.predict(X_test)
print("Random Search - Best Parameters:", best_random_params)
print("Random Search - Classification Report:\n", classification_report(y_test, y_pred_random))
print("Random Search - Accuracy:", accuracy_score(y_test, y_pred_random))
