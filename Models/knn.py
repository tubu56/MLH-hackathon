
import pandas as pd
import joblib

from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# Load dataset
df = pd.read_csv(r'Dataset/disaster_cleaned_dataset.csv')


# Remove columns that should not be used as input features
drop_cols = [
    'Unnamed: 0',
    'event_id',
    'date',
    'event_within_7d',
    'event_within_30d',
    'event_within_90d',
    'timeframe_bucket',
    'post_severity',
    'post_impact_score'
]

df = df.drop(columns=drop_cols, errors='ignore')


# Remove rows without a target value
df = df.dropna(subset=['disaster_type'])


# Separate features and target
x = df.drop(columns=['disaster_type'])
y = df['disaster_type'].astype(str)


# Convert boolean-like columns to actual boolean values
for col in x.columns:
    if x[col].dtype == 'object':
        values = set(
            x[col].dropna().astype(str).str.strip().str.lower().unique()
        )

        if values and values.issubset({'true', 'false'}):
            x[col] = x[col].map(
                lambda value: (
                    str(value).strip().lower() == 'true'
                    if pd.notna(value)
                    else None
                )
            )

        elif values and values.issubset({'yes', 'no'}):
            x[col] = x[col].map(
                lambda value: (
                    str(value).strip().lower() == 'yes'
                    if pd.notna(value)
                    else None
                )
            )


# Define ordinal columns manually if present in your dataset.
# Example: 'severity_level': ['Low', 'Medium', 'High', 'Extreme']
ordinal_orders = {
    # 'severity_level': ['Low', 'Medium', 'High', 'Extreme']
}

# Keep only ordinal columns that exist in the dataset
ordinal_orders = {
    col: order
    for col, order in ordinal_orders.items()
    if col in x.columns
}

ordinal_cols = list(ordinal_orders.keys())


# Detect boolean columns
boolean_cols = x.select_dtypes(
    include=['bool']
).columns.tolist()


# Detect numerical columns
numerical_cols = x.select_dtypes(
    include=['number']
).columns.tolist()


# Detect nominal categorical columns
nominal_cols = x.select_dtypes(
    include=['object', 'category']
).columns.tolist()


# Remove boolean and ordinal columns from other groups
numerical_cols = [
    col for col in numerical_cols
    if col not in boolean_cols and col not in ordinal_cols
]

nominal_cols = [
    col for col in nominal_cols
    if col not in ordinal_cols
]


# Display detected column types
print("Numerical columns:", numerical_cols)
print("Ordinal columns:", ordinal_cols)
print("Nominal columns:", nominal_cols)
print("Boolean columns:", boolean_cols)


# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Numerical preprocessing
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


# Ordinal preprocessing
ordinal_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(
        categories=[
            ordinal_orders[col] for col in ordinal_cols
        ],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )),
    ('scaler', StandardScaler())
])


# Nominal preprocessing
nominal_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    ))
])


# Boolean preprocessing
boolean_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(
        categories=[[False, True]],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ))
])


# Combine all preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', numerical_pipeline, numerical_cols),
        ('ordinal', ordinal_pipeline, ordinal_cols),
        ('nominal', nominal_pipeline, nominal_cols),
        ('boolean', boolean_pipeline, boolean_cols)
    ],
    remainder='drop'
)


# Build KNN model pipeline
pipe = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', KNeighborsClassifier())
])


# Hyperparameter tuning
params = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11, 15],
    'classifier__metric': ['euclidean', 'manhattan'],
    'classifier__weights': ['uniform', 'distance']
}


# Grid search with cross-validation
grid_model = GridSearchCV(
    pipe,
    params,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)


# Train model
grid_model.fit(x_train, y_train)


# Get best model
best_model = grid_model.best_estimator_


# Make predictions
y_pred = best_model.predict(x_test)


# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average='macro',
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average='macro',
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average='macro',
    zero_division=0
)


# Store evaluation scores
scores = {
    'accuracy': accuracy,
    'precision': precision,
    'f1': f1,
    'recall': recall,
    'best_params': grid_model.best_params_,
    'classes': best_model.classes_.tolist()
}


# Display results
print("\nBest Parameters:", grid_model.best_params_)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Recall:    {recall:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=best_model.classes_
    )
)


# Save model and evaluation scores
Path('Pkl_Files').mkdir(
    parents=True,
    exist_ok=True
)

Path('Scores').mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    best_model,
    'Pkl_Files/KNNClassifier.pkl'
)

joblib.dump(
    scores,
    'Scores/KNNClassifier_scores.pkl'
)

print("\nModel and evaluation scores saved successfully.")
