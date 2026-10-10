
import pandas as pd
import joblib
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

df = pd.read_csv(r'Dataset\disaster_cleaned_dataset.csv')
df.drop(columns=['Unnamed: 0', 'event_id', 'date', 'event_within_7d', 'event_within_30d',
                 'event_within_90d', 'timeframe_bucket', 'post_severity', 'post_impact_score'],
        errors='ignore', inplace=True)
df.dropna(subset=['disaster_type'], inplace=True)

x = df.drop(columns=['disaster_type']).copy()
y = df['disaster_type'].astype(str)

ordinal_orders = {}
ordinal_orders = {c: v for c, v in ordinal_orders.items() if c in x.columns}
ordinal_cols = list(ordinal_orders)

for col in x.select_dtypes(include=['object', 'category']).columns:
    values = set(x[col].dropna().astype(str).str.strip().str.lower())
    if values and values.issubset({'true', 'false'}):
        x[col] = x[col].map(lambda v: str(v).strip().lower() == 'true' if pd.notna(v) else None)
    elif values and values.issubset({'yes', 'no'}):
        x[col] = x[col].map(lambda v: str(v).strip().lower() == 'yes' if pd.notna(v) else None)

boolean_cols = x.select_dtypes(include=['bool']).columns.tolist()
numerical_cols = [c for c in x.select_dtypes(include=['number']).columns if c not in ordinal_cols]
nominal_cols = [c for c in x.select_dtypes(include=['object', 'category']).columns if c not in ordinal_cols]

print('Numerical:', numerical_cols, '\nOrdinal:', ordinal_cols,
      '\nNominal:', nominal_cols, '\nBoolean:', boolean_cols)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

transformers = []
if numerical_cols:
    transformers.append(('numerical', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numerical_cols))
if ordinal_cols:
    transformers.append(('ordinal', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=[ordinal_orders[c] for c in ordinal_cols],
        handle_unknown='use_encoded_value', unknown_value=-1)),
        ('scaler', StandardScaler())
    ]), ordinal_cols))
if nominal_cols:
    transformers.append(('nominal', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), nominal_cols))
if boolean_cols:
    transformers.append(('boolean', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=[[False, True] for _ in boolean_cols],
        handle_unknown='use_encoded_value', unknown_value=-1))
    ]), boolean_cols))

preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
pipe = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))
])

params = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__class_weight': [None, 'balanced']
}

cv = min(5, int(y_train.value_counts().min()))
if cv < 2:
    raise ValueError('Each class needs at least two training samples for cross-validation.')

grid_model = GridSearchCV(pipe, params, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1)
grid_model.fit(x_train, y_train)

best_model = grid_model.best_estimator_
y_pred = best_model.predict(x_test)

scores = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
    'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
    'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
    'best_params': grid_model.best_params_,
    'best_cv_f1_macro': grid_model.best_score_,
    'classes': best_model.classes_.tolist()
}

print('\nBest Parameters:', grid_model.best_params_)
print(f"Accuracy: {scores['accuracy']:.4f}")
print(f"Precision: {scores['precision']:.4f}")
print(f"Recall: {scores['recall']:.4f}")
print(f"F1 Score: {scores['f1']:.4f}")
print('\nClassification Report:\n', classification_report(y_test, y_pred, zero_division=0))
print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred, labels=best_model.classes_))

Path('Pkl_Files').mkdir(parents=True, exist_ok=True)
Path('Scores').mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, 'Pkl_Files/RandomForestClassifier.pkl')
joblib.dump(scores, 'Scores/RandomForestClassifier_scores.pkl')
print('\nModel and scores saved.')
