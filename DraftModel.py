import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from joblib import dump

# Load your dataset
df = pd.read_csv('pitch_movement_all.csv')

# Define the numeric and categorical features
numeric_features = ['avg_speed', 'pitches_thrown', 'total_pitches', 'pitches_per_game',
                    'pitch_per', 'pitcher_break_z', 'league_break_z', 'diff_z', 'rise',
                    'pitcher_break_x', 'league_break_x', 'diff_x', 'tail', 
                    'percent_rank_diff_z', 'percent_rank_diff_x']
categorical_features = ['pitch_hand']

# Apply imputation only on numeric features
imputer = SimpleImputer(strategy='mean')
df[numeric_features] = imputer.fit_transform(df[numeric_features])
dump(imputer, 'imputer.joblib')  # Save the imputer

# Encode categorical data
label_encoders = {}
for column in categorical_features + ['pitch_type_name']:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    label_encoders[column] = label_encoder
    dump(label_encoder, f'{column}_encoder.joblib')

# Combine numeric and categorical features
feature_cols = numeric_features + categorical_features

# Prepare the data
X = df[feature_cols]
y = df['pitch_type_name']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rfc = RandomForestClassifier(random_state=42)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
}

grid_search = GridSearchCV(rfc, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
best_rfc = grid_search.best_estimator_
dump(best_rfc, 'best_rfc_model.joblib')  # Save the best Random Forest model

# SVM Model
svc = SVC()
svc.fit(X_train, y_train)

# Neural Network Model
mlp = MLPClassifier()
mlp.fit(X_train, y_train)

# Evaluating all models
models = {'Random Forest': best_rfc, 'SVM': svc, 'Neural Network': mlp}
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"{name} Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}\n")
