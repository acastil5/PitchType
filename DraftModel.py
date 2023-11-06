''' DraftModel.py '''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
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

# Save the imputer
dump(imputer, 'imputer.joblib')

# Now encode categorical data
label_encoders = {}
for column in categorical_features + ['pitch_type_name']:  # Add 'pitch_type_name' to the list
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    label_encoders[column] = label_encoder
    # Save each encoder
    dump(label_encoder, f'{column}_encoder.joblib')  # This line now saves the 'pitch_type_name' encoder

# Combine numeric and categorical features
feature_cols = numeric_features + categorical_features

# Prepare the data
X = df[feature_cols]
y = df['pitch_type_name']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rfc.fit(X_train, y_train)

# Save the model to a file
dump(rfc, 'rfc_model.joblib')

# No need to save the imputer again, it's already saved above
# dump(imputer, 'imputer.joblib')

# Predict on the test set
y_pred = rfc.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
