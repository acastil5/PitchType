''' DraftAI.py '''

import pandas as pd
import numpy as np
from joblib import load

# Define the subset of features you want to prompt the user for
user_features = ['avg_speed', 'pitcher_break_x', 'pitcher_break_z', 'pitch_hand', 'rise', 'tail']

# Also define the numeric features from the model script here
numeric_features = ['avg_speed', 'pitches_thrown', 'total_pitches', 'pitches_per_game',
                    'pitch_per', 'pitcher_break_z', 'league_break_z', 'diff_z', 'rise',
                    'pitcher_break_x', 'league_break_x', 'diff_x', 'tail', 
                    'percent_rank_diff_z', 'percent_rank_diff_x']

# Load the imputer and label encoders from the trained model
imputer = load('imputer.joblib')
pitch_hand_encoder = load('pitch_hand_encoder.joblib')
pitch_type_encoder = load('pitch_type_name_encoder.joblib')
rfc_model = load('best_rfc_model.joblib')

# Function to prompt the user for input and preprocess that input
def get_user_input():
    user_input = {}
    print("Please enter the following pitch data:")

    inputs = {
        'avg_speed': "Enter the average speed of the pitch (in MPH): ",
        'pitcher_break_x': "Enter the value for horizontal break (in inches): ",
        'pitcher_break_z': "Enter the value for vertical break (in inches): ",
        'pitch_hand': "Enter the pitcher's hand (L for left-handed, R for right-handed): "
    }

    for key, message in inputs.items():
        while True:
            try:
                if key in ['avg_speed', 'pitcher_break_x', 'pitcher_break_z']:
                    value = float(input(message))
                elif key == 'pitch_hand':
                    value = input(message).strip().upper()
                    if value not in ['L', 'R']:
                        raise ValueError("Invalid input. Please enter 'L' or 'R'.")
                user_input[key] = value
                break
            except ValueError as e:
                print(f"Invalid input: {e}")

    return user_input

# Function to encode categorical features using the fitted LabelEncoders
def encode_categorical_data(user_input):
    user_input['pitch_hand'] = pitch_hand_encoder.transform([user_input['pitch_hand']])[0]
    return user_input

# Function to impute missing features
def impute_missing_features(encoded_user_input, imputer, numeric_features):
    # Create a full features dictionary with NaN for imputation
    full_features = {feature: [np.nan] for feature in numeric_features}

    # Update with the user input for features that were given
    for feature in encoded_user_input:
        if feature in full_features:
            full_features[feature] = [encoded_user_input[feature]]

    # Convert to DataFrame for imputation
    features_df = pd.DataFrame(full_features)

    # Impute missing values
    imputed_features = imputer.transform(features_df)

    # Ensure the order of columns matches the order expected by the model
    imputed_features = pd.DataFrame(imputed_features, columns=imputer.get_feature_names_out())

    # Now add the encoded pitch_hand to the dataframe since it's not missing and doesn't need imputation
    imputed_features['pitch_hand'] = encoded_user_input['pitch_hand']

    # Ensure the order of columns matches the order expected by the model,
    # especially if pitch_hand is supposed to be the first or in another specific position
    return imputed_features


# Function to predict the pitch type
def predict_pitch_type(model, imputed_features):
    # Predict the pitch type
    pitch_type_code = model.predict(imputed_features)
    # Decode the pitch type
    pitch_type = pitch_type_encoder.inverse_transform(pitch_type_code)
    return pitch_type[0]

# Interactive script to get user input and predict pitch type
def main():
    while True:
        user_input = get_user_input()
        encoded_user_input = encode_categorical_data(user_input)
        imputed_features = impute_missing_features(encoded_user_input, imputer, numeric_features)
        pitch_type = predict_pitch_type(rfc_model, imputed_features)
        print(f"The predicted pitch type is: {pitch_type}")

        if input("\nDo you want to predict another pitch type? (yes/no): ").strip().lower() != 'yes':
            break

if __name__ == '__main__':
    main()
