''' SortData.py '''

import pandas as pd

# Read in data from all files
df_2018 = pd.read_csv('Data/pitch_movement_2018.csv')
df_2019 = pd.read_csv('Data/pitch_movement_2019.csv')
df_2020 = pd.read_csv('Data/pitch_movement_2020.csv')
df_2021 = pd.read_csv('Data/pitch_movement_2021.csv')
df_2022 = pd.read_csv('Data/pitch_movement_2022.csv')
df_2023 = pd.read_csv('Data/pitch_movement_2023.csv')

# Concatenate all data into one dataframe
df = pd.concat([df_2018, df_2019, df_2020, df_2021, df_2022, df_2023])

# Filter out the 'Slurve' and 'Screwball' pitch types
df_filtered = df[(df['pitch_type_name'] != 'Slurve') & (df['pitch_type_name'] != 'Screwball')]

# Sort the filtered data by pitch type name
df_sorted = df_filtered.sort_values(by=['pitch_type_name'])

# Write the sorted, filtered data to csv
df_sorted.to_csv('pitch_movement_all.csv', index=False)
