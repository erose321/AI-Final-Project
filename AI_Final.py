import pandas as pd
from scipy.stats import zscore

# File path to the Parquet file
parquet_file = 'yellow_tripdata_2018-08.parquet'
# Read the Parquet file into a DataFrame
df = pd.read_parquet(parquet_file, engine='auto')

# Display the number of columns in the DataFrame
num_columns = len(df.columns)
print("Number of columns:", num_columns)

# Display the number of columns in the DataFrame
num_rows = len(df)
print("Number of rows:", num_rows)
df.head()

df.describe()

# Read the CSV file into a pandas DataFrame
taxi_zones = pd.read_csv('taxi_zone_lookup.csv')
manhattan_zones = taxi_zones[taxi_zones['Borough'] == 'Manhattan']
manhattan_zone_numbers = manhattan_zones['LocationID'].tolist()
print("Manhattan taxi zone numbers:", manhattan_zone_numbers)

# filter so we only have the Manhattan data
filtered_df = df[
    df['PULocationID'].isin(manhattan_zone_numbers) |
    df['DOLocationID'].isin(manhattan_zone_numbers)
]

# Print some information about the filtered DataFrame
print("Number of rows in the original DataFrame:", len(df))
print("Number of rows in the filtered DataFrame:", len(filtered_df))

#Remove invalid values (negative duration, negative cost, negative tip, negative distance)
df = df[df['tip_amount'] <= 0]
df = df[df['total_amount'] <= 0]

#Remove invalid values
df = df[df['tip_amount'] <= 0]
df = df[df['total_amount'] <= 0]

#total amount outliers
df['Z-score'] = zscore(df['total_amount'])
rm_totalamount_outliers = df[(df['Z-score'] > 3) | (df['Z-score'] < -3)]

#tip amount outliers 
df['Z-score'] = zscore(df['tip_amount'])
rm_tipamount_outliers = df[(df['Z-score'] > 3) | (df['Z-score'] < -3)]

#trip distance outliers 
df['Z-score'] = zscore(df['trip_distance'])
rm_tripdist_outliers = df[(df['Z-score'] > 3) | (df['Z-score'] < -3)]

#Generate tip percentage column
filtered_df['tip_percentage'] = (filtered_df['tip_amount'] / filtered_df['total_amount']) * 100

#split into 24 hour time slots
filtered_df['pickup_hour'] = filtered_df['tpep_pickup_datetime'].dt.hour
random_rows = filtered_df.sample(n=20)

# Print the 20 random rows
print(random_rows[['tip_amount', 'total_amount', 'tip_percentage']])

#filter data frame to the columns we need
