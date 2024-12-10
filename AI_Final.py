import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
#print("Number of rows in the original DataFrame:", len(df))
#print("Number of rows in the filtered DataFrame:", len(filtered_df))

#Remove invalid values
filtered_df = filtered_df[filtered_df['tip_amount'] > 0]
filtered_df = filtered_df[filtered_df['total_amount'] > 0]

#total amount outliers
filtered_df['Z-score'] = zscore(filtered_df['total_amount'])
rm_totalamount_outliers = filtered_df[(filtered_df['Z-score'] > 3) | (filtered_df['Z-score'] < -3)]

#tip amount outliers 
filtered_df['Z-score'] = zscore(filtered_df['tip_amount'])
rm_tipamount_outliers = filtered_df[(filtered_df['Z-score'] > 3) | (filtered_df['Z-score'] < -3)]

#trip distance outliers 
filtered_df['Z-score'] = zscore(filtered_df['trip_distance'])
rm_tripdist_outliers = filtered_df[(filtered_df['Z-score'] > 3) | (filtered_df['Z-score'] < -3)]

#Generate tip percentage column
filtered_df['tip_percentage'] = (filtered_df['tip_amount'] / filtered_df['total_amount']) * 100

#split into 24 hour time slots
filtered_df['pickup_hour'] = filtered_df['tpep_pickup_datetime'].dt.hour
random_rows = filtered_df.sample(n=20)

# Print the 20 random rows
#print(random_rows[['tip_amount', 'total_amount', 'tip_percentage']])


#filter data frame to the columns we need
useful_df = filtered_df[['pickup_hour', 'tip_percentage','trip_distance' ]]
useful_df = useful_df.dropna()
print (useful_df.describe())

normalized_arr = normalize(useful_df)
print(normalized_arr)

final_data_frame = pd.DataFrame(normalized_arr, columns=['pickup_hour', 'tip_percentage','trip_distance'])
final_data_frame2 = final_data_frame.sample(300)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot using the DataFrame columns
ax.scatter(final_data_frame2['pickup_hour'], final_data_frame2['tip_percentage'], final_data_frame2['trip_distance'], c='b', marker='o')

# Set labels
ax.set_xlabel('X pickup')
ax.set_ylabel('Y tip)')
ax.set_zlabel('Z distance')

# Show the plot
plt.show()
