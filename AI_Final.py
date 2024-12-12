import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import normalize #scalar?? 
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#imports for elbow
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


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

scaler = StandardScaler()
normalized_arr = scaler.fit_transform(useful_df)
print(normalized_arr)

final_data_frame = pd.DataFrame(normalized_arr, columns=['pickup_hour', 'tip_percentage','trip_distance'])
final_data_frame2 = final_data_frame.sample(1000)

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

#ELBOW PLOT - getting appropriate k value  
# Initialize lists to store distortion and inertia values
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)

# Fit K-means for different values of k
X = final_data_frame2.to_numpy()
for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=42).fit(X)
    
    # Calculate distortion as the average squared distance from points to their cluster centers
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)**2) / X.shape[0])
    
    # Inertia is calculated directly by KMeans
    inertias.append(kmeanModel.inertia_)
    
    # Store the mappings for easy access
    mapping1[k] = distortions[-1]
    mapping2[k] = inertias[-1]
    
# Plotting the graph of k versus Distortion
plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.grid()
plt.show()


kmeans = KMeans(n_clusters = 4, random_state = 42)

pred = kmeans.fit_predict(X)
import matplotlib.cm as cm

X_scaled = scaler.inverse_transform(X)
original_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Assuming X is a 2D NumPy array with at least 4 features
# and pred contains the predicted cluster labels from k-means
# kmeans.cluster_centers_ is the array of cluster centers

fig = plt.figure(figsize=(14, 6))

# First 3D plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
sc1 = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=pred, cmap=cm.Accent, s=20, alpha=0.1)
ax1.grid(True)
for center in original_centers:
    ax1.scatter(center[0], center[1], center[2], marker='^', c='blue', s=100)
ax1.set_xlabel("pickup-hour")
ax1.set_ylabel("tip-percentage")
ax1.set_zlabel("trip distance")

plt.tight_layout()
plt.show()
