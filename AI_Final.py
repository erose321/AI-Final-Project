import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def filter_NYC_Borough(dataframe, taxizone_info, name_borough):
    """Function that filters a NYC Taxi Dataframe to be only the borough entered as the parameter
    Taxizone_info contains the information of which zone is which borough. The function returns only
    The taxirides in that borough"""
    borough_taxi_zones = taxizone_info[taxizone_info['Borough'] == 'Manhattan']
    borough_zone_numbers = borough_taxi_zones['LocationID'].tolist()
    # filter only rides that started or ended in that borough
    filter_borough_df = dataframe[
        dataframe['PULocationID'].isin(borough_zone_numbers) |
        dataframe['DOLocationID'].isin(borough_zone_numbers)
    ]
    return filter_borough_df


def filter_outliers_Z_score(dataframe, columnname):
    """filter the chosen column [columnname] from dataframe by getting the Z-score 
    and then taking out all entries that are more than 3 standard deviations from the mean"""
    dataframe['Z-score'] = zscore(dataframe[columnname])
    return dataframe[(dataframe['Z-score'] > 3) | (dataframe['Z-score'] < -3)]


def prepare_and_filter_data_frame(dataframe, zone_filter):
    """This function handles all the data-preprocessing that needs to be done:
        -filtering for relevance
        -removing outliers
        -feature engineering columns
        It calls multiple helper functions to do so.
        """
    # filter for Manhattan
    filtered_df = filter_NYC_Borough(dataframe, zone_filter, 'Manhattan')

    # Remove invalid values
    filtered_df = filtered_df[filtered_df['tip_amount'] > 0]
    filtered_df = filtered_df[filtered_df['total_amount'] > 0]

    # remove all outliers in relevant columns
    rm_totalamount_outliers = filter_outliers_Z_score(
        filtered_df, 'total_amount')
    rm_tipamount_outliers = filter_outliers_Z_score(filtered_df, 'tip_amount')
    rm_tripdist_outliers = filter_outliers_Z_score(
        filtered_df, 'trip_distance')

    # Generate tip percentage column
    filtered_df['tip_percentage'] = (
        filtered_df['tip_amount'] / filtered_df['total_amount']) * 100

    # split into 24 hour time slots
    filtered_df['pickup_hour'] = filtered_df['tpep_pickup_datetime'].dt.hour

    # filter data frame to the 3 columns we need
    useful_df = filtered_df[['pickup_hour', 'tip_percentage', 'trip_distance']]
    # drop rows with N/A values, which means invalid measurement
    useful_df = useful_df.dropna()
    print(useful_df.describe())
    return useful_df


def scatter_plot_sample(dataframe):
    """Makes a scatterplot of data from [dataframe]"""
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot using the DataFrame columns
    ax.scatter(dataframe['pickup_hour'], dataframe['tip_percentage'],
               dataframe['trip_distance'], c='b', marker='o')

    # Set labels
    ax.set_title('normalized data scatter')
    ax.set_xlabel('X pickup')
    ax.set_ylabel('Y tip)')
    ax.set_zlabel('Z distance')


def plot_elbow_plot(dataframe_sample):
    """Plots the elbow plot for k-mean clustering for a certain sample of a dataframe
        From there, it can be manually determined which k-value seems best for the clustering
        algorithm.
    """
    # Initialize lists to store distortion and inertia values
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, 10)

    # Fit K-means for different values of k
    X = dataframe_sample
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=42).fit(X)

        # Calculate distortion as the average squared distance from points to their cluster centers
        distortions.append(sum(np.min(
            cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)**2) / X.shape[0])

        # Calculate inertia
        inertias.append(kmeanModel.inertia_)

    # Plotting the graph of k versus Distortion
    plt.figure(2)
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for k-values')
    plt.grid()


def plot_k_mean_clusters(X_scaled, pred, original_centers):
    """Plots the k-mean clusters for X-data using the de-scaled centers"""
    fig = plt.figure(3, figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    sc1 = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1],
                      X_scaled[:, 2], c=pred, cmap=cm.Accent, s=20, alpha=0.1)
    ax1.grid(True)
    for center in original_centers:
        ax1.scatter(center[0], center[1], center[2],
                    marker='^', c='blue', s=100)
    ax1.set_title('k-mean clusters for Manhattan Taxi rides (august 2018)')
    ax1.set_xlabel("pickup-hour (in 24 1-hour timeframes)")
    ax1.set_ylabel("tip-percentage (in %)")
    ax1.set_zlabel("trip distance (in miles)")

    plt.tight_layout()


def run_clustering_algo():
    """Main function: Reads file, filters data, generates 3 plots: scattered sample, elbow plot and k-mean clusters"""
    # File path to the Parquet file
    parquet_file = 'yellow_tripdata_2018-08.parquet'
    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(parquet_file, engine='auto')

    # give some information on the data
    df.describe()

    # Read the CSV file into a pandas DataFrame
    taxi_zones = pd.read_csv('taxi_zone_lookup.csv')

    # filter dataframe as described in the documentation of prepare_and_filter_data_frame()
    filtered_df = prepare_and_filter_data_frame(df, taxi_zones)

    # scale data
    scaler = StandardScaler()
    normalized_arr = scaler.fit_transform(filtered_df)

    # choose relevant columns
    final_data_frame = pd.DataFrame(normalized_arr, columns=[
                                    'pickup_hour', 'tip_percentage', 'trip_distance'])
    # print some information on dataframe
    print(final_data_frame.describe())
    # take a reasonable size sample for calculations
    final_data_frame_sample = final_data_frame.sample(40000)

    # visualize data
    scatter_plot_sample(final_data_frame_sample)
    X = final_data_frame.to_numpy()
    # plot the elbow plot to find ideal number of clusters
    plot_elbow_plot(X)
    
    X_sample = final_data_frame_sample.to_numpy()
    # kmean clusters
    kmeans = KMeans(n_clusters=4, random_state=42)
    pred = kmeans.fit_predict(X_sample)
    X_scaled = scaler.inverse_transform(X_sample)
    original_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    print("Coordinates of original centers:")
    print(original_centers)
    plot_k_mean_clusters(X_scaled, pred, original_centers)
    plt.show()


# run the algorithm
run_clustering_algo()
