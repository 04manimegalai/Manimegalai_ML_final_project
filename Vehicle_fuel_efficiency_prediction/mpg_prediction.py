import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("cluster_mpg.csv")
print(df)

# Exploratory Data Analysis (EDA)
print(df.head())
print(df.info())
print(df.describe())

# Checking for missing values
print(df.isnull().sum())

print(df['origin'].value_counts())

df_w_dummies = pd.get_dummies(df.drop('name',axis=1))
print(df_w_dummies)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_w_dummies)
print(scaled_data)




# Heatmap Visualization
scaled_df = pd.DataFrame(scaled_data,columns=df_w_dummies.columns)
plt.figure(figsize=(15,8))
sns.heatmap(scaled_df,cmap='magma')
plt.title("Heatmap of Scaled Features")
plt.show()

# Clustermap Visualization
sns.clustermap(scaled_df,row_cluster=False)
plt.title("Cluster Map Without Row Clustering")
plt.show()

sns.clustermap(scaled_df,col_cluster=False)
plt.title("Cluster Map Without Column Clustering")
plt.show()


model = AgglomerativeClustering(n_clusters=4)
cluster_labels = model.fit_predict(scaled_df)
print(cluster_labels)

# Scatter Plot of Clusters
plt.figure(figsize=(12,4),dpi=200)
sns.scatterplot(data=df,x='mpg',y='weight',hue=cluster_labels)
plt.title("Clusters based on MPG vs Weight")
plt.show()

# Dendrogram Analysis



# Compute the linkage matrix once
linkage_matrix = linkage(scaled_data, method='ward')

# Full Dendrogram
plt.figure(figsize=(20, 10))
dendrogram(linkage_matrix)
plt.title("Dendrogram - Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Truncated Dendrogram (Last 48 Clusters)
plt.figure(figsize=(20, 10))
dendrogram(linkage_matrix, truncate_mode='lastp', p=48)
plt.title("Truncated Dendrogram - Last 48 Clusters")
plt.xlabel("Cluster Index")
plt.ylabel("Distance")
plt.show()


print(scaled_df.describe())

# Identifying extreme MPG values
print(df['mpg'].idxmax())
print(df['mpg'].idxmin())



# Euclidean Distance Calculation
a = scaled_df.iloc[320]
b = scaled_df.iloc[28]
dist = np.linalg.norm(a-b)
print("\nEuclidean Distance Between Two Points:", dist)

# Rule of Thumb for Optimal Clusters
print("\nRule of Thumb for Cluster Count:", np.sqrt(len(scaled_df)))

# Drop non-numeric columns
df = df.drop(columns=['name', 'origin'], errors='ignore')


# Define features (X) and target variable (y)
y = df['mpg']
X = df.drop(columns=['mpg'])

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



# Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)



print("\nRandom Forest Regressor Performance:")
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R² Score:", r2_score(y_test, y_pred_rf))

# Visualization: Actual vs Predicted MPG
plt.figure(figsize=(8,5))
plt.plot(y_test.values, label="Actual MPG", marker='o')

plt.plot(y_pred_rf, label="Predicted MPG (RF)", marker='^')
plt.xlabel("Sample Index")
plt.ylabel("MPG")
plt.title("Actual vs Predicted MPG")
plt.legend()
plt.show()

# # Prepare new vehicle data
new_vehicle_data = pd.DataFrame([{
    'cylinders': 8,
    'displacement': 307,
    'horsepower': 130,
    'weight': 3504,
    'acceleration': 12,
    'model_year': 70
}])

# Ensure correct order of columns before scaling
new_vehicle_data = new_vehicle_data.reindex(columns=X.columns, fill_value=0)

# Scale the new input using the trained MinMaxScaler
new_vehicle_scaled = scaler.transform(new_vehicle_data)

# Predict MPG
predicted_mpg = rf.predict(new_vehicle_scaled)
print(f"Predicted MPG: {predicted_mpg[0]}")
