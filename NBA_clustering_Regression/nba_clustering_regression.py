
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score


# Load the dataset

df = pd.read_csv("NBA-PLAYERS.csv")
print(df)
print(df.columns)

#unique values
for col in df.columns:
    print(f"{col}: {df[col].unique()[:5]}")

# Handle Missing Values
df.replace("?", np.nan, inplace=True)

# Drop irrelevant or fully missing columns (modify if needed)
drop_columns = ['Player.x', 'Player_ID']  # Remove non-numeric, non-useful columns
df.drop(columns=drop_columns, inplace=True)


print(df.dtypes)
# Convert 'Season' to numeric by extracting the starting year
df['Season'] = df['Season'].str[:4].astype(int)
print(df.dtypes)

print(df.isnull().sum())

missing_cols = df.columns[df.isnull().sum() > 0]
print("Columns with Missing Values:", missing_cols)

#Columns with Missing Values: Index(['Pos2', 'FG.', 'X3P.', 'X2P.', 'eFG.', 'FT.', 'Salary', 'mean_views', 'Pvot', 'PRank', 'Mvot', 'MRank'],dtype='object')

# Convert percentage columns to numeric
percentage_cols = ['FG.', 'X3P.', 'X2P.', 'eFG.', 'FT.']
print(df[percentage_cols].dtypes)

for col in percentage_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
print(df[percentage_cols].dtypes)

df[percentage_cols] = df[percentage_cols].apply(lambda x: x.fillna(x.median()))
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')  # Convert to numeric
df['mean_views'] = pd.to_numeric(df['mean_views'], errors='coerce')  # Convert to numeric



df['Salary'] = df['Salary'].fillna(df['Salary'].median())
df['mean_views'] = df['mean_views'].fillna(df['mean_views'].median())



vote_cols = ['Pvot', 'PRank', 'Mvot', 'MRank']
df[vote_cols] = df[vote_cols].fillna(0)

df['Pos2'] = df['Pos2'].fillna(df['Pos2'].mode()[0])


print(df.isnull().sum())


#identify numeric and non numeric
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
        print(f"{col} is numeric")
    except:
        print(f"{col} is NOT numeric")


# Select only numeric columns (int and float)
numeric_cols = df.select_dtypes(include=['number']).columns
print(numeric_cols)

# Convert categorical columns using Label Encoding
label_cols = ['Pos1', 'Pos2', 'Tm', 'Play']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col].astype(str))  # Convert to string to avoid NaN issues
print(df[['Pos1', 'Pos2', 'Tm', 'Play']])

# One-Hot Encoding for high-cardinality categorical variables
df = pd.get_dummies(df, columns=['Conference', 'Role'], drop_first=True)

# Convert boolean columns to integer (0/1)
one_hot_columns = [col for col in df.columns if 'Conference' in col or 'Role' in col]
df[one_hot_columns] = df[one_hot_columns].astype(int)
print(df[one_hot_columns].nunique())  # Check unique values (should be 0 and 1)
print(df[one_hot_columns].head())  # Check sample rows

# Standardize numerical data (important for clustering & PCA)
scaler = StandardScaler()
df_scaled_array = scaler.fit_transform(df[numeric_cols])
df_scaled = pd.DataFrame(df_scaled_array, columns=numeric_cols)
print(df_scaled.head())


pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

# Plot the explained variance ratio
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.grid(True)
plt.show()

#KMeans clustering
wcss = []  # Within-cluster sum of squares
silhouette_scores = []

# Test for clusters from 2 to 10
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_result)

    wcss.append(kmeans.inertia_)

    # Calculate silhouette score
    score = silhouette_score(pca_result, cluster_labels)
    silhouette_scores.append(score)

# Plot Elbow Method
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(2, 11), wcss, marker='o', linestyle='--', label='WCSS')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.grid(True)

# Plot Silhouette Score
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='x', linestyle='-', color='green', label='Silhouette Score')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Cluster Quality')
plt.grid(True)

plt.tight_layout()
plt.show()




# Choose the optimal number of clusters based on the elbow method and silhouette score
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 since range starts from 2
print(f'Optimal number of clusters based on silhouette score: {optimal_clusters}')



# Perform K-means clustering with optimal clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(pca_result)

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', label='Centroids')
plt.title('K-means Clustering with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()

# Compute the Silhouette Score for the final clustering
final_silhouette_score = silhouette_score(pca_result, cluster_labels)
print(f'\nSilhouette Score for the final clustering with {optimal_clusters} clusters: {final_silhouette_score:.4f}')


# Combine PCA DataFrame with the original one
df = pd.concat([df, pd.DataFrame(pca_result, columns=['PC1', 'PC2'])], axis=1)
df['cluster'] = kmeans.labels_

# Group by cluster and calculate mean values for profiling
cluster_profile = df.groupby('cluster').mean().reset_index()

# Display the cluster profile
print("Cluster Profiling Summary:")
print(cluster_profile[['cluster', 'PTS', 'AST', 'TRB', 'Salary', 'mean_views', 'FG.', 'X3P.', 'FT.']])


# Visualizations for Cluster Characteristics
 # Metrics for visualization
player_stats = ['PTS', 'AST', 'TRB']
performance_metrics = ['FG.', 'X3P.', 'FT.']
popularity_metrics = ['Salary', 'mean_views']


# Function to create bar charts
def plot_cluster_barplots(metrics, title, palette):
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6))

    for i, metric in enumerate(metrics):
        sns.barplot(x='cluster', y=metric, hue='cluster', data=df, palette=palette, legend=False, ax=axes[i])
        axes[i].set_title(f'{metric} by Cluster')
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel(metric)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


# Plot characteristics by cluster
plot_cluster_barplots(player_stats, "Player Statistics by Cluster", "muted")
plot_cluster_barplots(performance_metrics, "Performance Metrics by Cluster", "deep")
plot_cluster_barplots(popularity_metrics, "Popularity Metrics by Cluster", "pastel")


# supervised learning
features = ['PTS', 'AST', 'TRB', 'FG.', 'X3P.', 'FT.', 'mean_views',
            'G', 'Season']
target = 'Salary'

X = df[features]
y = df[target]

#  Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#  Define the hyperparameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

#  Instantiate the RandomForest model
rf = RandomForestRegressor(random_state=42)

#  Use RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                   n_iter=50, cv=5, n_jobs=-1, random_state=42, verbose=2, scoring='r2')

#  Fit the model
random_search.fit(X_train, y_train)

#  Best parameters
best_params = random_search.best_params_
print("Best Parameters (RandomizedSearchCV):", best_params)

#  Train the best model
best_rf_model = RandomForestRegressor(**best_params, random_state=42)
best_rf_model.fit(X_train, y_train)

#  Predict salaries
y_pred = best_rf_model.predict(X_test)

#  Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (RandomizedSearchCV): {mse:.2f}")
print(f"R² Score (RandomizedSearchCV): {r2:.4f}")




#  Sort the values for better visualization
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
comparison_df = comparison_df.sort_values(by='Actual').reset_index(drop=True)

# Plot the predicted vs actual salaries with a line plot
plt.figure(figsize=(14, 7))

plt.plot(comparison_df['Actual'], label='Actual Salary', color='blue', marker='o', linestyle='-')
plt.plot(comparison_df['Predicted'], label='Predicted Salary', color='orange', marker='x', linestyle='--')

plt.title('Predicted vs. Actual Salaries (Line Plot)')
plt.xlabel('Player Index')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()

importance =best_rf_model.feature_importances_

# Display feature importance in a bar plot
importances_df = pd.DataFrame({'Feature': features, 'Importance': importance})
importances_df = importances_df.sort_values(by='Importance', ascending=False)


# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', hue='Feature', data=importances_df, palette='viridis', legend=False)

plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.grid(True)
plt.show()




