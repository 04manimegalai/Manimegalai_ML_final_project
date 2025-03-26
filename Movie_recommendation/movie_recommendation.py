
import pandas as pd
import numpy as np
from sklearn.preprocessing import  MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import  silhouette_score

import random
# warnings
import warnings
warnings.filterwarnings('ignore')

df_netflix = pd.read_csv("netflix_tvshows_movies_titles.csv")
df_amazon =  pd.read_csv("Amazonprime_tvshows_movies_titles.csv")
df_hbo =  pd.read_csv("HBO_tvshows_movies_titles.csv")

print(df_netflix)
print(df_amazon)
print(df_hbo)

#concatenation
df = pd.concat([df_netflix, df_amazon, df_hbo], axis=0)
print(df)
print(df.columns)

#Data preprocessing
df_movies = df.drop_duplicates()
print(df_movies)
print(df_movies.isnull().sum())

# Drop unnecessary columns
df_movies.drop(['description', 'age_certification'], axis=1, inplace=True)
print(df_movies.columns)

print(df['production_countries'])

#Remove unwanted characters from the 'production_countries' column

df_movies['production_countries'] = df_movies['production_countries'].str.replace(r"\[", '', regex=True).str.replace(r"'", '', regex=True).str.replace(r"\]", '', regex=True)
print(df_movies['production_countries'])

# Extract the first country from the cleaned 'production_countries' column

df_movies['lead_prod_country'] = df_movies['production_countries'].str.split(',').str[0]

print(df_movies['lead_prod_country'])


#Calculate the number of countries involved in the production of each movie

df_movies['prod_countries_cnt'] = df_movies['production_countries'].str.split(',').str.len()
print(df_movies['prod_countries_cnt'])

# Replace any empty values in the 'lead_prod_country' column with NaN

df_movies['lead_prod_country'] = df_movies['lead_prod_country'].replace('', np.nan)
print(df_movies['lead_prod_country'])

print(df_movies['genres'])


# Remove unwanted characters from the 'genres' column

df_movies['genres'] = df_movies['genres'].str.replace(r"\[", '', regex=True).str.replace(r"'", '', regex=True).str.replace(r"\]", '', regex=True)

# Extract the first genre from the cleaned 'genres' column

df_movies['main_genre'] = df_movies['genres'].str.split(',').str[0]

# Replace any empty values in the 'main_genre' column with NaN

df_movies['main_genre'] = df_movies['main_genre'].replace('', np.nan)
print(df_movies['main_genre'])


#  Drop unnecessary columns 'genres' and 'production_countries' from the DataFrame
df_movies.drop(['genres', 'production_countries'], axis=1, inplace=True)
print(df_movies.shape)

print(df_movies.isnull().sum())

# Drop rows with any missing values to clean the dataset
df_movies.dropna(inplace=True)

# Set the 'title' column as the DataFrame index
print(df_movies.set_index('title', inplace=True))
print(df_movies)

# Drop the 'id' and 'imdb_id' columns as they are not needed for further analysis
df_movies.drop(['id', 'imdb_id'], axis=1, inplace=True)
print(df_movies.shape)

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# Create dummy variables for categorical columns ('type', 'lead_prod_country', 'main_genre')
dummies = pd.get_dummies(df_movies[['type', 'lead_prod_country', 'main_genre']], drop_first=True)

# Concatenate the dummy variables with the original DataFrame
df_movies_dum = pd.concat([df_movies, dummies], axis=1)

# Drop the original categorical columns after creating dummy variables
df_movies_dum.drop(['type', 'lead_prod_country', 'main_genre'], axis=1, inplace=True)
print(df_movies_dum)

# Apply MinMaxScaler to scale the data for model training
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_movies_dum)
df_scaled = pd.DataFrame(df_scaled, columns=df_movies_dum.columns)

# Display the scaled DataFrame

print(df_scaled)




# DBSCAN Hyperparameter Tuning
best_eps, best_min_samples, best_silhouette = None, None, -1
eps_values = [0.2, 0.5, 1]
min_samples_values = [5, 10, 30]

for eps in eps_values:
    for min_samples in min_samples_values:
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        clusterer.fit(df_scaled)
        labels = clusterer.labels_
        if len(set(labels)) > 1:
            silhouette_avg = silhouette_score(df_scaled, labels)
            if silhouette_avg > best_silhouette:
                best_eps, best_min_samples, best_silhouette = eps, min_samples, silhouette_avg

# Train DBSCAN with best hyperparameters
dbscan_model = DBSCAN(eps=best_eps, min_samples=best_min_samples)
dbscan_model.fit(df_scaled)
df_movies['dbscan_clusters'] = dbscan_model.labels_
print("Best DBSCAN Parameters:", best_eps, best_min_samples)


print(df_movies['dbscan_clusters'].value_counts())


#movie recommendation
def recommend_movie(movie_name: str):
    # Convert the input movie name to lowercase for case-insensitive matching
    movie_name = movie_name.lower()

    # Create a new column 'name' with lowercase movie names for comparison
    df_movies['name'] = df_movies.index.str.lower()

    # Find the movie that matches the input name
    movie = df_movies[df_movies['name'].str.contains(movie_name, na=False)]

    if not movie.empty:
        # Get the cluster label of the input movie
        cluster = movie['dbscan_clusters'].values[0]

        # Get all movies in the same cluster
        cluster_movies = df_movies[df_movies['dbscan_clusters'] == cluster]

        # If there are more than 5 movies in the cluster, randomly select 5
        if len(cluster_movies) >= 5:
            recommended_movies = random.sample(list(cluster_movies.index), 5)
        else:
            # If fewer than 5, return all the movies in the cluster
            recommended_movies = list(cluster_movies.index)

        # Print the recommended movies
        print('--- We can recommend you these movies ---')
        for m in recommended_movies:
            print(m)
    else:
        print('Movie not found in the database.')

s = input('Input movie name: ')

print("\n\n")
recommend_movie(s)