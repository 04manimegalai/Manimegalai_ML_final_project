
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#Load data
pd.set_option('display.max_columns',None)
df = pd.read_csv("snsdata.csv")
print(df.head())

#Summary Statistics of Numerical Variables
print(df.describe())

# Summary Statistics of Categorical Variables
print(df.describe(include='object'))

# Treating Missing Values
print(df.isnull().sum())

 # number of male and female candidates

print(df['gender'].value_counts())

# number of male, female and msiing values

print(df['gender'].value_counts(dropna = False))


# fill all the null values in gender column with “No Gender”
df['gender'] = df['gender'].fillna('not disclosed')
print(df.isnull().sum())


# Handle missing values (fill missing 'age' values with median)
imputer = SimpleImputer(strategy="median")
df[['age']] = imputer.fit_transform(df[['age']])

print(df.isnull().sum())


#  Outliers
# Visualizing outliers in 'age' column
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['age'])
plt.title("Boxplot of Age Column")
plt.show()


q1 = df['age'].quantile(0.25)
q3 = df['age'].quantile(0.75)
iqr = q3-q1
print(iqr)

df = df[(df['age'] > (q1 - 1.5*iqr)) & (df['age'] < (q3 + 1.5*iqr))]
print(df['age'].describe())

#visualisation after outlier removal
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['age'])
plt.title("Boxplot of Age Column")
plt.show()



names = df.columns[4:40]
scaled_feature = df.copy()
print(names)



# Select only numerical columns for scaling
features = scaled_feature[names].values  # Convert to NumPy array

# Initialize and fit StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)  # Fit and transform together

# Store the scaled values back
scaled_feature[names] = features_scaled



#Convert object variable to numeric
def gender_to_numeric(x):
    if x=='M':
        return 1
    if x=='F':
        return 2
    if x=='not disclosed':
        return 3


# Selecting only numerical features for scaling (Excluding categorical columns like 'gender')

# Ensure numeric features are selected
numeric_features = df.drop(columns=['gender'])  # Keep only numerical features

# Standardize Features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numeric_features.values)
print(scaled_features)


kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)
model = kmeans.fit(scaled_features)
# Creating a funtion with KMeans to plot "The Elbow Curve"
wcss = []
for i in range(1,20):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,20),wcss)
plt.title('The Elbow Curve')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') ##WCSS stands for total within-cluster sum of square
plt.show()

kmeans = KMeans(n_clusters=5)
kmeans.fit(scaled_features)
df['Cluster'] = kmeans.labels_


print(df[['Cluster']].head())
print("Cluster Centers:\n", kmeans.cluster_centers_)
print(df['Cluster'].value_counts())

#Visualizing Clusters
sns.scatterplot(x=features[:,0], y=features[:,1], hue=df['Cluster'], palette='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.show()


#Bar Plot for Cluster Distribution
plt.figure(figsize=(8, 5))
df['Cluster'].value_counts().plot(kind='bar', color=['blue', 'green', 'red'])
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Bar Plot of Cluster Distribution')
plt.show()

# number of students belonging to each cluster
size_array = df.groupby(['Cluster'])['age'].count().to_list()

print(size_array)

# check the cluster statistics
# Select only numeric columns before applying mean
numeric_cols = df.select_dtypes(include=['number']).columns  # Get only numeric columns
selected_cols = ['basketball', 'football', 'soccer', 'softball', 'volleyball', 'swimming',
                 'cheerleading', 'baseball', 'tennis', 'sports', 'cute', 'sex', 'sexy', 'hot',
                 'kissed', 'dance', 'band', 'marching', 'music', 'rock', 'god', 'church',
                 'jesus', 'bible', 'hair', 'dress', 'blonde', 'mall', 'shopping', 'clothes',
                 'hollister', 'abercrombie', 'die', 'death', 'drunk', 'drugs']

# Filter only the numeric columns that exist in the dataset
valid_cols = [col for col in selected_cols if col in numeric_cols]

# Apply mean only to valid numeric columns
print(df.groupby(['Cluster'])[valid_cols].mean())
