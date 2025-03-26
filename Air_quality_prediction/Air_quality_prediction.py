import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Air_quality_index.csv")
print(df.head())

# Check for missing values

print(df.isnull().sum())

# Handle missing values
num_imputer = SimpleImputer(strategy='mean')
df[['pollutant_min', 'pollutant_max', 'pollutant_avg']] = num_imputer.fit_transform(df[['pollutant_min', 'pollutant_max', 'pollutant_avg']])
print(df.isnull().sum())

# Encode categorical variable
label_encoder = LabelEncoder()
df['pollutant_id'] = label_encoder.fit_transform(df['pollutant_id'])
print(df['pollutant_id'])

# Using a threshold (the median of pollutant_avg)  classify air quality into two categories:
# "Poor" if the pollution level is above the threshold.
# "Better" if the pollution level is below or equal to the threshold.

# Define threshold for classification
threshold = df['pollutant_avg'].median()
print(f"Threshold for classification: {threshold}")

# Classify air quality efficiently using np.where()
df['Air_Quality'] = np.where(df['pollutant_avg'] > threshold, 'Poor', 'Better')



# Display count of each category
print(df['Air_Quality'].value_counts())

# Display first few rows with classification
print(df[['pollutant_avg', 'Air_Quality']].head())

# Count plot of Air Quality classification
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Air_Quality', hue='Air_Quality', palette={'Poor': 'red', 'Better': 'green'}, legend=False)
plt.title("Count of Poor vs. Better Air Quality")
plt.xlabel("Air Quality Category")
plt.ylabel("Count")
plt.show()

# Box plot to show distribution of pollutant_avg for both categories
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Air_Quality', y='pollutant_avg', hue='Air_Quality', palette={'Poor': 'red', 'Better': 'green'}, legend=False)

plt.title("Distribution of Pollutant Levels by Air Quality Category")
plt.xlabel("Air Quality Category")
plt.ylabel("Pollutant Average")
plt.show()

# Scatter plot showing geographical distribution of air quality
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='longitude', y='latitude', hue='Air_Quality', palette={'Poor': 'red', 'Better': 'green'})
plt.title("Geographical Distribution of Air Quality")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Air Quality")
plt.show()

# Define features and target variable
X = df[['latitude', 'longitude', 'pollutant_id', 'pollutant_min', 'pollutant_max']]
y = df['pollutant_avg']

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)


# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    }

# Display results
results_df = pd.DataFrame(results).T
print("\nModel Evaluation Results:")
print(results_df)


# Visualizing model performance
plt.figure(figsize=(10, 6))
sns.barplot(x=results_df.index, y=results_df['R2'], hue=results_df.index, palette='viridis', legend=False)
plt.ylabel("R² Score")
plt.title("Model Comparison")
plt.xticks(rotation=45)
plt.show()

# Actual vs Predicted plot (for best model)
best_model_name = results_df['R2'].idxmax()
best_model = models[best_model_name]
y_pred = best_model.predict(X_test)

plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:50], label='Actual', marker='o')
plt.plot(y_pred[:50], label='Predicted', marker='s', linestyle='dashed')
plt.xlabel("Sample Index")
plt.ylabel("Pollutant Average")
plt.title(f"Actual vs Predicted Pollutant Average - {best_model_name}")
plt.legend()
plt.show()

# KDE Plot
plt.figure(figsize=(10, 5))
sns.kdeplot(y_test.values, label='Actual', color='blue', fill=True)
sns.kdeplot(y_pred, label='Predicted', color='red', fill=True)
plt.xlabel("Pollutant Average")
plt.ylabel("Density")
plt.title(f"KDE Plot of Actual vs Predicted - {best_model_name}")
plt.legend()
plt.show()

"""Air Quality Prediction & Analysis
This project focuses on predicting air pollution levels using various regression models and classifying air quality into "Poor" and "Better" categories. 
The key outcomes are:

Air Quality Classification:

If pollutant_avg is high, air quality is classified as "Poor" (indicating more pollution).
If pollutant_avg is low, air quality is classified as "Better" (indicating cleaner air).

Geographical Insights:

A scatter plot helps visualize the distribution of air quality across different locations based on latitude and longitude.
Areas with higher pollutant_avg levels are red (Poor air quality), while areas with lower pollution levels are green (Better air quality).

Model Evaluation & Prediction:

Machine learning models like Linear Regression, Decision Tree, Random Forest, and Gradient Boosting were trained to predict pollutant_avg.
The best-performing model (determined using R² Score, MAE, MSE, RMSE) provides predictions on air pollution levels.
A comparison bar chart shows the performance of different models.

Data Visualization for Insights:

Box plots display how pollutant_avg varies between "Poor" and "Better" air quality.
Actual vs. Predicted plots evaluate how well the model predicts pollutant levels.
KDE plots compare the distribution of actual vs. predicted pollution levels.
Conclusion:
This project successfully predicts air pollution levels based on pollutant concentration and geographic data. The insights help in understanding pollution trends and identifying areas with poor air quality. 
This can assist policymakers in making data-driven decisions for pollution control and urban planning"""


