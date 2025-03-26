
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

#load datset
crop = pd.read_csv("Crop_recommendation.csv")
print(crop.head())
print(crop.describe())


#check for missing values
print(crop.isnull().sum())

#count
print(crop['label'].value_counts())


#Creating dictionary
crop_dict = {
    'rice': 1,
    'maize': 2,
    'jute': 3,
    'cotton': 4,
    'coconut': 5,
    'papaya': 6,
    'orange': 7,
    'apple': 8,
    'muskmelon': 9,
    'watermelon': 10,
    'grapes': 11,
    'mango': 12,
    'banana': 13,
    'pomegranate': 14,
    'lentil': 15,
    'blackgram': 16,
    'mungbean': 17,
    'mothbeans': 18,
    'pigeonpeas': 19,
    'kidneybeans': 20,
    'chickpea': 21,
    'coffee': 22
}
crop['crop_num']=  crop['label'].map(crop_dict)
print(crop)

crop.drop(['label'],axis=1,inplace=True)
crop.head()


#tain-test data
X = crop.drop(['crop_num'],axis=1)
y = crop['crop_num']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)

#scaling
ms = MinMaxScaler()

X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)
print(X_train)



# create instances of all models
models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Bagging': BaggingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Extra Trees': ExtraTreeClassifier(),
}
# Train and evaluate models
accuracy_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = acc
    print(f"{name} with accuracy: {acc:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("=" * 60)


# Select best model
best_model_name = max(accuracy_scores, key=accuracy_scores.get)
best_model = models[best_model_name]
print(f"Best model selected: {best_model_name} with accuracy {accuracy_scores[best_model_name]:.4f}")


# Function for recommendation
def recommendation(N, P, k, temperature, humidity, ph, rainfall):
    features = pd.DataFrame([[N, P, k, temperature, humidity, ph, rainfall]], columns=X.columns)  # Fix here
    transformed_features = ms.transform(features)
    prediction = best_model.predict(transformed_features)
    return prediction[0]


#1
N = 50       # Nitrogen content in soil
P = 60      # Phosphorus content in soil
K = 30        # Potassium content in soil
temperature = 30 # Temperature (in Celsius)
humidity = 80     # Humidity (%)
ph = 7.0         # pH of the soil
rainfall = 120   # Rainfall (mm)
predicted_crop = recommendation(N, P, K, temperature, humidity, ph, rainfall)

# Crop dictionary for decoding prediction
crop_dict_inv = {v: k for k, v in crop_dict.items()}
if predicted_crop in crop_dict_inv:
    print(f"{crop_dict_inv[predicted_crop]} is the best crop to be cultivated.")
else:
    print("Sorry, we are not able to recommend a proper crop for this environment.")


#2
N, P, k, temperature, humidity, ph, rainfall = 40, 50, 50, 40.0, 20, 100, 100
predicted_crop = recommendation(N, P, k, temperature, humidity, ph, rainfall)

# Crop dictionary for decoding prediction
crop_dict_inv = {v: k for k, v in crop_dict.items()}
if predicted_crop in crop_dict_inv:
    print(f"{crop_dict_inv[predicted_crop]} is the best crop to be cultivated.")
else:
    print("Sorry, we are not able to recommend a proper crop for this environment.")
