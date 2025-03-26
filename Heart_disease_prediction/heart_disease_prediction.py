import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Load dataset
data = pd.read_csv("heart_disease_prediction.csv")
print(data.head())
print(data.columns)
print(data.shape)
print(data.describe())

# Checking for missing values
print("\nMissing Values in Dataset:")
print(data.isnull().sum())

# Check class distribution
print(data['target'].value_counts(normalize=True))
# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()
#
# Splitting features and target variable
X = data.drop(columns=['target'])  # Assuming 'target' is the column for heart disease
y = data['target']



# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning for Random Forest
rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
gscv_rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_jobs=-1, verbose=1)
gscv_rf.fit(X_train, y_train)
best_rf = gscv_rf.best_estimator_
print(f'Best Random Forest Parameters: {gscv_rf.best_params_}')

# Hyperparameter tuning for Gradient Boosting
gb_params = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]}
gscv_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params, cv=5, n_jobs=-1, verbose=1)
gscv_gb.fit(X_train, y_train)
best_gb = gscv_gb.best_estimator_
print(f'Best Gradient Boosting Parameters: {gscv_gb.best_params_}')

# Hyperparameter tuning for SVM
svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
gscv_svm = GridSearchCV(SVC(probability=True, random_state=42), svm_params, cv=5, n_jobs=-1, verbose=1)
gscv_svm.fit(X_train, y_train)
best_svm = gscv_svm.best_estimator_
print(f'Best SVM Parameters: {gscv_svm.best_params_}')

# Model Selection
models = {"Random Forest": best_rf, "Gradient Boosting": best_gb, "Support Vector Machine": best_svm}

# Training and Evaluating Models
best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'\n{name} Accuracy: {acc:.2f}')
    print(classification_report(y_test, y_pred))


    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# Best Model Selection
print(f'\nBest Model: {best_model}')

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, best_model.predict(X_test)), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Best Model Confusion Matrix')
plt.show()



# ROC Curve
y_probs = best_model.predict_proba(X_test)[:, 1]  # Get probability estimates
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Best Model ROC Curve')
plt.legend()
plt.show()

print(f'Best Model AUC Score: {roc_auc:.2f}')



# Predicting Heart Disease Risk for Example Patients

#Example
def predict_heart_disease(input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)  # Ensure column names match
    input_scaled = scaler.transform(input_df)
    prediction = best_model.predict(input_scaled)
    return "At Risk" if prediction[0] == 1 else "Not at Risk"



# Example Input Sets
example_patients = [
    [1, 65, 3, 150, 250, 1, 1, 120, 1, 2.5, 2, 2, 3],
    [0, 50, 2, 130, 220, 0, 0, 160, 0, 1.2, 1, 1, 2],
    [0, 40, 1, 120, 180, 0, 0, 180, 0, 0.5, 1, 0, 2]
]

for i, patient in enumerate(example_patients, 1):
    risk_prediction = predict_heart_disease(patient)
    print(f'Example Patient {i} Predicted Risk: {risk_prediction}')

