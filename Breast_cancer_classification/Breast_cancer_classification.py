
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay

# Load the dataset

df=pd.read_csv("breast_cancer.csv")

print(df)
print(df.head())
print(df.info())
print(df.shape)
print(df.describe())
print(df.columns)

# Check for missing values
print(df.isnull().sum())

# Drop unnecessary columns
df.drop(columns=['S/N', 'Year'], inplace=True)

# Analysis of categorical and numerical variables
categorical_columns = ['Menopause', 'Breast', 'Metastasis', 'Breast Quadrant', 'History', 'Diagnosis Result']
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()


print("Categorical Variables Summary:")
for col in categorical_columns:
    print(f"\n{col} Value Counts:\n", df[col].value_counts())

print("\nNumerical Variables Summary:")
print(df[numerical_columns].describe())

# Distribution of target variable

print(df['Diagnosis Result'].value_counts())

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Diagnosis Result', hue='Diagnosis Result', palette='Set2', legend=False)
plt.title("Distribution of Target Variable 'Target'")
plt.grid(False)
plt.show()

# Handle categorical variables using Label Encoding
label_encoders = {}
categorical_columns = [ 'Breast', 'Metastasis', 'Breast Quadrant', 'History', 'Diagnosis Result']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Convert tumor size and inv-nodes to numeric
df['Tumor Size (cm)'] = pd.to_numeric(df['Tumor Size (cm)'], errors='coerce')
df['Inv-Nodes'] = pd.to_numeric(df['Inv-Nodes'], errors='coerce')

# Fill missing values with median
df.fillna(df.median(), inplace=True)

# Statistical analysis
print(df.describe())



# Visualizations
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()


# Define features and target variable
X = df.drop(columns=['Diagnosis Result'])
y = df['Diagnosis Result']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train.shape)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression Model

log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)

# Make predictions
y_train_pred_log = log_model.predict(X_train)
y_test_pred_log = log_model.predict(X_test)
y_train_pred_prob_log = log_model.predict_proba(X_train)[:, 1]
y_test_pred_prob_log = log_model.predict_proba(X_test)[:, 1]


#Random Forest Evaluation
print("Logistic Regression Training Accuracy:", accuracy_score(y_train, y_train_pred_log))
print("Logistic Regression Testing Accuracy:", accuracy_score(y_test, y_test_pred_log))
print("\nClassification Report - Logistic Regression ")
print(classification_report(y_test, y_test_pred_log, target_names=['Benign', 'Malignant'], digits=2))

# Plot Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred_log, ax=axes[0], cmap="Blues")
axes[0].set_title("Training Data Confusion Matrix")
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred_log, ax=axes[1], cmap="Reds")
axes[1].set_title("Testing Data Confusion Matrix")
plt.show()




# Precision-Recall Curve for Logistic Regression
plt.figure(figsize=(8, 6))
train_pred_log, train_recall_log, _ = precision_recall_curve(y_train, y_train_pred_prob_log)
test_pred_log, test_recall_log, _ = precision_recall_curve(y_test, y_test_pred_prob_log)
plt.plot(train_recall_log, train_pred_log, color='blue', lw=2, label='Train Precision-Recall')
plt.plot(test_recall_log, test_pred_log, color='cyan', lw=2, label='Test Precision-Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Logistic Regression)')
plt.legend()
plt.show()

# ROC Curve for Logistic Regression
plt.figure(figsize=(8, 6))
train_fpr_log, train_tpr_log, _ = roc_curve(y_train, y_train_pred_prob_log)
test_fpr_log, test_tpr_log, _ = roc_curve(y_test, y_test_pred_prob_log)
train_roc_auc_log = auc(train_fpr_log, train_tpr_log)
test_roc_auc_log = auc(test_fpr_log, test_tpr_log)
plt.plot(train_fpr_log, train_tpr_log, color='blue', lw=2, label=f'Train ROC (AUC = {train_roc_auc_log:.2f})')
plt.plot(test_fpr_log, test_tpr_log, color='cyan', lw=2, label=f'Test ROC (AUC = {test_roc_auc_log:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Logistic Regression)')
plt.legend()
plt.show()


# Random Forest Model

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)
y_train_pred_prob_rf = rf_model.predict_proba(X_train)[:, 1]
y_test_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]

#Random Forest Evaluation
print("Random Forest Training Accuracy:", accuracy_score(y_train, y_train_pred_rf))
print("Random Forest Testing Accuracy:", accuracy_score(y_test, y_test_pred_rf))
print("\nClassification Report - Random Forest")
print(classification_report(y_test, y_test_pred_rf, target_names=['Benign', 'Malignant'], digits=2))

# Plot Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred_rf, ax=axes[0], cmap="Blues")
axes[0].set_title("Random Forest - Training")
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred_rf, ax=axes[1], cmap="Reds")
axes[1].set_title("Random Forest - Testing")
plt.show()


# Precision-Recall Curve for Train and Test Data
plt.figure(figsize=(8, 6))
train_pred_rf, train_recall_rf, _ = precision_recall_curve(y_train, y_train_pred_prob_rf)
test_pred_rf, test_recall_rf, _ = precision_recall_curve(y_test, y_test_pred_prob_rf)
plt.plot(train_recall_rf, train_pred_rf, color='green', lw=2, label='Train Precision-Recall')
plt.plot(test_recall_rf, test_pred_rf, color='red', lw=2, label='Test Precision-Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Random Forest)')
plt.legend()
plt.show()


# ROC Curve
plt.figure(figsize=(8, 6))
train_fpr_rf, train_tpr_rf, _ = roc_curve(y_train, y_train_pred_prob_rf)
test_fpr_rf, test_tpr_rf, _ = roc_curve(y_test, y_test_pred_prob_rf)
train_roc_auc_rf = auc(train_fpr_rf, train_tpr_rf)
test_roc_auc_rf = auc(test_fpr_rf, test_tpr_rf)
plt.plot(train_fpr_rf, train_tpr_rf, color='green', lw=2, label=f'Train ROC (AUC = {train_roc_auc_rf:.2f})')
plt.plot(test_fpr_rf, test_tpr_rf, color='red', lw=2, label=f'Test ROC (AUC = {test_roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Random Forest)')
plt.legend()
plt.show()

# Support Vector Classifier (SVC) Model
svc_model = SVC(probability=True, random_state=42)
svc_model.fit(X_train, y_train)
y_train_pred_svc = svc_model.predict(X_train)
y_test_pred_svc = svc_model.predict(X_test)
y_train_pred_prob_svc = svc_model.predict_proba(X_train)[:, 1]
y_test_pred_prob_svc = svc_model.predict_proba(X_test)[:, 1]

#Support Vector Classifier (SVC) Evaluation
print("SVC Training Accuracy:", accuracy_score(y_train, y_train_pred_svc))
print("SVC Testing Accuracy:", accuracy_score(y_test, y_test_pred_svc))
print("\nClassification Report - SVC")
print(classification_report(y_test, y_test_pred_svc, target_names=['Benign', 'Malignant'], digits=2))


# Plot Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred_svc, ax=axes[0], cmap="Blues")
axes[0].set_title("SVC - Training")
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred_svc, ax=axes[1], cmap="Reds")
axes[1].set_title("SVC - Testing")
plt.show()

# Precision-Recall Curve for Train and Test Data
plt.figure(figsize=(8, 6))
train_precision_svc, train_recall_svc, _ = precision_recall_curve(y_train, y_train_pred_prob_svc)
test_precision_svc, test_recall_svc, _ = precision_recall_curve(y_test, y_test_pred_prob_svc)
plt.plot(train_recall_svc, train_precision_svc, marker='.', label='Train')
plt.plot(test_recall_svc, test_precision_svc, marker='.', label='Test')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# ROC Curve for Train and Test Data
plt.figure(figsize=(8, 6))
train_fpr_svc, train_tpr_svc, _ = roc_curve(y_train, y_train_pred_prob_svc)
test_fpr_svc, test_tpr_svc, _ = roc_curve(y_test, y_test_pred_prob_svc)
train_roc_auc_svc = auc(train_fpr_svc, train_tpr_svc)
test_roc_auc_svc = auc(test_fpr_svc, test_tpr_svc)
plt.plot(train_fpr_svc, train_tpr_svc, color='orange', lw=2, label=f'Train ROC (AUC = {train_roc_auc_svc:.2f})')
plt.plot(test_fpr_svc, test_tpr_svc, color='brown', lw=2, label=f'Test ROC (AUC = {test_roc_auc_svc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()



# Gaussian Naive Bayes Model
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)
y_train_pred_gnb = gnb_model.predict(X_train)
y_test_pred_gnb = gnb_model.predict(X_test)
y_train_pred_prob_gnb = gnb_model.predict_proba(X_train)[:, 1]
y_test_pred_prob_gnb = gnb_model.predict_proba(X_test)[:, 1]

# Gaussian Naive Bayes Evaluation
print("Gaussian Naive Bayes Training Accuracy:", accuracy_score(y_train, y_train_pred_gnb))
print("Gaussian Naive Bayes Testing Accuracy:", accuracy_score(y_test, y_test_pred_gnb))
print("\nClassification Report - Gaussian Naive Bayes")
print(classification_report(y_test, y_test_pred_gnb, target_names=['Benign', 'Malignant'], digits=2))

#plot confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred_gnb, ax=axes[0], cmap="Blues")
axes[0].set_title("Training Data Confusion Matrix (Gaussian Naive Bayes)")
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred_gnb, ax=axes[1], cmap="Reds")
axes[1].set_title("Testing Data Confusion Matrix (Gaussian Naive Bayes)")
plt.show()

#Precision-Recall Curve (Gaussian Naive Bayes)
plt.figure(figsize=(8, 6))
train_prec_gnb, train_recall_gnb, _ = precision_recall_curve(y_train, y_train_pred_prob_gnb)
test_prec_gnb, test_recall_gnb, _ = precision_recall_curve(y_test, y_test_pred_prob_gnb)
plt.plot(train_recall_gnb, train_prec_gnb, color='blue', lw=2, label='Train Precision-Recall')
plt.plot(test_recall_gnb, test_prec_gnb, color='cyan', lw=2, label='Test Precision-Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Gaussian Naive Bayes)')
plt.legend()
plt.show()


#ROC curve
plt.figure(figsize=(8, 6))
train_fpr_gnb, train_tpr_gnb, _ = roc_curve(y_train, y_train_pred_prob_gnb)
test_fpr_gnb, test_tpr_gnb, _ = roc_curve(y_test, y_test_pred_prob_gnb)
train_roc_auc_gnb = auc(train_fpr_gnb, train_tpr_gnb)
test_roc_auc_gnb = auc(test_fpr_gnb, test_tpr_gnb)
plt.plot(train_fpr_gnb, train_tpr_gnb, color='blue', lw=2, label=f'Train ROC (AUC = {train_roc_auc_gnb:.2f})')
plt.plot(test_fpr_gnb, test_tpr_gnb, color='cyan', lw=2, label=f'Test ROC (AUC = {test_roc_auc_gnb:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Gaussian Naive Bayes)')
plt.legend()
plt.show()



# Store the accuracy scores for each model
models = ['Logistic Regression', 'Random Forest', 'SVC', 'Gaussian Naive Bayes']
train_accuracies = [
    accuracy_score(y_train, y_train_pred_log),
    accuracy_score(y_train, y_train_pred_rf),
    accuracy_score(y_train, y_train_pred_svc),
    accuracy_score(y_train, y_train_pred_gnb)
]
test_accuracies = [
    accuracy_score(y_test, y_test_pred_log),
    accuracy_score(y_test, y_test_pred_rf),
    accuracy_score(y_test, y_test_pred_svc),
    accuracy_score(y_test, y_test_pred_gnb)
]

# Plot the comparison bar chart
plt.figure(figsize=(10, 6))
x = range(len(models))
plt.bar(x, train_accuracies, width=0.4, label='Train Accuracy', align='center', color='skyblue')
plt.bar(x, test_accuracies, width=0.4, label='Test Accuracy', align='edge', color='salmon')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.xticks(x, models)
plt.legend()
plt.show()





