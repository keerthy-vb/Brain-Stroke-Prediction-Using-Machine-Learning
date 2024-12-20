# -*- coding: utf-8 -*-
"""LogisticRegression.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ujgO_as9FeXo94XlLgDPPPzhH__6F7lD
"""

from google.colab import drive
drive.mount('/content/drive')

"""## Preprocessed dataset"""

import pandas as pd
data= pd.read_csv("/content/drive/MyDrive/Mini_Project_BrainStroke/Dataset /updated_dataset.csv")
data.head(7)

data.drop('Unnamed: 0', axis=1, inplace=True)

data.stroke.value_counts()

data.info()

"""## Split dataset into test and train"""

x = data.drop("stroke", axis=1)
x.head()

y = data[['stroke']]
y.head()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

x_train

y_train

"""### Logistic Regression"""

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(x_train,y_train['stroke'].values.ravel())

print("Accuracy is :" )
lg.score(x_test,y_test)

prep = lg.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, prep))

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, prep)

from sklearn.metrics import log_loss

y_pred_proba = lg.predict_proba(x_test)
loss = log_loss(y_test, y_pred_proba)
print(f"Log Loss: {loss}")

"""## Confusion Matrix"""

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Compute and display the confusion matrix
conf_matrix = confusion_matrix(y_test, prep, labels=[0,1])

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

class_labels = {0, 1}

data.columns

import pandas as pd


models = {
    "Logistic Regression": lg
}

# Define the labels for prediction
class_labels = {0: "No Stroke", 1: "Stroke"}

user_data_list = [

    [[55, 0, 0, 105.5, 23.5, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]],
    [[45, 1, 1, 80.3, 27.8, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1]],

]

# Iterate through each user data and make predictions
for user_data in user_data_list:
    print(f"Testing on data: {user_data}")
    for model_name, model in models.items():
        # Convert test data to DataFrame with correct column names
        user_data_df = pd.DataFrame(user_data, columns=[
            'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
           'gender_Male', 'gender_Other', 'ever_married_Yes',
            'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed',
            'work_type_children', 'Residence_type_Urban',
            'smoking_status_formerly smoked', 'smoking_status_never smoked',
            'smoking_status_smokes'
        ])

        # Make prediction
        prediction = model.predict(user_data_df)
        class_label = class_labels[prediction[0]]
        print(f"{model_name} Prediction: {prediction[0]} - {class_label}")
    print("\n")

import pandas as pd

# Assuming rf is your trained RandomForestClassifier model
models = {
    "Logistic Regression": lg
}

# Define the labels for prediction
class_labels = {0: "No Stroke", 1: "Stroke"}

user_data_list = [
    [[70, 1, 1, 200.5, 75.5, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1]],
    [[45, 1, 1, 80.3, 27.8, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1]],
]

# Iterate through each user data and make predictions
for user_data in user_data_list:
    print(f"Testing on data: {user_data}")
    for model_name, model in models.items():
        # Convert test data to DataFrame with correct column names
        user_data_df = pd.DataFrame(user_data, columns=[
            'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
           'gender_Male', 'gender_Other', 'ever_married_Yes',
            'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed',
            'work_type_children', 'Residence_type_Urban',
            'smoking_status_formerly smoked', 'smoking_status_never smoked',
            'smoking_status_smokes'
        ])

        # Convert the DataFrame to a NumPy array to avoid the warning
        user_data_array = user_data_df.values

        # Make prediction using the NumPy array
        prediction = model.predict(user_data_array)
        class_label = class_labels[prediction[0]]
        print(f"{model_name} Prediction: {prediction[0]} - {class_label}")

import pickle

pickle.dump(lg, open('lg_new.pkl', 'wb'))