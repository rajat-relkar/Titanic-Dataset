#Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv('titanic.csv')

print(data.info())
print(data.isnull().sum())
# print(data['Pclass'].unique())

#Data Cleaning and Feature Engineering
def preprocess_data(df):
    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

    # df['Embarked'].fillna("S", inplace=True)
    df.drop(columns=['Embarked'], inplace=True)

    fill_missing_age(df)

    #Convert gender to numerical value
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    #Feature Engineering

    #Find family size by combining Siblings and parents
    df['FamilySize'] = df['SibSp'] + df['Parch']
    
    #If travelling alone, 1 else 0
    df['IsAlone'] = np.where(df['FamilySize'] == 0, 1, 0)

    #Divide passengers into 4 classes based on ticket fare
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)

    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 15, 25, 40, 60, np.inf], labels=False)

    return df

#Fill in the missing ages based on the passenger class
#If age of passenger from class 1 is missing, fill the median age of passengers of class 1
def fill_missing_age(df):
    age_fill_map = {}

    for pclass in df['Pclass'].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df['Pclass'] == pclass]['Age'].median()

#Replace the missing age values with median age of the passenger class
    df["Age"] = df.apply(lambda row: age_fill_map[row['Pclass']] if pd.isnull(row['Age']) 
                         else row['Age'], axis=1)
    

data = preprocess_data(data)

#Create Features
X = data.drop(columns=['Survived'])
y = data['Survived']

#ML Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Hyperparameter tuning - KNN
def tune_model(X_train, y_train):
    param_grid = {
        "n_neighbors": range(1, 21), 
        "metric": ["euclidean", "manhattan", "minkowski"],
        "weights": ["uniform", "distance"]
    }

    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

best_model = tune_model(X_train, y_train)

#Predictions and evaluation
def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)

    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, X_test, y_test)

print(f'Accuracy: {accuracy*100:.2f}%')
print("Confusion Matrix:")
print(matrix)


#Plot Model
def plot_model(matrix):
    plt.figure(figsize=(10,7))
    sns.heatmap(matrix, fmt='d', annot=True,
                xticklabels=['Survived', 'Not Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Values")
    plt.ylabel("Actual Values")
    plt.show()

plot_model(matrix)

