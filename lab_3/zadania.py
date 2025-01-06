import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC as svm
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data


data = pd.read_excel('practice_lab_3.xlsx')
title = list(data)[:-1]

# zamiana tekstowych cech binarnych na liczbowe
binary_features = {'Gender' : 'Female',
                   'Married' : 'Yes',
                   'Education' : 'Graduate',
                   'Self_Employed' : 'Yes'}
for column in binary_features:
    data = qualitative_to_0_1(data, column, binary_features[column])

# rozbicie 'one hot'
cat_feature = pd.Categorical(data.Property_Area)
one_hot = pd.get_dummies(cat_feature)
data = pd.concat([data, one_hot], axis=1)
data = data.drop(columns=['Property_Area'])

# przeniesienie wyniku na koniec
y = data['Loan_Status']
data = data.drop(columns=['Loan_Status'])
data = pd.concat([data, y], axis=1)

# przygotowanie danych
x = data.values[:,:-1]
y = data.values[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# trenowanie modelu
models = [knn(), svm(), dt(max_depth=3)]
for model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f'Model = {model}')
    print(confusion_matrix(y_test, y_pred))
    print(f"Dokładność = {accuracy_score(y_test, y_pred)}")
    
# skalowanie danych
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# trenowanie po przeskalowaniu danych (czy to dobrze dla drzewa?)
for model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f'Model = {model}')
    print(confusion_matrix(y_test, y_pred))
    print(f"Dokładność = {accuracy_score(y_test, y_pred)}")

    
model = dt(max_depth=5)                                                       
model.fit(x_train, y_train)                                                   
y_pred = model.predict(x_test)
print(f'Model = {model}')
print(confusion_matrix(y_test, y_pred))
print(f"Dokładność = {accuracy_score(y_test, y_pred)}")

columns = data.drop(columns=['Loan_Status']).columns.to_list()
plt.figure(figsize=(16,8))
tree_vis = plot_tree(model, feature_names=columns, class_names=['0', '1'], fontsize=12)    