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

print("Bez skalowania")
models = [knn(), svm(), dt(max_depth=3)]
for model in models:
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)
  print(f'model = {model},\n{confusion_matrix(y_test, y_pred)}')

print("Standard Scaler")
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
models = [knn(), svm(), dt(max_depth=3)]
for model in models:
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)
  print(f'model = {model},\n{confusion_matrix(y_test, y_pred)}')



model = dt(max_depth=3)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(f'model = {model},\n{confusion_matrix(y_test, y_pred)}')

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=data.columns[:-1], class_names=['N', 'Y'])
plt.show()

