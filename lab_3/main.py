import pandas as pd
import numpy as np

data = pd.read_excel('practice_lab_3.xlsx')

# zamiana cechy jakościowej male/female na czytelniejsze 0/1
mask = data['Gender'].values == 'Female'
data['Gender'][mask] = 1 # female
data['Gender'][~mask] = 0 # male

mask = data['Education'].values == 'Graduate'
data['Education'][mask] = 1 # graduate
data['Education'][~mask] = 0 # not graduate

# zamiana kilku cech metodą 1-z-n
# wybór cechy Property_Area
cat_feature = pd.Categorical(data.Property_Area)
# przekształcenie danyc
one_hot = pd.get_dummies(cat_feature)
# aktualizacja listy z danymi
data = pd.concat([data, one_hot], axis=1)
data = data.drop(columns=['Property_Area'])


features = data.columns
values = data.values.astype(np.float)
x = values[:, :-1]
y = values[:, -1]


