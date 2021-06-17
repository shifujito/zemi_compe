import pandas as pd

data = pd.read_csv('data/train.csv')
data['Gender'] = data['Gender'].replace('Male', 1).replace('Female', 0)
print(data)
