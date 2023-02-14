import pandas

from sklearn.neighbors import KNeighborsClassifier

import kfold_template

dataset = pandas.read_csv("abalone.csv")
dataset = dataset.sample(frac=1, random_state=1)

dataset['Rings'] = dataset['Rings'] + 1.5

# print(dataset['Rings'].mean())

dataset['Age Group'] = pandas.cut(dataset['Rings'], [0, 9, 11, 13, 100], labels=[1,2,3,4])

print(dataset['Age Group'].value_counts())

dataset = dataset.drop(['Rings'], axis=1)




target = dataset.iloc[:,8]
target = target.values

data = dataset.iloc[:,0:8]
data = pandas.get_dummies(data, columns=['Sex'])
data = data.values


trials = []
for w in ['uniform', 'distance']:
  for k in range(2, 50):
    machine = KNeighborsClassifier(n_neighbors=k, weights=w)
    return_values = kfold_template.run_kfold(data, target, machine, 4, False, True, False)
    all_r2 = [i[0] for i in return_values]
    average_r2 = sum(all_r2)/len(all_r2)
    trials.append((average_r2, k, w))

# print(trials)

trials.sort(key=lambda x: x[0], reverse=True)

print(trials[:5])




