import pandas

from sklearn.neighbors import KNeighborsRegressor

import kfold_template

dataset = pandas.read_csv("abalone.csv")
dataset = dataset.sample(frac=1, random_state=1)

target = dataset.iloc[:,8]
target = target + 1.5
target = target.values

data = dataset.iloc[:,0:8]
data = pandas.get_dummies(data, columns=['Sex'])
data = data.values


# print(target)
# print(data)

trials = []
for w in ['uniform', 'distance']:
  for k in range(2, 50):
    machine = KNeighborsRegressor(n_neighbors=k, weights=w)
    return_values = kfold_template.run_kfold(data, target, machine, 4, True, False, False)
    all_r2 = [i[0] for i in return_values]
    average_r2 = sum(all_r2)/len(all_r2)
    trials.append((average_r2, k, w))

# print(trials)

trials.sort(key=lambda x: x[0], reverse=True)

print(trials[:5])



