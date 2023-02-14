import pandas

from sklearn import linear_model

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

machine = linear_model.LinearRegression()

return_values = kfold_template.run_kfold(data, target, machine, 4, True, False, False)

print(return_values)
all_r2 = [i[0] for i in return_values]
print("Average R2: ", sum(all_r2)/len(all_r2))

