import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

url = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"

data = pd.read_csv(url)
data.head()

data_bought = data["bought"]
data_behavior = data[["home", "how_it_works", "contact"]]

train_bought = data_bought[:75]
train_behavior = data_behavior[:75]

test_bought = data_bought[75:]
test_behavior = data_behavior[75:]

model = LinearSVC()
model.fit(train_behavior, train_bought)

predictions = model.predict(test_behavior)

hit_rate = accuracy_score(test_bought, predictions) * 100

print("A taxa de acerto foi %.2f" % hit_rate)