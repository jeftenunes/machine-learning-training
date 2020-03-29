import pandas as pd
from sklearn.model_selection import train_test_split

url = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"

data = pd.read_csv(url)
data.head()

data_bought = data["bought"]
data_behavior = data[["home", "how_it_works", "contact"]]

train_bought, test_bought, train_behavior, test_behavior = train_test_split(data_behavior, data_bought, test_size = 0.25)

print(train_bought.shape)
print(test_bought.shape)