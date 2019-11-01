import pandas as pd
import dabl
titanic = pd.read_csv(dabl.datasets.data_path("titanic.csv"))
dabl.plot(titanic, 'survived')
# Target looks like classification
# Linear Discriminant Analysis training set score: 0.578
import matplotlib.pyplot as plt; plt.show()
