import pandas as pd
import dabl
titanic = pd.read_csv(dabl.datasets.data_path("titanic.csv"))
dabl.plot(titanic, 'survived')
# Expected:
## Target looks like classification
## Linear Discriminant Analysis training set score: 0.578
## [[<Figure size 1500x1500 with 30 Axes>, <Figure size 1600x400 with 4 Axes>, <Figure size 640x480 with 1 Axes>], None]
#
import matplotlib.pyplot as plt; plt.show()
