import pandas as pd
import dabl
titanic = pd.read_csv(dabl.datasets.data_path("titanic.csv"))
dabl.plot_supervised(titanic, 'survived')
# Target looks like classification
# baseline score: 0.500
import matplotlib.pyplot as plt; plt.show()
