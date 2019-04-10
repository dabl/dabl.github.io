import pandas as pd
import dabl
titanic = pd.read_csv(dabl.datasets.data_path("titanic.csv"))
dabl.plot_supervised(titanic, 'survived')
# Target looks like classification
import matplotlib.pyplot as plt; plt.show()
