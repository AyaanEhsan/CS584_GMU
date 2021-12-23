import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_dataset = pd.read_csv("./credit_train.csv")

#Calculating the correlation using the corr method in pandas
correlation = train_dataset.corr()

#Plotting the heat map for correlation obtained
plt.figure(figsize=(16, 10))
mask = np.zeros_like(correlation)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(correlation, mask=mask,annot= True,cmap="YlGnBu")
plt.show()



