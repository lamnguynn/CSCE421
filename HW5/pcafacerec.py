# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_olivetti_faces

sns.set()
data = fetch_olivetti_faces()
data.keys()

inputs=data.data
target=data.target
images=data.imagesinputs.shape

print("Showing faces.....")
plt.figure(figsize=(20,25))
for i in range(50,70):
    plt.subplot(4,5,i-49)
    plt.imshow(data.images[i], cmap=plt.cm.gray)
plt.show()