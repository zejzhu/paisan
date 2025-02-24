import knn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

xTrain = pd.read_csv("./simxTrain.csv").to_numpy()
yTrain = pd.read_csv("./simyTrain.csv").to_numpy().flatten()
xTest = pd.read_csv("./simxTest.csv").to_numpy()
yTest = pd.read_csv("./simyTest.csv").to_numpy().flatten()
print("data loaded")

numBirts = xTrain.shape[0]

#will end up with 3 columns: value of k, training acc, test acc
plotmatrix = []

#model only needs to be trained once
birtknn = knn.Knn(1)
birtknn.train(xTrain, yTrain)

#test for all values of k from 1 to n
for k in range(1, numBirts):
    print(f"processing k={k}")
    birtknn.k = k
    rtrain = birtknn.predict(xTrain)
    rtest = birtknn.predict(xTest)

    #find accuracy and store results into plotmatrix
    atrain = knn.accuracy(rtrain, yTrain)
    atest = knn.accuracy(rtest, yTest)
    plotmatrix.append([k, atrain, atest])

#seaborn time
plotdataframe = pd.DataFrame(plotmatrix, columns=['k', 'Training', 'Test'])
print(plotdataframe)

plt.figure(figsize=(10, 6))
sns.lineplot(data=plotdataframe, x='k', y='Training', label='Training Accuracy')
sns.lineplot(data=plotdataframe, x='k', y='Test', label='Test Accuracy')

plt.title("Effect of k value on Accuracy")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.legend()
plt.show()