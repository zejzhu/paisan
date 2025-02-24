import dt
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

xTrain = pd.read_csv("./space_trainx.csv").to_numpy()
yTrain = pd.read_csv("./space_trainy.csv").to_numpy().flatten()
xTest = pd.read_csv("./space_testx.csv").to_numpy()
yTest = pd.read_csv("./space_testy.csv").to_numpy().flatten()
print("data loaded")

#plot the accuracy of test and train dataset as a function of max depth from 1 to 15
#default min leaf samples is 5
mdplotmatrix = []
# for md in range(1, 16):
#     print(f"processing max depth={md}")
#     birtree = dt.DecisionTree('entropy', md, 5)
#     birtree.train(xTrain, yTrain)
#     rtrain = birtree.predict(xTrain)
#     rtest = birtree.predict(xTest)
#     atrain = dt._accuracy(rtrain, yTrain)
#     atest = dt._accuracy(rtest, yTest)
#     mdplotmatrix.append([md, atrain, atest])
#model needs to be retrained for each max depth

#plot the accuracy of test and train dataset as a function of min leaf samples from 1 to 15
#default max depth is 5
for mls in range(1, 16):
    print(f"processing min leaf samples={mls}")
    birtree = dt.DecisionTree('entropy', 5, mls)
    birtree.train(xTrain, yTrain)
    rtrain = birtree.predict(xTrain)
    rtest = birtree.predict(xTest)
    atrain = dt._accuracy(rtrain, yTrain)
    atest = dt._accuracy(rtest, yTest)
    mdplotmatrix.append([mls, atrain, atest])

#seaborn
plotdataframe = pd.DataFrame(mdplotmatrix, columns=['Max Depth', 'Training', 'Test'])
print(plotdataframe)

plt.figure(figsize=(10, 6))
sns.lineplot(data=plotdataframe, x='Max Depth', y='Training', label='Training Accuracy')
sns.lineplot(data=plotdataframe, x='Max Depth', y='Test', label='Test Accuracy')

plt.title("Effect of Min Leaf Sample on Accuracy")
plt.xlabel("MinLeafSample")
plt.ylabel("Accuracy")
plt.legend()
plt.show()