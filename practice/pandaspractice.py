import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
pandas is used for computing tabular (tables of) data efficiently
it has lots of advanced data aggregation and statitstical functions

mixed data types ARE allowed in a single table
columns and rows can be named

#a Series is a 1 dimensional object with a sequence of values
#a DataFrame is a 2d object (often thought of as tabular data)
"""

#pandas can read a csv file and other common data file formats
# if u give it the filepath

birdtable = pd.read_csv("./practice table.csv")

"""
pandas attributes

columns (column labels)
index (row labels or indices)
shape (# of rows and cols)
size (# of elements in the dataframe)
dtypes (data types of the columns)
"""

"""
pandas methods

head(n) prints the first n rows
tail(n) prints the last n rows
describe() gives summary statistics of the dataframe
"""
#print(birdtable)
#print(birdtable.dtypes)
#print(birdtable.columns)
#print(birdtable.describe())
#print(birdtable.head(2))

"""
common pandas statistical calculations on series

a.sum()
a.prod()        product of series
a.mean()
a.median()
a.std()         standard deviation
a.max()
a.min()
a.argmax()      index of max
a.argmin()      index of min
"""
#make a new np array
chickenarray = np.array([123,3,2,6,65,353444])
#make a series called chicken using the np array
chicken = pd.Series(chickenarray)
#print the product of all the elements in chicken

"""
pandas indexing

loc - axis labels (row labels and column names) to subset data
ex. foo.loc[0:1, "x1"]
    foo.loc[3:5, ["x1", "y"]]

iloc - uses position of rows and columns only (MUST BE INTEGERS)
ex. foo.iloc[0:1, 0]
    foo.iloc[3:5, [0,2]]
ALSO iloc doesnt include the last number in the index
"""
#loc is short for locate :D

#prints indexes 0-4 of the column labeled "species"
#print(birdtable.loc[0:4, "species"])

#prints indexes 0-3 of the columns index 2 and 5
#print(birdtable.iloc[0:4, [2, 5]])
#note that iloc doesnt include the end index (4) bc idk integers are different ig 


"""
pandas subsetting

you can use comparison to find samples with specific values in columns
    foo.loc[foo["y"] == 0, :]
    foo.loc[foo["x1"] < 1, :]
the colon is where the column selection goes (colon means select all)


another example: finding the samples with the largest and smallest value in a column
    foo.iloc[foo["x1"].argmax(), :]
    foo.iloc[foo["x2"].argmin(), :]

"""
#print all columns of the rows(tuples) where species = budgie
#yea thats what the : is for it says to select all the columns
#print(birdtable.loc[birdtable["species"] == "budgie", :])

#lets print the median of weight
#print(birdtable["weight (g)"].median())

#lets print the standard deviation of age
#print(birdtable["age"].std())

#lets print.. THE MEAN OF WEIGHT FOR BUDGIES!
#print(birdtable.loc[birdtable["species"] == "budgie", "weight (g)"].mean())
#THATS WHY IM THE FUCKING GOAT

"""
time to learn ab data visualization lil bro

matplotlib is a low level graph plotting library
pandas (via matplotlib)
seaborn (via matplotlib) is a high level plotting library

theres also lots of other plotting libraries yaknow
"""


"""
but heres the methods for matplotlib
plt.show() displays all open figures
plt.savefig(filename) saves the figure in the filename specified

scatter() makes the scatterplot
    you can color code points by setting c to the appropriate categorical variable

set_xlabel(), set_ylabel(), set_title() labels your plot

legend()  plots the legend explicitly

boxplot info below
"""

#lets try sm simple
x = [12, 34, 2, 51, 46, 78, 89]
y = [3, 85, 73, 5, 24, 24, 8]
vibes = [0, 0, 0, 1, 1, 1, 1]

#making subplots allows u to have multiple graphs in the same figure!
#silly = plt.subplot()
#x is x axis, y is y axis, vibes determines the color
#sillyscatter = silly.scatter(x, y, c=vibes)

#silly.set_xlabel("altitude")
#silly.set_ylabel("evilness")
#silly.set_title("birt flight evilness")

#making a legend for sillyscatter
#sillylegend = silly.legend(*sillyscatter.legend_elements(), loc = "upper right", title="vibes yeah")


#ok lets try a box plot!
#goofy = plt.subplot()

#this is a series of all the ages of budgies and mourning doves
#data = [birdtable.loc[birdtable["species"] == "budgie", 'age'], birdtable.loc[birdtable["species"] == "mourning dove", 'age']]

#goofy.boxplot(data)

#goofy.set_title("boxplot of age distribution based on species")
#goofy.set_xticklabels(["budgerigar", "mourning doverino"])
#goofy.set_xlabel("species")
#goofy.set_ylabel("ageness")


"""
ok now lets try using pandas to plot directly from dataframes

call dataframe.plot(kind=<method>) or dataframe.plot.<method>
    the methods can be
    box()
    scatter()
    hist()
    and probably more yeah
"""
#weird = plt.subplot()

#lookie u can assign specific colors to values look
#heres a dictionary that will map colors to values
#colordict = {"budgie": 'blue', "egg": "yellow", "mourning dove": "purple", "linnie" : "green"}

#this function creates a list of colors for each data point based on the categorical value of species
#colorlist = [colordict[spec] for spec in birdtable["species"]]

#and boooom we made a scatterplot
#weird = birdtable.plot.scatter("age", "weight (g)", c=colorlist)

#lets customize it more
#weird.set_xlabel("oldness")
#weird.set_ylabel("massness in gwams")
#weird.set_title("birt age vs weight scatterplawt")

#lets look at the coolness we can do with the legend
#so yea doing patches makes it a patch instead of just a lil dot
import matplotlib.patches as mpatches
# legend_handles = [
#     mpatches.Patch(color=colordict["budgie"], label = "bodji"),
#     mpatches.Patch(color=colordict["mourning dove"], label = "mornduv"),
#     mpatches.Patch(color=colordict["linnie"], label = "rini"),
#     mpatches.Patch(color=colordict["egg"], label = "ayg")    
# ]

#weird.legend(handles = legend_handles, loc='upper right')


"""
seaborn!!! powerful and pretty
wow seaborn is so cool

you can make all sorts of plots but lets use
boxplot()
scatterplot()
"""

import seaborn as sns

#strange = sns.scatterplot(data=birdtable, x="age", y="weight (g)", hue="species")
#strange.set(title="birt age vs weight based on speshees")


#cooool now lets do a boxplawt

ridiculous = sns.boxplot(data=birdtable, x="species", y="weight (g)")
ridiculous.set(title="weight distrubitasudn by speeshees")
plt.show()

#congrats!!


