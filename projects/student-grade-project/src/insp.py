import matplotlib.pyplot as plt
import pickle
import pandas as pd

pick= open ("../model/student.pickle","rb")
model=pickle.load(pick)
data=pd.read_csv("../data/student-mat.csv",sep=";")
data=data[["age","absences","G1","G2","G3"]]
plt.scatter(data["G1"],data["G3"])
plt.show()