import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\DELL\Desktop\ML Files\Market_Basket_Optimisation.csv", header=None)



record=[]
for i in range(0, 7501):
        record.append([str(data.values[i,j]) for j in range(0,20)])

#train data and get results
from apyori import apriori
rule = apriori(record, min_supports=0.04,min_length=2, min_confidence=0.02, min_lift=3)
print(rule)
####################################################################