import pandas as pd
from Granger import Granger

df = pd.read_csv("Data/PRICE_AND_DEMAND_202209_NSW1.csv")
print(df)
Gr = Granger(df)
Gr.varCalculation()
