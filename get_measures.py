print("loading packages...")
import pandas as pd
import Measures
print("Done!")

print("Reading data...")
data = pd.read_csv("./Data/performances.csv", index_col=False)
print("Done!")
M = Measures.Measuring(data, Num_Loc=8, TOLERANCIA=0)
data = M.get_measures('123')
outputFile = "./Data/humans_only_absent.csv"
data.to_csv(outputFile, index=False)
print("Results saved to " + outputFile)
