import numpy as np
import pandas as pd

data = pd.read_csv("evaluation.csv")

losses = np.array(map(float, data['losses'].values))

print(losses)

'''
with open('evaluation.csv', 'rt') as f:
    reader = csv.reader(f)
    for row in reader:
        print()
        #accuracy = float(row[1])
        #print(np.fromstring(row[3]))

    f.close()
'''