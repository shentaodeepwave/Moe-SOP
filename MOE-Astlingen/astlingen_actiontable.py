import pandas as pd
import numpy as np


table=[]
t2=[0.0,0.1075,0.2366,1.0]
t3=[0.0,0.3159,0.6318,1.0]
t4=[0.0,0.1894,0.3523,1.0]
t6=[0.0,0.1687,0.4303,1.0]
for i1 in t2:

    for i2 in t3:

        for i3 in t4:

            for i4 in t6:
                tem = [i1,i2,i3,i4]

                table.append(tem)

                            

table = np.array(table)

pd.DataFrame(table).to_csv('DQN_action_table.csv')