# import STPN
# from STPN import STPN

'''
Fix library import problem
Demonstrate the use of the functions

- Include the whole process (follow HPS or WHISPER workflow)

'''

'''
# Window fill
# Test import
from STPN.time_func import *
# from STPN.STPN.time_func import *

import pandas as pd
import numpy as np
data = np.zeros((8640,))
data = pd.DataFrame(data)
data.columns = ["occupied"]


# Insert some ones
data["occupied"].iloc[15] = 1
data["occupied"].iloc[4320] = 1
data["occupied"].iloc[-15] = 1


time_win = gen_win("cos", 5, 10, data_len=len(data["occupied"]))
data["occupied"] = win_fill(data["occupied"], time_win)
'''
# ==========================

'''
import pandas as pd
from STPN.time_func import *

# load demo.csv
data = pd.read_csv("demo.csv",index_col=0)
data['timestamp'] = pd.to_datetime(data.index)

# OR
# data = pd.read_csv("demo.csv")
# data['timestamp'] = pd.to_datetime(data["timestamp"])

embedded_time = time_encode(data['timestamp'])

# Visualize embedded time
import matplotlib.pyplot as plt
plt.plot(embedded_time["sin_time"],embedded_time["cos_time"])
plt.plot(embedded_time["sin_time"])
plt.plot(embedded_time["cos_time"])
'''

# ==========================
import pandas as pd
from STPN.util import *

# load timestamped_illum1.csv
data = pd.read_csv("timestamped_illum1.csv",index_col=0)

boundaries = discretize(data['illum'].values, mode="uniform", numparts=5)
print(boundaries)

visualize_bounds(data['illum'].values, boundaries)