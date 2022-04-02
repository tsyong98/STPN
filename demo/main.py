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

# Use datatest.csv temperature and occ as example

# Check target trimming with depth and tau

# Check if the code works with depth and tau = 0

import pandas as pd
# from STPN.util import *
from util import *

# Define parameters
depth = 2
tau = 1

# Load data
data = pd.read_csv("datatest.csv",index_col=0)

# Temperature	Humidity	Light	CO2	HumidityRatio	Occupancy

# Generate boundaries
boundaries = discretize(data['Temperature'].values, mode="uniform", numparts=5)
# print(boundaries)

# Visualize boundaries
# visualize_bounds(data['Temperature'].values, boundaries)

# Symbolize data using defined boundaries
symbolized_data = symbolize(data['Temperature'].values, boundaries)

# Generate state(symbolized data with defined depth) using symbolized data
state = state_gen(symbolized_data, depth=depth, tau=tau)
# print(state)

# (optional) Re-define state's element to single "symbol" representation
# Function coming soon


# Define prediction target
occupancy = data['Occupancy'].values # Classification (Binary)
# Use symbolized_data(temperature) # Regression (Multi-class Classification)

# Trim state with depth (see figure for clarification)
occupancy = occupancy[depth+tau:]
temperature = symbolized_data[depth+tau:]

assert len(temperature)==(len(data['Temperature'].values)-depth-tau)
assert len(occupancy)==(len(data['Temperature'].values)-depth-tau)

assert len(temperature)==len(state)
assert len(occupancy)==len(state)

# Construct state-occupancy Trans. Mat.
TM_binary, TM_binary_c, binary_targets = compute_TM(state, occupancy, True)

# Construct state-temperature Trans. Mat.
TM_multi, TM_multi_c, multi_targets = compute_TM(state, temperature, True)


# Inferencing using state-occupancy Trans. Mat.
occ_pred = inference(state, TM_binary, binary_targets, mode="ffill", verbose=1)

# Inferencing using state-temperature Trans. Mat.
temp_pred = inference(state, TM_multi, multi_targets, mode="ffill", verbose=1)

# Evaluate inference results
# occ_results = eval(occupancy, occ_pred, binary_targets, save_resu=False)
# plot_confusion_matrix(occ_results['cm'],binary_targets)

temp_results = eval(temperature, temp_pred, multi_targets, save_resu=False, print_cls_acc=False)
plot_confusion_matrix(temp_results['cm'],multi_targets)




