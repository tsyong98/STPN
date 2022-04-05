import numpy as np

'''
max_win = 10
If time t is predicted as 1, extend the predction 1 to t+1 ~ t+10.
[i.e: NEXT 10 prediction extended as ones (the 10 does not include current t)] 
Same goes for min_win
'''

def quad_win(min_win=5, max_win=30, data_len=8640):
	'''
	min_pt = (m,n), max_pt = (k,q)
	a = (n-q)/(m^2-k*m)
	b = -a*k
	c = q
	'''
	k = data_len-1
	q = max_win
	m = k/2
	n = min_win

	a = (n-q)/(m**2-k*m)
	b = -a*k
	c = q

	t = np.arange(data_len)
	win_lim = np.round(a*(t**2) + b*t + c).astype(int)
	return win_lim

def V_win(min_win=5, max_win=30, data_len=8640):
	'''
	max win - min win find y_intercept
	define negative graph
	absolute it
	shift it back by min win
	'''
	y_in = max_win-min_win
	x_in = data_len/2
	grad = -y_in/x_in
	t = np.arange(data_len)
	win_lim = np.round(abs(grad*t + y_in) + min_win).astype(int)
	return win_lim

def cos_win(min_win=5, max_win=30, data_len=8640):
	'''
	define t len within 2*pi
	cos graph with desired win range
	shift 
	'''
	win_range = max_win - min_win
	t = np.linspace(0, 2*np.pi, data_len)
	win_lim = np.round((win_range/2)*np.cos(t) + (win_range/2) + min_win).astype(int)
	return win_lim


win_style = {"quad": quad_win, "V": V_win, "cos": cos_win}


def gen_win(style="cos", min_win=5, max_win=30, data_len=8640):
	# Generate window with specified style
	win_lim = win_style[style](min_win, max_win, data_len)
	return win_lim


def win_fill(data, ffill_time_win, bfill_time_win=None)
	'''
	Input: DataFrame with specified column
	bfill_time_win is same as ffill_time_win by default / if not provided
	'''
	ones_idx = np.argwhere(data.values == 1)
	ones_idx = ones_idx.reshape((len(ones_idx),))

	if bfill_time_win == None:
		bfill_time_win = ffill_time_win

	for idx in ones_idx:
		# Timewindow ffill
		data.iloc[idx:idx+ffill_time_win[idx]+1] = 1
		# Timewindow bfill
		data.iloc[idx-bfill_time_win[idx]:idx] = 1


def time_encode(time_data):
	'''
	Takes in Pandas Series or single column timestamp DataFrame
	For example: time_data = data['timestamp']

	============ Encoding option 2 =============
	seconds_in_day = 24*60*60
	df['sin_time'] = np.sin(2*np.pi*df.seconds/seconds_in_day)
	df['cos_time'] = np.cos(2*np.pi*df.seconds/seconds_in_day)
	df.drop('seconds', axis=1, inplace=True)
	'''
	time_data = pd.to_datetime(time_data)
	hour = time_data.dt.hour
	minute = time_data.dt.minute
	second = time_data.dt.second

	embedded_time = pd.DataFrame()
	embedded_time["sin_time"] = np.sin(2 * np.pi * ((hour + (minute + second / 60) / 60) / 24))
	embedded_time["cos_time"] = np.cos(2 * np.pi * ((hour + (minute + second / 60) / 60) / 24))

	return embedded_time



__all__ = [
"gen_win",
"win_fill",
"time_encode"
]

