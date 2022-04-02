import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from copy import deepcopy
import pandas as pd
import itertools

import difflib # get close matches for states(strings)

'''
Compute transition matrix for multiple output symbol (currently works for binary only)

Make a diagram for state gen (mainly to make clear the timeseamp used)

Include a test folder with data and sample pipeline for this library

Implement uniform boundaries

Prallellize thresh_opt()
- and symbolize()?

'''

def __MEP(data, numparts):
	num_per_part = np.round(len(data)/numparts)
	data = np.sort(data)
	boundaries = []
	for i in range(1, numparts): # number of boundary line = numparts - 1
		idx = int(num_per_part*i)
		bound = np.mean([data[idx-1],data[idx]])
		boundaries.append(bound)
	return np.unique(boundaries)


def __unique_boundaries(data):
	unique = np.unique(data)
	boundaries = unique + 1e-8
	boundaries = boundaries[:-1]
	if len(boundaries) > 20:
		print(f"Warning: There are {len(boundaries)} boundaries.")
	return boundaries


def __uniform_boundaries(data, numparts):
	max_ = np.max(data)
	min_ = np.min(data)
	range_ = max_ - min_
	interval = range_/numparts
	boundaries = np.arange(1,numparts)*interval + min_
	return boundaries


def discretize(data, mode="MEP", numparts=2):
	# Acceptable modes
	modes = ["uniform", "MEP", "unique"]
	assert mode in modes, f"mode should be one of {modes}"
	assert numparts>1, f"numparts should be >= 2"

	if mode == "uniform":
		boundaries = __uniform_boundaries(data, numparts)
	elif mode == "MEP":
		boundaries = __MEP(data, numparts)
	elif mode == "unique":
		boundaries = __unique_boundaries(data)

	return boundaries


def visualize_bounds(data, bounds):

	plt.figure()
	# plt.plot(data,'x')
	plt.plot(data)
	for j in bounds:
		plt.axhline(y=j, color='r', linestyle='-')
	
	# plt.title("")
	# plt.savefig('_partition.png')


def visualize_TM(TM, TM_title, show_prob=False):
	fig, ax = plt.subplots()
	plot = ax.imshow(TM, cmap=plt.cm.Blues)
	if show_prob:
		for i in range(np.shape(TM)[1]):
			for j in range(np.shape(TM)[0]):
				c = TM.values[j,i]
				ax.text(i, j, "%.2f"%(c), va='center', ha='center')
	plt.title(TM_title)
	plt.xlabel("state (t)")
	plt.ylabel("sym (t+1)")
	fig.colorbar(plot)


def visualize_pred(pred, truth):

	plt.figure()
	# plt.plot(data,'x')
	plt.plot(data,'x-')
	for j in bounds:
		plt.axhline(y=j, color='r', linestyle='-')
	
	# plt.title("")


def symbolize(data, boundaries):

	data = np.asarray(data)
	placeholder_len = len(str(len(boundaries)+1))
	sym_data = np.asarray(["0"*placeholder_len for _ in range(len(data))])

	if len(boundaries) == 1:
		sym_data[data <= boundaries[0]] = "1"
		sym_data[data > boundaries[0]] = "2"
	else:
		# deal with middle parts first, then lastly the lowest and highest portion
		for i in range(len(boundaries)-1):
			sym_data[((data > boundaries[i]) & (data <= boundaries[i+1]))] = str(i+2) # one-index symbols, but start with "2" for first symbol of middle portion
			
		# lowest and highest portion
		sym_data[data <= boundaries[0]] = "1" # lower portion
		sym_data[data > boundaries[-1]] = str(len(boundaries)+1) # upper portion

	return sym_data


def state_gen(sym_data, depth, tau, verbosity=0):
	'''
	Num sym in each state = depth + 1
	'''
	depth_sym_data = []

	# Depth moving window
	for i in range(len(sym_data)-depth-tau): # 自动去尾 (ie: the last prediction will still be in the timeframe of the input data!)
		if verbosity > 0:
			print('Current depth window starting index:', i)
		depth_sym_data.append(','.join(sym_data[i:i+depth+1]))

	return depth_sym_data


def compute_TM(state_data, target_data, return_counts=False):
	'''
	List unique state and targest
	for each unique state:
		find targets for each state
		count unique targets
		fill in the TM with probabilities and counts
	'''
	state_data = np.asarray(state_data)
	target_data = np.asarray(target_data)

	# Initialize Trans. Mat. Placeholder
	TM_p = {} # Probabilities Dict.
	if return_counts:
		TM_c = {} # Counts Dict.

	# Find unique states and targets
	states = np.unique(state_data) # Auto sorts the states in ascending order
	targets = np.unique(target_data)

	for state in states:
		TM_p[state] = [0.0 for _ in range(len(targets))] # Probabilities Placeholder
		if return_counts:
			TM_c[state] = [0.0 for _ in range(len(targets))] # Counts Placeholder
		idx = np.argwhere(state_data == state).flatten() # Idx for current state
		temp_target = target_data[idx] # Selected target list for current state
		unique_temp_target, counts = np.unique(temp_target,return_counts=True) # Unique target for current state's targets

		for (u,c) in zip(unique_temp_target,counts):
			idx = np.argwhere(targets == u).flatten()[0] # return idx of where this count should be added
			TM_p[state][idx] = c # Counts
			if return_counts:
				TM_c[state][idx] = c
		TM_p[state] = TM_p[state]/np.sum(TM_p[state]) # Compute probabilities

	if return_counts:
		return TM_p, TM_c, targets # Return probability, counts and targets
	else:
		return TM_p, targets # Return probability and targets


def __get_close_match(state, existing_states):
	# Match starting from the first symbol to the last
	
	# To replace difflib.get_close_matches()?

	# Example debatable case:
	# If state ['1,2,3'] DNE, 
	# difflib.get_close_matches() gives ['2,2,3'], 
	# when ['1,2,2'] is available

	close_match = 0
	return close_match


def inference(states, TM, targets, mode="ffill",verbose=0):
	'''
	==== mode ====
	"ffill": forward fill the prediction if state DNE
	"close": use closest/most similar state for pred. if state DNE

	==== verbose ====
	0: suppress printing
	1: print prediction details when state DNE
	'''
	pred = []
	for s, state in enumerate(states):
		if state in TM.keys():
			idx = np.argmax(TM[state])
			pred.append(int(targets[idx]))
		else:
			if mode == "ffill":
				if len(pred) > 0:
					if verbose == 1:
						print(f"State {state} DNE, forwarding previous prediction.")
					pred.append(pred[-1]) # If state doesn't exist, take previous states's pred
				else: # If pred. is empty, ie: no previous pred.
					print(f"FIRST state DNE, ffill not possible, using mode=='close' for this state instead...")
					close_match = difflib.get_close_matches(state, TM.keys(), 1, 0)[0]
					if verbose == 1:
						print(f"Using close match: {close_match}")
					idx = np.argmax(TM[close_match])
					pred.append(int(targets[idx]))
			elif mode == "close":
				close_match = difflib.get_close_matches(state, TM.keys(), 1, 0)[0]
				print(close_match)
				if verbose == 1:
					print(f"State '{state}' DNE, using close match: {close_match}")

				idx = np.argmax(TM[close_match])
				pred.append(int(targets[idx]))
	return np.asarray(pred)

# For binary only
def thresh_opt(occ_prob, true_occ):

	threshold = np.unique(occ_prob) + 1e-8 # multiple thresholds
	temp_prediction = np.asarray(occ_prob)
	thresh_acc = []

	for thresh in threshold:
		temp_prediction_thresh = deepcopy(temp_prediction)
		
		index_1 = np.argwhere(temp_prediction > thresh)
		index_0 = np.argwhere(temp_prediction < thresh)
		
		temp_prediction_thresh[index_1] = 1
		temp_prediction_thresh[index_0] = 0

		temp = np.sum(abs(temp_prediction_thresh - true_occ))
		thresh_acc.append(1-(temp/len(true_occ)))

	# print best threshold:
	best_acc = np.amax(thresh_acc)
	print("max acc: %.4f"%(best_acc))
	best_acc_index = np.argwhere(np.asarray(thresh_acc) == best_acc)
	print("best_acc_index: ",best_acc_index[0][0])
	best_threshold = threshold[best_acc_index]
	best_threshold = best_threshold[0][0]
	print("best threshold: ",best_threshold)

	return best_threshold


def prob_thresh(occ_prob, thresh):

	occ_prob = np.asarray(occ_prob) # interested prediction prob. series
	pred = deepcopy(occ_prob) # placeholder for thresholded prediction
	
	index_1 = np.argwhere(occ_prob > thresh)
	index_0 = np.argwhere(occ_prob < thresh)
	
	pred[index_1] = 1
	pred[index_0] = 0

	return pred


def __class_acc(cm, target_names):
	assert len(cm)==len(target_names)

	TP = np.diagonal(cm) # True positive
	samples_per_class = np.sum(cm,axis=1) # Total sample in each class
	acc = [tp/s for tp,s in zip(TP,samples_per_class)]	

	class_acc = {}
	for i in range(len(TP)):
		class_acc[target_names[i]] = acc[i]

	return class_acc


def eval(truth, pred, target_names, save_resu=False, print_cls_acc=False):
	'''
	Returns dictionary containing:
	- class accuracy
	- confusion matrix
	- classification_report(clf_rep)
		- precision
		- recall
		- F1-score
		- overall accuracy
		- macro average
		- weighted average
	
	if binary target class, also returns:
	- tn_fp_fn_tp

	'''
	truth = np.array(truth).astype(int)
	pred = np.array(pred).astype(int)
	target_names = [str(i) for i in target_names]

	print('='*55)

	cm = confusion_matrix(truth,pred)
	cm = np.asarray(cm)

	class_acc = __class_acc(cm, target_names)

	if len(target_names) == 2:
		tn_fp_fn_tp = confusion_matrix(truth,pred).ravel() # Deepcopy cm instead?
		tn_fp_fn_tp = np.asarray(tn_fp_fn_tp)

	# precision, recall, f1-score, support(num samples)
	clf_rep = classification_report(truth,pred, target_names=target_names, digits=6) # For printing
	print(clf_rep)
	if print_cls_acc:
		print("Class Accuracies:",class_acc)
	print('='*55)
	print('\n')

	clf_rep = classification_report(truth,pred, target_names=target_names,output_dict=True) # To be output as dict

	# ================ Save Performances ================
	performance = {}
	performance['class_acc'] = class_acc
	performance['cm'] = cm
	performance['clf_rep'] = clf_rep
	if len(target_names) == 2:
		performance['tn_fp_fn_tp'] = tn_fp_fn_tp

	if save_resu:
		np.save('resu.npy',performance)
	return performance

# Plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, save_fig=False):

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
			horizontalalignment="center",
			color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('Ground Truth')
	plt.xlabel('Predicted')
	plt.show()
	if save_fig:
		plt.savefig('CM.png')



# Complete this
__all__ = [
"discretize",
"visualize_bounds",
"symbolize",
"state_gen",
"compute_TM",
"inference",
"eval",
"plot_confusion_matrix"
]

