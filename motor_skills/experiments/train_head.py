import motor_skills.cip.head.head as Head
import tkinter
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

def train_head(Xtrain, Ytrain, Xeval, Yeval):
	##########  Training and Evaluation ##########
	#TODO: We should implement a DataLoader here to make training more efficient, but that should be connected to data from RL loop of running cip, so we will implement it later

	#Head model
	head = Head.Head()

	if device == "cuda":
		head.to(device)

	#The logits of head will be unormalized and of the form [batch_size, class], so we use softmax to normalize
	softmax = nn.Softmax(dim=1)

	#Loss function
	loss = nn.CrossEntropyLoss()
	#Learning Rate
	lr = 0.0005
	#Optimizer
	optimizer = optim.Adam(head.parameters(),lr)

	#number of training episodes
	episodes = 3000
	#size of each minibatch in an episode #TODO currently assumes batch_size divides training data perfectly
	batch_size = 32

	#number of minibatches in each iteration
	n_batches = int(len(Xtrain) / batch_size)

	training_losses = []
	eval_losses = []

	#print every p episodes
	p = 100

	for e in range(episodes):
		#running training loss across minibatches
		running_loss = 0
		for b in range(n_batches):
			optimizer.zero_grad()

			#Grab minibatch
			local_x = Xtrain[b*batch_size:(b+1)*batch_size]
			local_x = torch.stack(local_x)

			if device == "cuda":
				local_x = local_x.to(device)

			local_y = Ytrain[b*batch_size:(b+1)*batch_size]
			local_y = torch.LongTensor(local_y)

			if device == "cuda":
				local_y = local_y.to(device)

			#forward!
			output = head(local_x)
			#print("output probabilities: ", softmax(output))
			l = loss(output, local_y)
			l.backward()
			optimizer.step()
			
			running_loss += l.item()

		av_train_loss = running_loss / float(n_batches)
		if e % p == 0:
			print("Episode %s av training loss: %s" % (e, av_train_loss))
		training_losses.append(av_train_loss)

		### Calculate av eval loss
		if device == "cuda":
			Xeval = Xeval.to(device)
			Yeval = Yeval.to(device)
		output = head(Xeval)
		l = loss(output, Yeval)
		if e % p == 0:
			print("Episode %s av eval loss: %s" % (e, l.item()))
		eval_losses.append(l.item())

	#Print F1 score of model on eval
	#print("Yeval: ", Yeval)
	detach_output = output.detach().numpy()
	model_max = np.argmax(detach_output, axis=1)

	model_f1 = f1_score(Yeval,model_max,average="weighted")
	print("F1 score for model:", model_f1)
	all_zero = [0 for _ in range(len(Yeval))]
	all_one = [1 for _ in range(len(Yeval))]
	all_random = [random.choice([0,1]) for _ in range(len(Yeval))]

	zero_f1 = f1_score(Yeval,all_zero,average="weighted")
	one_f1 = f1_score(Yeval,all_one,average="weighted")
	random_f1 = f1_score(Yeval,all_random,average="weighted")
	print("F1 score for all 0:", zero_f1)
	print("F1 score for all 1:", one_f1)
	print("F1 score for random:", random_f1)

	#sanity check
	#print("TRAINING outputs")
	#print(softmax(head(torch.stack(Xtrain))))
	#print("TRAINING real")
	#print(Ytrain)

	#print("EVAL outputs")
	#print(softmax(head(Xeval)))
	#print("EVAL real")
	#print(Yeval)

	return(training_losses,eval_losses, model_f1, zero_f1, one_f1, random_f1)

########## DATASET ##########
#Load in grasp poses from GPD
#GPD_POSES_PATH = "/home/eric/Github/motor_skills/motor_skills/envs/mj_jaco/assets/MjJacoDoorGrasps"
#GPD_POSES_PATH = "/home/eric/Github/motor_skills/motor_skills/experiments/gpd_data_dict"
GPD_POSES_PATH = "/home/eric/Github/motor_skills/motor_skills/planner/best.p"
#grasp_qs = pickle.load(open(GPD_POSES_PATH, "rb"))
with open(GPD_POSES_PATH, 'rb') as f:
	grasp_qs = pickle.load(f, encoding="latin1")
	#print(list(grasp_qs['xyz'][10]))
	#print(list(grasp_qs['quat'][10]))
	#print(list(grasp_qs['joint_pos'][10]))
	xyz_qs = grasp_qs['xyz']
	quat_qs = grasp_qs['quat']

	qs_zip = zip(xyz_qs,quat_qs)
	grasp_qs = [np.concatenate((qs[0],qs[1])) for qs in qs_zip]
	#grasp_qs = grasp_qs['joint_pos']

#Make full dataset
X = [torch.FloatTensor(grasp) for grasp in grasp_qs]
#Y = [0 for _ in X]
#Y = [1.000, 0.000, 1.000, 1.000, 0.000, 0.000, 1.000, 0.000, 0.000, 1.000, 0.000, 0.0000, 0.000, 0.000] #Precomputed success rates from replay.py for GPD_POSES_PATH
#Y = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
labels = [0, 7, 10, 52, 53, 57, 58, 60, 70, 71, 73, 90, 103, 118, 128, 135, 142, 143, 152, 162, 168, 176, 196, 199, 200, 205, 210, 215, 220, 223, 229, 230, 236, 237, 242, 246, 247, 250, 252, 256, 262, 263, 264, 267, 268, 269, 270, 271, 277, 278, 281, 287, 288, 299, 300, 301, 302, 304, 305, 306, 307, 309, 310, 316, 318, 319, 321, 325, 339, 340, 341, 345, 346, 349, 360, 363, 365, 366, 368, 369, 370, 371, 372, 379, 380, 382, 383, 384, 393, 395, 406, 414, 417, 423, 434, 436, 437, 440, 447, 452, 468, 469, 473, 477, 486, 487, 489, 497, 498, 499, 501, 503, 504, 519, 521, 527, 529, 539, 542, 544, 545, 548, 561, 572, 576, 579, 585, 588, 589, 593, 599, 602, 604, 605, 611, 612, 614, 615, 631, 634, 635, 645, 646, 647, 653, 661, 666, 672, 675, 676, 681, 684, 685, 687, 690, 698, 703, 706, 709, 710, 716, 717, 724, 725, 733, 744, 746, 747, 753, 757, 760, 762, 779, 781, 782, 790, 793, 798, 801, 806, 809, 810, 812, 813, 815, 816, 818, 820, 823, 824, 826, 827, 835, 836, 837, 841, 843, 845, 846, 847, 848, 850, 851, 856, 859, 860, 861, 862, 864, 865, 867, 873, 874, 876, 877, 881, 883, 887, 888, 889, 896, 897, 898, 900, 901, 905, 906, 911, 916, 919, 920, 922, 923, 925, 933, 934, 935, 941, 943, 945, 946, 951, 956, 959, 960, 962, 963, 967, 968, 969, 972, 973, 974, 980, 985, 986, 988, 989, 991, 992, 994, 995, 999, 1163, 1205, 1215, 1217, 1218, 1221, 1225, 1234, 1242, 1243, 1260, 1262, 1263, 1264, 1267, 1269, 1274, 1275, 1276, 1292, 1294, 1308, 1315, 1319, 1327, 1337, 1344, 1407, 1409, 1418, 1419, 1445, 1457, 1471, 1474, 1479, 1500, 1504, 1531, 1532, 1540, 1552, 1563, 1566, 1580, 1589, 1596, 1600, 1601, 1602, 1603, 1606, 1607, 1608, 1609, 1610, 1615, 1619, 1620, 1622, 1624, 1625, 1626, 1627, 1628, 1629, 1631, 1632, 1633, 1635, 1637, 1639, 1640, 1641, 1642, 1643, 1645, 1646, 1647, 1648, 1649, 1651, 1653, 1654, 1657, 1658, 1660, 1661, 1663, 1664, 1665, 1666, 1668, 1669, 1670, 1671, 1672, 1673, 1677, 1679, 1681, 1687, 1688, 1690, 1691, 1693, 1694, 1696, 1698, 1700, 1701, 1702, 1704, 1705, 1707, 1709, 1711, 1713, 1715, 1716, 1717, 1719, 1720, 1721, 1722, 1723, 1725, 1726, 1728, 1731, 1734, 1735, 1738, 1741, 1744, 1745, 1748, 1749, 1753, 1754, 1755, 1757, 1761, 1763, 1764, 1766, 1767, 1768, 1769, 1771, 1772, 1774, 1775, 1776, 1781, 1782, 1784, 1787, 1788, 1790, 1791, 1795, 1796, 1797, 1798, 1799, 1800, 1803, 1808, 1809, 1812, 1816, 1818, 1822, 1834, 1837, 1839, 1847, 1855, 1861, 1876, 1881, 1889, 1891, 1895, 1901, 1908, 1912, 1920, 1923, 1930, 1933, 1934, 1943, 1948, 1951, 1956, 1958, 1960, 1965, 1971, 1974, 1976, 1985, 1989, 1995, 1997, 2017, 2202, 2205, 2216, 2217, 2219, 2222, 2242, 2257, 2259, 2271, 2288, 2289, 2307, 2309, 2318, 2326, 2331, 2333, 2336, 2345, 2351, 2376, 2382, 2390, 2396, 2406, 2409, 2411, 2424, 2426, 2437, 2439, 2443, 2447, 2448, 2456, 2457, 2460, 2463, 2466, 2467, 2472, 2474, 2479, 2480, 2494, 2500, 2508, 2514, 2522, 2526, 2546, 2552, 2554, 2563, 2564, 2580, 2584, 2587, 2589, 2590, 2592, 2595, 2596, 2605, 2613, 2620, 2633, 2637, 2638, 2640, 2641, 2642, 2644, 2646, 2651, 2657, 2668, 2671, 2674, 2676, 2681, 2686, 2695, 2722, 2727, 2729, 2742, 2774, 2778, 2795, 2802, 2803, 2806, 2808, 2813, 2816, 2819, 2832, 2834, 2837, 2845, 2847, 2849, 2850, 2853, 2856, 2860, 2861, 2865, 2867, 2869, 2881, 2884, 2888, 2889, 2893, 2898, 2900, 2901, 2903, 2910, 2913, 2914, 2918, 2919, 2924, 2931, 2938, 2940, 2941, 2942, 2948, 2960, 2962, 2964, 2965, 2972, 2973, 2976, 2980, 2981, 2997, 2998, 3225, 3276, 3382]
Y = [1 if i in labels  else 0 for i in range(17*200)]
########### CLASS BALANCING ############
bb = zip(X,Y)
#bbb = [b for b in bb if b[1] == 1 or random.choice([True,False,False,False,False,False,False])]
bbb = [b for b in bb if b[1] == 1 or random.choice([True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False])]
XX = []
YY = []
for b in bbb:
	d,e = b
	XX.append(d)
	YY.append(e)
X = XX
Y = YY

percent_one = sum(Y) / float(len(Y))
print("LEN OF THE DATASET IS:", len(X))
######### END CLASS BALANCING #########

#shuffle dataset
c = list(zip(X,Y))
random.shuffle(c)
X, Y = zip(*c)

#Kfold-cross validation
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits)
kf.get_n_splits(X)
#print(kf)

#record train losses and eval losses for all n splits
k_train_losses = []
k_eval_losses = []
#record f1 scores for model, all0, all1, random
f1_model_l = []
f1_zero_l = []
f1_one_l = []
f1_random_l = []
for train_index, test_index in kf.split(X,Y):
	print("NEW K FOLD")
	#print("TRAIN:", train_index, "TEST:", test_index)
	Xtrain = [X[i] for i in train_index]
	Ytrain = [Y[i] for i in train_index]
	Xeval = [X[i] for i in test_index]
	Yeval = [Y[i] for i in test_index]

	#train_head expects Xeval and Yeval to be tensors instead of lists like Xtrain and Ytrain
	Xeval = torch.stack(Xeval)
	Yeval = torch.LongTensor(Yeval)

	train_loss, eval_loss, model_f1, zero_f1, one_f1, random_f1 = train_head(Xtrain,Ytrain,Xeval,Yeval)

	k_train_losses.append(train_loss)
	k_eval_losses.append(eval_loss)
	f1_model_l.append(model_f1)
	f1_zero_l.append(zero_f1)
	f1_one_l.append(one_f1)
	f1_random_l.append(random_f1)

print("Average F1 score for model: ", np.average(f1_model_l))
print("Average F1 score for 0s: ", np.average(f1_zero_l))
print("Average F1 score for 1s: ", np.average(f1_one_l))
print("Average F1 score for randoms: ", np.average(f1_random_l))

#Average and stardard deviations across n splits
av_train_losses = [np.average(los) for los in zip(*k_train_losses)]
av_eval_losses = [np.average(los) for los in zip(*k_eval_losses)]
std_train_losses = [np.std(los) for los in zip(*k_train_losses)]
std_eval_losses = [np.std(los) for los in zip(*k_eval_losses)]

#Only visualize some number of the losses
viz_av_train = []
viz_std_train = []
viz_av_eval = []
viz_std_eval = []
for i in range(len(av_train_losses)):
	if i % 100 == 0:
		viz_av_train.append(av_train_losses[i])
		viz_std_train.append(std_train_losses[i])
		viz_av_eval.append(av_eval_losses[i])
		viz_std_eval.append(std_eval_losses[i])

#Visualize training and eval losess
plt.errorbar(range(len(viz_av_train)),viz_av_train,yerr=viz_std_train,fmt='r')
plt.errorbar(range(len(viz_av_eval)),viz_av_eval,yerr=viz_std_eval,fmt='b')
plt.show()
