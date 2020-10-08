import motor_skills.cip.head.head as Head
import tkinter
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
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
	lr = 0.00005
	#Optimizer
	optimizer = optim.Adam(head.parameters(),lr)

	#number of training episodes
	episodes = 1000
	#size of each minibatch in an episode #TODO currently assumes batch_size divides training data perfectly
	batch_size = 10

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
GPD_POSES_PATH = "/home/eric/Github/motor_skills/motor_skills/experiments/gpd_data_dict"
#grasp_qs = pickle.load(open(GPD_POSES_PATH, "rb"))
with open(GPD_POSES_PATH, 'rb') as f:
	grasp_qs = pickle.load(f, encoding="latin1")['joint_pos']

#Make full dataset
X = [torch.FloatTensor(grasp) for grasp in grasp_qs]
#Y = [0 for _ in X]
Y = [1.000, 0.000, 1.000, 1.000, 0.000, 0.000, 1.000, 0.000, 0.000, 1.000, 0.000, 0.0000, 0.000, 0.000] #Precomputed success rates from replay.py for GPD_POSES_PATH
Y = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
########### CLASS BALANCING ############
bb = zip(X,Y)
bbb = [b for b in bb if b[1] == 1 or random.choice([True,False,False,False,False,False,False])]
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
kf = KFold(n_splits=n_splits)
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
for train_index, test_index in kf.split(X):
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
