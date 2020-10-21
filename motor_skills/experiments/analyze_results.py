import pickle
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np

#Load all the relevant log files
mypath = "./log_results"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

logs = {"naive": [], "cip": []}
for f in onlyfiles:
	log_file = pickle.load(open(mypath+"/"+f,"rb"))
	if "naive" in f:
		logs["naive"].append(log_file)
	else:
		logs["cip"].append(log_file)

#evaluate success rates over time
cip_succ = []
naive_succ = []
for c in logs["cip"]:
    cip_succ.append(c["success_rate"])
for c in logs["naive"]:
    naive_succ.append(c["success_rate"])

av_cip_succ = [np.average(s) for s in zip(*cip_succ)]
av_naive_succ = [np.average(s) for s in zip(*naive_succ)]
std_cip_succ = [np.std(s) for s in zip(*cip_succ)]
std_naive_succ = [np.std(s) for s in zip(*naive_succ)]

plt.errorbar(range(len(av_cip_succ)),av_cip_succ,yerr=std_cip_succ,fmt='r')
plt.errorbar(range(len(av_naive_succ)),av_naive_succ,yerr=std_naive_succ,fmt='b')
plt.show()

#evaluate CDE over time
cip_succ = []
naive_succ = []
for c in logs["cip"]:
    cip_succ.append(c["stoc_pol_mean"])
for c in logs["naive"]:
    naive_succ.append(c["stoc_pol_mean"])

av_cip_succ = [np.average(s) for s in zip(*cip_succ)]
av_naive_succ = [np.average(s) for s in zip(*naive_succ)]
std_cip_succ = [np.std(s) for s in zip(*cip_succ)]
std_naive_succ = [np.std(s) for s in zip(*naive_succ)]

plt.errorbar(range(len(av_cip_succ)),av_cip_succ,yerr=std_cip_succ,fmt='r')
plt.errorbar(range(len(av_naive_succ)),av_naive_succ,yerr=std_naive_succ,fmt='b')
plt.show()
