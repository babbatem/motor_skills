import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sns.set_palette("colorblind")

def get_data(rootDir):
    scores = {}
    runs=[]
    for dirName, subdirList, fileList in os.walk(rootDir):
        print('Found directory: %s' % dirName)
        for fname in fileList:
            desired_fname = 'log.csv'
            if fname == desired_fname:
                path = os.path.join(dirName,fname)
                log_df = pd.read_csv(path)
                log_df['tag']=path
                log_df['episode']=np.arange(len(log_df))*200
                print(len(log_df))
                runs.append(log_df)

                scores[path]=log_df['eval_score'].tolist()[-1]

    data = pd.concat(runs)
    return data, scores


data, scores=get_data('exps/starts/')
print(data)

for k in sorted(scores.keys(), key=lambda x: scores[x]):
    print(k, scores[k])
my_relplot = sns.relplot(x='episode',
                         y='eval_score',
                         kind='line',
                         data=data,
                         # height=4,
                         alpha=0.4,
                         hue='tag',
                         # col_wrap=2,
                         # legend=False,)
                         )

# plt.xlabel('iteration (10 rollouts per batch)')
# os.makedirs('data/plots/2020-01-15-latenite', exist_ok=True)
# f='data/plots/2020-01-15-latenite/eval.svg'
# plt.savefig(f)
plt.show()