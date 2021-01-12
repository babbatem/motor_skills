import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import simplejson as json

sns.set_palette("colorblind")

def get_data(rootDir):
    scores = {}
    runs=[]
    for dirName, subdirList, fileList in os.walk(rootDir):
        print('Found directory: %s' % dirName)
        for fname in fileList:
            desired_fname = 'log.csv'
            if fname == desired_fname:

                # if we find a log, read into dataframe
                path = os.path.join(dirName,fname)
                log_df = pd.read_csv(path)
                log_df['tag']=path
                log_df['episode']=np.arange(len(log_df))*100

                # process the job config
                config_path = os.path.join(dirName, '..', 'job_config.json')
                with open(config_path, 'r') as f:
                    job_data = json.loads(f.read())

                log_df['seed'] = job_data['seed']
                log_df['start'] = str(job_data['env_kwargs']['start_idx'])
                log_df['wrist_sensor'] = str(job_data['env_kwargs']['wrist_sensor'])
                log_df['sensor_type'] = str(job_data['env_kwargs']['sensor_type'])

                runs.append(log_df)

                scores[path]=log_df['eval_score'].tolist()[-1]

    data = pd.concat(runs)
    return data, scores


data, scores=get_data('exps/tactile2/')
print(data)

for k in sorted(scores.keys(), key=lambda x: scores[x]):
    print(k, scores[k])
# my_relplot = sns.relplot(x='episode',
#                          y='eval_score',
#                          kind='line',
#                          data=data,
#                          # height=4,
#                          alpha=0.4,
#                          hue='start',
#                          # col_wrap=2,
#                          # legend=False,)
#                          )
my_relplot = sns.relplot(x='episode',
                         y='eval_score',
                         kind='line',
                         data=data,
                         # height=4,
                         alpha=0.4,
                         hue='sensor_type',
                         style='wrist_sensor'
                         # col_wrap=2,
                         # legend=False,)
                         )

# plt.xlabel('iteration (10 rollouts per batch)')
# os.makedirs('data/plots/2020-01-15-latenite', exist_ok=True)
# f='data/plots/2020-01-15-latenite/eval.svg'
# plt.savefig(f)
plt.show()
