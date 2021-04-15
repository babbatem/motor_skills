import numpy as np
import random

result_labels = [0, 1, 1, 0, 2, 2, 1, 3, 0, 1, 3, 3,3]
result_time = [random.random() for i in result_labels]
label_types = list(np.unique(result_labels))
label_counts = [(l, result_labels.count(l)) for l in label_types]
for label_type in label_types:
	indices = np.where(np.array(result_labels) == label_type)[0]
	print(label_type, indices, [result_time[ind] for ind in indices])
