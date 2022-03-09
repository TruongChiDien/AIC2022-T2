import os
import json
import pickle
import numpy as np

# os.system('python3 test.py --config configs/motion_SE_NOCLS_nonlpaug_288.yaml')
# os.system('python3 scripts/get_submit.py')

with open('results_mergefinal.json') as f:
    res = json.load(f)

queries = np.array(list(res.keys()))
ans = np.array(list(res.values()))

mrr = 0
for query, an in zip(queries, ans):
    mrr += 1/(np.where(an == query)[0] + 1)

print('MRR = {}'.format(mrr/len(queries)))