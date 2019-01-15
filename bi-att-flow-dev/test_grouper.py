from my.tensorflow import grouper
import random
import itertools

idxs=random.sample(range(5),5)
print(idxs)
grouped = list(grouper(idxs, 2))
print (grouped)
random_grouped = lambda :random.sample(grouped,3)
print(random_grouped)
'''
batch_idx_tuples = itertools.chain.from_iterable(random_grouped() for _ in range(5))
for _ in range(5):
    batch_idxs = tuple(i for i in next(batch_idx_tuples) if i is not None)
    print("BATCH",batch_idxs)
'''
a=[[1,2],[3,4],[5,6]]
grouped1 = list(grouper(a,2))
print ("NEW",grouped1)