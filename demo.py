"""
this script shows what we mean by sorting data samples
"""


# import the huggingface datasets library
# pip install datasets
from datasets import load_dataset
# pip install sentence_transformers
from sentence_transformers import SentenceTransformer

from sorters import distance_to_mean

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

ds = load_dataset("trec")
orig_train = list(ds['train'])
orig_test = list(ds['test'])

all_data = orig_train + orig_test # disregard original train/test split

data_map = {}
for item in all_data:
    if item['label-coarse'] not in data_map:
        data_map[item['label-coarse']] = []
    data_map[item['label-coarse']].append(item['text'])

new_train = []
new_test = []
for (label, texts) in data_map.items():
    print('working on label #{}'.format(label))
    print('\t({} samples)'.format(len(texts)))
    embeddings = embedder.encode(texts)
    distances = distance_to_mean(embeddings)
    _texts = [{'text': texts[i], 'label-coarse': label} for (i, _) in distances]
    th = int(len(_texts) * 0.7)
    _train = _texts[:th]
    _test = _texts[th:]
    new_train.append(_train)
    new_test.append(_test)
