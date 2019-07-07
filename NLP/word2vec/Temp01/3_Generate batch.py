import urllib.request
import zipfile
import os
import numpy as np


#########################get the data###########################
WORDS_PATH = "datasets/words"
WORDS_URL = 'http://mattmahoney.net/dc/text8.zip'

def fetch_words_data(words_url=WORDS_URL, words_path=WORDS_PATH):
    os.makedirs(words_path, exist_ok=True)
    zip_path = os.path.join(words_path, "words.zip")
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(words_url, zip_path)
    with zipfile.ZipFile(zip_path) as f:
        data = f.read(f.namelist()[0])
    return data.decode("ascii").split()

words = fetch_words_data()
print(words[:5])
#############################################################
from collections import Counter
vocb_size = 50000
vocabulary = [("UNK", None)] + Counter(words).most_common(vocb_size - 1)
vocabulary = np.array([w for w, _ in vocabulary])
word_to_idx = {w: id for id, w in enumerate(vocabulary)}
data = np.array([word_to_idx.get(w, 0) for w in words])

test1 = " ".join(words[:9]), data[:9]
print(test1)
#('anarchism originated as a term of abuse first used',
# array([5234, 3081,   12,    6,  195,    2, 3134,   46,   59]))

print(len(data), len(word_to_idx))
#17005207 50000

test2 = " ".join([vocabulary[word_index] for word_index in [5241, 3081, 12, 6, 195, 2, 3134, 46, 59]])
print(test2)
#'cycles originated as a term of abuse first used'

print(words[24], data[24])
#('culottes', 0)
#############################################################

import random
from collections import deque

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

data_index=0
batch, labels = generate_batch(8, 2, 1)
print(batch, [vocabulary[idx] for idx in batch])
#[3081 3081   12   12    6    6  195  195] ['originated', 'originated', 'as', 'as', 'a', 'a', 'term', 'term']

print(labels, [vocabulary[idx] for idx in labels[:, 0]])
#[[5234]
# [  12]
# [3081]
# [   6]
# [ 195]
# [  12]
# [   6]
# [   2]] ['anarchism', 'as', 'originated', 'a', 'term', 'as', 'a', 'of']
