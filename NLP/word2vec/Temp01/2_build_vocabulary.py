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
# 构建word2id词表
vocb_size = 50000
vocabulary = [("UNK", None)] + Counter(words).most_common(vocb_size - 1)
# 转化为ndarray数组结构
vocabulary = np.array([w for w, _ in vocabulary])
word_to_idx = {w: id for id, w in enumerate(vocabulary)}
# 进行word2index 编码，对于未出现在词表中的采用默认值default=0（"UNK"）
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

