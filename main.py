import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

example = "This oatmeal is not good. Its mushy, soft, I don't like it. Quaker Oats is the way to go."

token = nltk.word_tokenize(example)
print(token[:10])

tagged = nltk.pos_tag(token)
print(tagged[:10])

entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

