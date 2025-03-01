from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

# import nltk
# nltk.download('vader_lexicon')

example = "This oatmeal is not good. Its mushy, soft, I don't like it. Quaker Oats is the way to go."


sia = SentimentIntensityAnalyzer()

print(sia.polarity_scores('I am so Happy'))
print(sia.polarity_scores('I am not so Happy'))
print(sia.polarity_scores('This is the worst thing ever'))
print(sia.polarity_scores(example))