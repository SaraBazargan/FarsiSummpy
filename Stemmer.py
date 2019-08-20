# Deriving Words Stems
from nltk.stem.api import StemmerI

class Stemmer(StemmerI):
	def __init__(self):
		self.ends = ['يشان', 'ات', 'ان','تر', 'ترين', 'يي', 'هاي', 'ها', 'ٔ']

	def stem(self, word):
		for end in self.ends:
			if word.endswith(end):
                                word = word[:-len(end)]
                                
		if word.endswith('ۀ'):
			word = word[:-len(end)] + 'ه'

		return word


# Stemmer().stem(word)
