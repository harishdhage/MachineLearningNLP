
corpus = '''Pikachu is the electric-type mouse Pokémon species and the mascot of the Pokémon franchise. Created by Atsuko Nishida and finalized by Ken Sugimori, it is known for its yellow fur, red cheeks, and ability to generate electricity, which it can use in attacks or to cause thunderstorms when many gather. Pikachu evolves from Pichu when leveled up with high friendship and can evolve into Raichu when exposed to a Thunder Stone. In the Alola region, it can evolve into Alolan Raichu, which is a dual Electric/Psychic-type Pokémon.'''
print(corpus)

# Tokenization using NLTK
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
# Word Tokenization
word_tokens = word_tokenize(corpus)
print("Word Tokens:", word_tokens)
# Sentence Tokenization
sent_tokens = sent_tokenize(corpus)
print("Sentence Tokens:", sent_tokens) 