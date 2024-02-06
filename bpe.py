
import numpy as np
from ahocorasick import Automaton



file_path = 'data/nq17-23_1min_moves.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

words = text.split('E') # split text into words
#add "E" in the end of each word
words = [word + 'E' for word in words]
print(f'Words count: {len(words)}, unique words: {len(set(words))}')
vocab = set(''.join(words)) # create vocabulary (set of unique chars)
# add "E" in fron of vocabulary
vocab.add('E')

def tokenize(word, vocab):
    """Split word into tokens using character vocabulary
        Most long tokens are created first
         - return list of tokenized words
    """
    tokens = []
    automaton = Automaton()
    for i, token in enumerate(vocab):
        automaton.add_word(token, (i, token))
    automaton.make_automaton()

    start = 0
    while start < len(word):
        end = start + 1
        for _, (token_index, _) in automaton.iter(word[start:]):
            end = start + len(vocab[token_index])
        tokens.append(word[start:end])
        start = end

    return tokens
def tokenize_alt(word, vocab):
    
    vocab_sorted = sorted(vocab, key=lambda x: len(x), reverse=True)
    tokens = {}
    for token in vocab_sorted:
       idx = word.find(token)
       if idx != -1:
           tokens[idx] = token
           

def char_pairs_entropy(char_pairs: dict):
    """Calculate entropy of char pairs
        - return float
    """
    dict_sum = sum(char_pairs.values())
    H = 0
    for pair, count in char_pairs.items():
        H += count/dict_sum * np.log2(count/dict_sum)
    return -H, dict_sum

#count pairs in tokens and return most frequent pair
def max_pair(tokens):
    """Count char pairs in tokens and return most frequent pair
        - return tuple (pair, count)
    """
    char_pairs = {}
    for token in tokens:
        for i in range(len(token)-1):
            pair = ''.join(token[i:i+2])
            char_pairs[pair] = char_pairs.get(pair, 0) + 1
  
    return max(char_pairs.items(), key=lambda x: x[1]), char_pairs


for i in range(100):
    # tokenize words
    print('Tokenizing...')
    tokens = [tokenize(word, vocab) for word in words]
    
    # get most frequent pair
    print('Counting pairs...')
    (pair, count),char_pairs = max_pair(tokens)
     
    h, tot_tokens = char_pairs_entropy(char_pairs)
    print(f'Entropy at step {i}: {h}. Vocabulary size: {len(vocab)}. Tokens in corpus: {tot_tokens}')
    #add new token to vocabulary
    vocab.add(pair)
    # print new token


#save vocabulary
with open('data/vocab.txt', 'w', encoding='utf-8') as file:
    file.write(''.join(vocab))

#tockenize with final vocabulary
print('Final tokenizing...')
tokens = [tokenize(word, vocab) for word in words]
#count frequency of tokens in the tokens list
print('Counting token frequency...')
tokens_freq = {}
for token in tokens:
    tokens_freq[''.join(token)] = tokens_freq.get(''.join(token), 0) + 1

#sort tokens by frequency
tokens_freq = sorted(tokens_freq.items(), key=lambda x: x[1], reverse=True)
#reate mapping from token to index and back
token_to_index = {tokens_freq[i]: i + 1 for i in range(len(tokens_freq))}
index_to_token = {i + 1: tokens_freq[i] for i in range(len(tokens_freq))}
#indexing token corpus
print('Indexing corpus...')
indexed_tokens = [[token_to_index[''.join(token)] for token in word] for word in tokens]

#save indexed corpus
with open('data/indexed_tokens.txt', 'w', encoding='utf-8') as file:
    for word in indexed_tokens:
        file.write(' '.join([str(token) for token in word]) + '\n')
    print('Indexed corpus saved')

#save mappings
with open('data/token_to_index.txt', 'w', encoding='utf-8') as file:
    for token, index in token_to_index.items():
        file.write(f'{token[0]} {index}\n')
    print('Token to index mapping saved')   

with open('data/index_to_token.txt', 'w', encoding='utf-8') as file:
    for index, token in index_to_token.items():
        file.write(f'{index} {token[0]}\n')
    print('Index to token mapping saved')


