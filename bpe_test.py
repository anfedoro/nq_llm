
import numpy as np
import time
from collections import Counter
import json

file_path = 'data/nq17-23_1min_moves_3.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

words = text.split() # split text into words
#add "E" in the end of each word
print(f'Words count: {len(words)}, unique words: {len(set(words))}')
vocab = set(''.join(words)) # create vocabulary (set of unique chars)
vocab_new = {1: vocab.copy()}


def tokenize_word(word, token_dicts):
    if not word:
        return []

    for token_length in sorted(token_dicts.keys(), reverse=True):
        tokens = token_dicts[token_length]
        for token in tokens:
            if token in word:
                parts = word.split(token)
                result = []
                for part in parts[:-1]:
                    result.extend(tokenize_word(part, token_dicts))
                    result.append(token)
                result.extend(tokenize_word(parts[-1], token_dicts))
                return result


    return [word]

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
    """Receive list of tokens, pair them and return most frequent pair."""
    char_pairs = Counter()
    for token in tokens:
        char_pairs.update(''.join(pair) for pair in zip(token, token[1:]))
    
    return char_pairs.most_common(1)[0], char_pairs


vocab_len = len(vocab)
target_vocab_len = 100
i = 0
start=time.time()
print(f'Starting BPE with {target_vocab_len} tokens target vocabulary...')
while vocab_len < target_vocab_len:
    # tokenize words
    #print('Tokenizing...')

  
    tokens = [tokenize_word(word, vocab_new) for word in words]


    # get most frequent pair
    #print('Counting pairs...')
    (pair, count),char_pairs = max_pair(tokens)
    
    h, tot_tokens = char_pairs_entropy(char_pairs)
    vocab.add(pair)

    token_set = vocab_new.get(len(pair), set())
    token_set.add(pair)
    vocab_new[len(pair)] = token_set

    vocab_len = len(vocab)

    if i % 10 == 0:
        print(f'Entropy at step {i}: {h}. Vocabulary size: {vocab_len}. Tokens in corpus: {tot_tokens}')
        print(f'Iteration done in {time.time()-start:.2f} sec')
        start = time.time()
    #add new token to vocabulary
    i += 1

    
    # print new token


#save vocabulary_new as json
for key in vocab_new.keys():
    vocab_new[key] = list(vocab_new[key])


with open('data/vocab100.json', 'w', encoding='utf-8') as file:
    json.dump(vocab_new, file)
    print('Vocabulary saved')

#tockenize with final vocabulary
print('Final tokenizing...')

tokens = [tokenize_word(word, vocab_new) for word in words]
#count frequency of tokens in the tokens list
print('Counting  unique token frequency...')
tokens_freq = {}
for word_tokens in tokens:
    for token in word_tokens:
        tokens_freq[''.join(token)] = tokens_freq.get(''.join(token), 0) + 1


#sort vocabulary tokens by frequency
    
tokens_freq = sorted(tokens_freq.items(), key=lambda x: x[1], reverse=True)
#reate mapping from token to index and back
token_to_index = {tokens_freq[i][0]: i + 1 for i in range(len(tokens_freq))}
index_to_token = {i + 1: tokens_freq[i][0] for i in range(len(tokens_freq))}
#indexing token corpus
print('Indexing corpus...')
indexed_tokens  = []
for word_tokens in tokens:
    indexed_word = []
    for token in word_tokens:
        indexed_word.append(token_to_index[token])
    indexed_tokens.append(indexed_word)


#save indexed corpus
with open('data/indexed_tokens_100.txt', 'w', encoding='utf-8') as file:
    for word in indexed_tokens:
        file.write(' '.join([str(token) for token in word]) + '\n')
    print('Indexed corpus saved')

#save mappings as json

with open('data/token_to_index_100.json', 'w', encoding='utf-8') as file:
    json.dump(token_to_index, file)
    print('Token to index mapping saved')
with open('data/index_to_token_100.json', 'w', encoding='utf-8') as file:
    json.dump(index_to_token, file)
    print('Index to token mapping saved')
  

