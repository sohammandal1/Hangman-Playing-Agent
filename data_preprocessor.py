import torch
import torch.nn as nn
import string

ALPHABET = list(string.ascii_lowercase)
CHAR2IDX = {ch: i for i, ch in enumerate(ALPHABET)}
IDX2CHAR = {i: ch for ch, i in CHAR2IDX.items()}
PAD_TOKEN = '<PAD>'
BLANK_TOKEN = '<BLANK>'
VOCAB = ALPHABET + [BLANK_TOKEN, PAD_TOKEN]
VOCAB_SIZE = len(VOCAB)

# character level mapping: a-z => 0-25, <BLANK> => 26, <PAD> => 27
VOCAB2IDX = {ch: i for i, ch in enumerate(VOCAB)}
IDX2VOCAB = {i: ch for ch, i in VOCAB2IDX.items()} 

def load_word_dataset(path="words.txt"):
    with open(path, "r") as f:
        words = [line.strip().lower() for line in f if line.strip().isalpha()]
    return words

def encode_word_state(word, revealed, max_len=20):
    """
    Encode the current state of the word (with blanks) into a fixed-length tensor.
    """

    assert len(word) == len(revealed), "Word and revealed mask must match"

    encoding = []
    for i, (w, r) in enumerate(zip(word, revealed)):
        if r:
            encoding.append(VOCAB2IDX[w])
        else:
            encoding.append(VOCAB2IDX[BLANK_TOKEN])

    # Pad the rest
    while len(encoding) < max_len:
        encoding.append(VOCAB2IDX[PAD_TOKEN])

    return torch.tensor(encoding[:max_len], dtype=torch.long)


def encode_guesses(guessed_letters):
    """
    Encode which letters have already been guessed into a 26-dim binary tensor.
    """
    vec = torch.zeros(len(ALPHABET), dtype=torch.float32)
    for ch in guessed_letters:
        if ch in CHAR2IDX:
            vec[CHAR2IDX[ch]] = 1.0
    return vec


def decode_word_state(encoded):
    """
    Decode a tensor of indices back to a human-readable word form with _ for blanks.
    """
    return ''.join(IDX2VOCAB[idx.item()] if IDX2VOCAB[idx.item()] not in [PAD_TOKEN, BLANK_TOKEN] else '_' for idx in encoded)


def encode_word_state_from_revealed(revealed: str, max_len=20):
    """
    Converts a string like '_e__o' into a tensor using known letters and blanks.
    """
    encoding = []
    for ch in revealed:
        if ch == '_':
            encoding.append(VOCAB2IDX[BLANK_TOKEN])
        else:
            encoding.append(VOCAB2IDX[ch])

    while len(encoding) < max_len:
        encoding.append(VOCAB2IDX[PAD_TOKEN])
    return torch.tensor(encoding[:max_len], dtype=torch.long)


# test
if __name__ == '__main__':
    word = 'hangman'
    revealed = [False, True, True, False, False, False, False]  # -an----
    guesses = ['e', 'a', 'n', 's']
    enc = encode_word_state(word, revealed)
    guess_vec = encode_guesses(guesses)
    print("Encoded word:", enc)
    print("Guesses vector:", guess_vec)
    print("Decoded:", decode_word_state(enc))
