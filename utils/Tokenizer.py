from utils.utils import text_spilt
from nltk import pad_sequence


# tokenize the Chemical Identifier string
class Tokenizer:
    def __init__(self):
        self.token2idx = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2}
        self.idx2token = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]'}
        self.idx = 3
        self.max_seq_len = 0
        self.vocab_size = 0

    def text2sequence(self, text, check_max_len=False):
        text = text_spilt(text)
        sequence = [self.token2idx['[BOS]']]
        for s in text.split(' '):
            if s not in self.token2idx:
                self.token2idx[s] = self.idx
                self.idx2token[self.idx] = s
                self.idx += 1

            sequence.append(self.token2idx[s])
        sequence.append(self.token2idx['[EOS]'])
        if not check_max_len:
            sequence = list(
                pad_sequence(sequence, self.max_seq_len - len(sequence) + 1, pad_right=True, right_pad_symbol=0))

            # sequence=np.array(sequence)

        return sequence

    def sequence2text(self, sequence):
        text = []
        for idx in sequence:
            text.append(self.idx2token[idx])

        return text

    def fit(self, train_df, test_df):
        train_sequence_list = list(map(self.text2sequence, train_df['InChI'], [True] * len(train_df)))
        train_max_sequence_len = max([len(seq) for seq in train_sequence_list])

        test_sequence_list = list(map(self.text2sequence, test_df['InChI'], [True] * len(test_df)))
        test_max_sequence_len = max([len(seq) for seq in test_sequence_list])

        self.max_seq_len = max(train_max_sequence_len, test_max_sequence_len)
        self.vocab_size = len(self.token2idx)
