import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class CharRNN(nn.Module):

    @staticmethod
    def _device(model):
        return next(model.parameters()).device

    def __init__(self, vocabulary, hidden_size, num_layers, dropout, device):
        super(CharRNN, self).__init__()

        self.vocabulary = vocabulary
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.vocab_size = self.input_size = self.output_size = len(vocabulary)

        self.embedding_layer = nn.Embedding(self.vocab_size, self.vocab_size, padding_idx=vocabulary.pad)
        self.lstm_layer = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout,
                                  batch_first=True)
        self.linear_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, lengths, hiddens=None):
        x = self.embedding_layer(x)

        x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True)

        x, hiddens = self.lstm_layer(x, hiddens)

        x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)

        x = self.linear_layer(x)

        return x, lengths, hiddens

    def sample_smiles(self, max_length, batch_size):
        with torch.no_grad():
            starts = [torch.tensor([self.vocabulary.bos], dtype=torch.long, device=self.device)
                      for _ in range(batch_size)
                      ]

            starts = torch.tensor(starts, dtype=torch.long, device=self.device).unsqueeze(1)

            new_smiles_list = [
                torch.tensor(self.vocabulary.pad, dtype=torch.long, device=self.device).repeat(max_length + 2)
                for _ in range(batch_size)
            ]

            for i in range(batch_size):
                new_smiles_list[i][0] = self.vocabulary.bos

            len_smiles_list = [1 for _ in range(batch_size)]
            lens = torch.tensor([1 for _ in range(batch_size)], dtype=torch.long, device=self.device)
            end_smiles_list = [False for _ in range(batch_size)]

            hiddens = None
            for i in range(1, max_length + 1):
                output, _, hiddens = self.forward(starts, lens, hiddens)

                # probabilities
                probs = [F.softmax(o, dim=-1) for o in output]

                # sample from probabilities
                ind_tops = [torch.multinomial(p, 1) for p in probs]

                for j, top in enumerate(ind_tops):
                    if not end_smiles_list[j]:
                        top_elem = top[0].item()
                        if top_elem == self.vocabulary.eos:
                            end_smiles_list[j] = True

                        new_smiles_list[j][i] = top_elem
                        len_smiles_list[j] = len_smiles_list[j] + 1

                starts = torch.tensor(ind_tops, dtype=torch.long, device=self.device).unsqueeze(1)

            new_smiles_list = [new_smiles_list[i][:l] for i, l in enumerate(len_smiles_list)]

            return new_smiles_list
