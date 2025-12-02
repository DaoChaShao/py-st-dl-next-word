#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/24 23:45
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   rnn_lstm_classification.py
# @Desc     :   

from torch import nn, cat

WIDTH: int = 64


class LSTMRNNForClassification(nn.Module):
    """ AN RNN model for multi-class classification tasks using PyTorch """

    def __init__(
            self,
            vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int,
            num_classes: int, dropout_rate: float = 0.3
    ):
        super().__init__()
        """ Initialise the CharsRNNModel class
        :param vocab_size: size of the vocabulary
        :param embedding_dim: dimension of the embedding layer
        :param hidden_dim: dimension of the hidden layer
        :param num_layers: number of RNN layers
        :param num_classes: number of output classes
        :param dropout_rate: dropout rate for regularization
        """
        self._L = vocab_size  # Lexicon/Vocabulary size
        self._N = embedding_dim  # Embedding dimension
        self._M = hidden_size  # Hidden dimension
        self._C = num_layers  # RNN layers count

        self._embed = nn.Embedding(self._L, self._N)
        self._lstm = nn.LSTM(self._N, self._M, self._C, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self._dropout = nn.Dropout(dropout_rate)
        self._classifier = nn.Linear(self._M * 2, num_classes)

        self._init_params()

    def _init_params(self):
        """ Initialize model parameters """
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, X):
        """ Forward pass of the model
        :param X: input tensor, shape (batch_size, sequence_length)
        :return: output tensor and new hidden state tensor, shapes (batch_size, sequence_length, vocab_size) and (num_layers, batch_size, hidden_dim)
        """
        out = self._embed(X)
        _, (hn, _) = self._lstm(out)

        forward_hn = hn[-2]  # [batch_size, hidden_size]
        backward_hn = hn[-1]  # [batch_size, hidden_size]
        out = cat([forward_hn, backward_hn], dim=1)  # [batch_size, hidden_size*2]

        out = self._dropout(out)
        # Fully connected layer, shape (batch_size, num_classes)
        out = self._classifier(out)

        return out

    def summary(self):
        """ Print the model summary """
        print("=" * WIDTH)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model Summary for {self.__class__.__name__}")
        print("-" * WIDTH)
        print(f"- Vocabulary size: {self._L}")
        print(f"- Embedding dim: {self._N}")
        print(f"- Hidden size: {self._M}")
        print(f"- Num layers: {self._C}")
        print(f"- Output classes: {self._classifier.out_features}")
        print(f"- Total parameters: {total_params:,}")
        print(f"- Trainable parameters: {trainable_params:,}")
        print("=" * WIDTH)
        print()


if __name__ == "__main__":
    pass
