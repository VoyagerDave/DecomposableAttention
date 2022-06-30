import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self,
                 num_embeddings,
                 embedding_size,
                 hidden_size,
                 device,
                 max_length=100):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_size)
        self.pos_embedding = nn.Embedding(max_length, self.embedding_size)
        self.scale = torch.sqrt(torch.FloatTensor([self.embedding_size
                                                   ])).to(device)
        self.input_linear = nn.Linear(self.embedding_size,
                                      self.hidden_size,
                                      bias=False)  # linear transformation
        self.device = device

    def forward(self, sent1, sent2):
        '''
               sent: batch_size x length (Long tensor)
        '''
        batch_size = sent1.shape[0]
        length1 = sent1.shape[1]
        length2 = sent2.shape[1]
        pos1 = torch.arange(0, length1).unsqueeze(0).repeat(batch_size,
                                                           1).to(self.device)
        pos2 = torch.arange(0, length2).unsqueeze(0).repeat(batch_size,
                                                           1).to(self.device)
        sent1 = (self.embedding(sent1) * self.scale) + self.pos_embedding(pos1)
        sent2 = (self.embedding(sent2) * self.scale) + self.pos_embedding(pos2)
        sent1_linear = self.input_linear(sent1).view(batch_size, -1,
                                                     self.hidden_size)
        sent2_linear = self.input_linear(sent2).view(batch_size, -1,
                                                     self.hidden_size)

        return sent1_linear, sent2_linear

class MLP(nn.Module):
    def __init__(self, input_dim, num_hiddens, dropout):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(input_dim, num_hiddens)
        self.relu1 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(num_hiddens, num_hiddens)
        self.relu2 = nn.LeakyReLU()

    def forward(self, inputs):
        x = self.dropout1(inputs)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return x

class Attend(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.f = MLP(input_dim, hidden_dim, dropout=0.2)

    def forward(self, A, B):
        # Shape of `A`/`B`: (b`atch_size`, no. of tokens in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of tokens in sequence A/B,
        # `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of tokens in sequence A,
        # no. of tokens in sequence B)
        e = torch.bmm(f_A, f_B.transpose(1, 2))
        # Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
        # `embed_size`), where sequence B is softly aligned with each token
        # (axis 1 of `beta`) in sequence A
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
        # `embed_size`), where sequence A is softly aligned with each token
        # (axis 1 of `alpha`) in sequence B
        alpha = torch.bmm(F.softmax(e.transpose(1, 2), dim=-1), A)
        return beta, alpha


class Compare(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.g = MLP(2 * input_dim, hidden_dim, dropout=0.2)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat((A, beta), dim=-1))
        V_B = self.g(torch.cat((B, alpha), dim=-1))
        return V_A, V_B


class Aggregate(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_outputs, dropout):
        super().__init__()
        self.h = MLP(2 * input_dim, hidden_dim, dropout=0.2)
        self.linear = nn.Linear(hidden_dim, num_outputs)

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = self.h(torch.cat([V_A, V_B], dim=1))
        Y_hat = self.linear(Y_hat)
        return Y_hat


class DecompAttentionPos(nn.Module):

    def __init__(self, num_embeddings, embedding_size, num_hiddens,
                 num_outputs, dropout, device):
        super().__init__()
        self.embedding = Encoder(num_embeddings, embedding_size, num_hiddens,
                                 device)
        self.attend = Attend(input_dim=num_hiddens,
                             hidden_dim=num_hiddens,
                             dropout=dropout)
        self.compare = Compare(input_dim=num_hiddens,
                               hidden_dim=num_hiddens,
                               dropout=dropout)
        self.aggregate = Aggregate(input_dim=num_hiddens,
                                   hidden_dim=num_hiddens,
                                   num_outputs=num_outputs,
                                   dropout=dropout)

    def forward(self,
                X): 
        premises, hypotheses = X
        A, B = self.embedding(premises, hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
