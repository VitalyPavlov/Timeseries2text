import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


class GeM(nn.Module):
    """
    Code modified from the 2d code in
    https://amaarora.github.io/2020/08/30/gempool.html
    """

    def __init__(self, kernel_size=8, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool1d(x.clamp(min=eps).pow(p), self.kernel_size).pow(1.0 / p)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


def sinusoids(channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    emb = []
    for length in range(1,200):
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        emb.append(torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1))

    return torch.cat(emb)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.input_dim
        self.cnn_1_dim = config.cnn_1_dim
        self.cnn_1_kernel_size = config.cnn_1_kernel_size
        self.cnn_1_stride = config.cnn_1_stride
        self.hid_dim = config.hid_dim
        self.num_layers = config.num_layers
        
        self.cnn1 = nn.Sequential(
            nn.Conv1d(config.input_dim, config.cnn_1_dim, kernel_size=config.cnn_1_kernel_size, stride=config.cnn_1_stride),
            GeM(kernel_size=6),
            nn.BatchNorm1d(config.cnn_1_dim),
            nn.SiLU(),
        )
        # self.cnn2 = nn.Sequential(
        #     nn.Conv1d(64, 128, kernel_size=8, stride=1),
        #     nn.BatchNorm1d(128),
        #     nn.SiLU(),
        # )
        
        # self.cnn3 = nn.Sequential(
        #     nn.Conv1d(128, 256, kernel_size=3, stride=1),
        #     nn.BatchNorm1d(256),
        #     nn.SiLU(),
        # )
        
        self.dropout_1 = nn.Dropout(0.8)
        self.dropout_2 = nn.Dropout(0.5)

        self.gru = nn.GRU(config.cnn_1_dim, config.hid_dim, num_layers=config.num_layers, batch_first=True, bidirectional=False)
        self.register_buffer("positional_embedding", sinusoids(config.cnn_1_dim))
        

    def forward(self, x):
        # x = self.dropout_1(x)
        x = self.cnn1(x)
        # x = self.cnn2(x)
        # x = self.cnn3(x)

        # x = x.permute(0,2,1)
        # x = (x + self.positional_embedding[x.shape[1]]).to(x.dtype)

        x = self.dropout_2(x)
        # x = x.permute(1,0,2)
        x = x.permute(0,2,1)
        output, hidden = self.gru(x)
        
        return hidden

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.hid_dim = config.hid_dim
        self.num_layers = config.num_layers
        
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd) # 10 3
        self.position_emb = nn.Embedding(config.block_size, config.n_embd) # 14 3
        self.gru = nn.GRU(config.n_embd + config.hid_dim, config.hid_dim, num_layers=config.num_layers, batch_first=True, bidirectional=False)
        self.fc_out = nn.Linear(config.hid_dim, config.vocab_size)
    
    def forward(self, x, hidden, context):
        tok_emb = self.token_emb(x)
        pos_emb = self.position_emb(torch.arange(x.shape[1], device=x.device))

        emb = tok_emb + pos_emb
        _context = context[-1:,:,:].permute(1,0,2)
        emb = torch.cat((emb, _context.repeat(1, emb.shape[1], 1)), dim=2)
        output, hidden = self.gru(emb, hidden)
        output = self.fc_out(output.squeeze(0))
        return output, hidden
    
    
class Seq2Seq(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
    def forward(self, x, trg):
        context = self.encoder(x)
        # print(x.shape, context.shape)
        outputs, _ = self.decoder(trg[:,:-1], context, context)
        return outputs


    def generate(self, x, trg, teacher_forcing_ratio):
        device = x.device

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=device)
        inputs = torch.zeros(batch_size, trg_len, device=device)
        context = self.encoder(x)
        hidden = context

        
        inputs[:, 0] = trg[:, 0]
        for t in range(1, trg_len):
            input = inputs[:, :t].long()
            # input = input.unsqueeze(1)
            # print(input.shape, hidden.shape, context.shape)
            output, hidden = self.decoder(input, context, context)
            outputs[:,t,:] = output[:,-1,:]

            # print(1, inputs[:, t].shape)
            # print(output[:,-1,:].argmax(1))
            # print()
            
            teacher_force = random.random() < teacher_forcing_ratio
            inputs[:, t] = trg[:, t] if teacher_force else output[:,-1,:].argmax(1) # torch.nn.functional.softmax(output, dim=1).detach()
        
        # print(trg)
        # print(outputs)
        return outputs
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, warmup=None):
        optimizer = AdamOpt(torch.optim.Adam(self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay))
        return optimizer
    

class AdamOpt(object):
    "Optim wrapper that implements rate."
    def __init__(self, optimizer):
        self.optimizer = optimizer
        
    def step(self):
        self.optimizer.step()
        
    def rate(self):
        return self.optimizer.param_groups[0]["lr"]


