import math
import random
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F


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


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("bias", [
        #     torch.tril(torch.ones(x, x)).view(1, 1, x, x) for x in range(100)
        # ])
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, xa=None):
        device = x.device
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if xa is None:
            k = self.key(x)
            v = self.value(x)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        else:
            Ba, Ta, Ca = xa.size()

            k = self.key(xa)
            v = self.value(xa)
            k = k.view(Ba, Ta, self.n_head, Ca // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(Ba, Ta, self.n_head, Ca // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        bias = torch.tril(torch.ones(att.shape[2], att.shape[3], device=device)).view(1, 1, att.shape[2], att.shape[3]).detach()

        att = att.masked_fill(bias[:,:,:att.shape[2],:att.shape[3]] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.self_attn = CausalSelfAttention(config)
        self.cross_attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, xa=None):
        x = x + self.self_attn(self.ln_1(x))

        if xa is not None:
            x = x + self.cross_attn(self.ln_1(x), xa)

        x = x + self.mlp(self.ln_2(x))
        return x


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.cnn1 = nn.Sequential(
            nn.Conv1d(config.input_dim, config.cnn_1_dim, kernel_size=config.cnn_1_kernel_size, stride=config.cnn_1_stride),
            GeM(kernel_size=6),
            nn.BatchNorm1d(config.cnn_1_dim),
            nn.SiLU(),
        )

        self.transformer = nn.ModuleDict(dict(
            # wte = nn.Embedding(config.vocab_size, config.n_embd),
            # wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.register_buffer("positional_embedding", sinusoids(config.cnn_1_dim))

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (LayerNorm, nn.LayerNorm)):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx):
        # device = idx.device
        # b, t = idx.size()
        # assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        # tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        # pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)

        tok_emb = self.cnn1(idx)
        tok_emb = tok_emb.permute(0,2,1)
        pos_emb = self.positional_embedding[tok_emb.shape[1]]

        emb = (tok_emb + pos_emb).to(tok_emb.dtype)
        
        x = self.transformer.drop(emb)
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)

        return x


class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (LayerNorm, nn.LayerNorm)):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, xa):
        # print(idx.shape, print(torch.max(idx, axis=0)))
        # print(idx)
        device = idx.device
        b, t = idx.size()
        # assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        # print(tok_emb.shape, pos_emb.shape)
        x = tok_emb + pos_emb
        # x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, xa)
        x = self.transformer.ln_f(x)

        return self.lm_head(x)


class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, idx, targets):
        encoder = self.encoder(idx)
        return self.decoder(targets[:,:-1], encoder)

    @torch.no_grad()
    def generate(self, idx, trg, teacher_forcing_ratio):
        device = idx.device

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.config.vocab_size

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=device)
        inputs = torch.zeros(batch_size, trg_len, device=device)

        encoder = self.encoder(idx)

        inputs[:, 0] = trg[:, 0]
        for t in range(1, trg_len):
            input = inputs[:, :t].long()

            output = self.decoder(input, encoder)

            outputs[:,t,:] = output[:,-1,:]

            teacher_force = random.random() < teacher_forcing_ratio
            inputs[:, t] = trg[:, t] if teacher_force else output[:,-1,:].argmax(1)

        return outputs

    
    def configure_optimizers(self, weight_decay, learning_rate, betas, warmup=None):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        if warmup is None:
            warmup = 400

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        blacklist_modules = ['bias', 'ln', 'wte', 'wpe']

        no_decay = set([n for n,p in self.named_parameters() if any(nd in n for nd in blacklist_modules)])
        decay = set([n for n,p in self.named_parameters() if not any(nd in n for nd in blacklist_modules)])

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        input_dim = sum(p.numel() for p in self.encoder.parameters())
        optimizer = NoamOpt(input_dim, warmup,
                            torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas))

        return optimizer


###############################
### Helper optimizer class ####
###############################

class NoamOpt(object):
    "Optim wrapper that implements rate."
    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0
    
    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict) 
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5))) 
