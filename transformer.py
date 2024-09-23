import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, heads, d, k, m, dropout=0.):
        super(TransformerBlock, self).__init__()
        self.k = k
        self.heads = heads

# Linear Transformations for Queries, Keys, and Values
# d-dimensional vectors
        self.wq = nn.Linear(d, heads*k, bias=False)
        self.wk = nn.Linear(d, heads*k, bias=False)
        self.wv = nn.Linear(d, heads*k, bias=False)
# Output Linear Layer
        self.wc = nn.Linear(heads*k, d, bias=False)
        self.dropoutatt = nn.Dropout(dropout)

# Feedforward Neural Network (FFN) 
# m (the hidden layer size).
        self.w1 = nn.Linear(d, m)
        self.dropoutfc = nn.Dropout(dropout)
# Output Layer of FFN
        self.w2 = nn.Linear(m, d)

        # task define the dropout
        self.dropout = nn.Dropout(dropout) 

        # task define the layer normalization
        self.norm1 = nn.LayerNorm(d)  # Layer norm after attention
        self.norm2 = nn.LayerNorm(d)  # Layer norm after feed-forward network

      

        nn.init.normal_(self.wq.weight, 0, .02)
        nn.init.normal_(self.wk.weight, 0, .02)
        nn.init.normal_(self.wv.weight, 0, .02)
        nn.init.normal_(self.wc.weight, 0, .02)

        nn.init.normal_(self.w1.weight, 0, .02)
        nn.init.constant_(self.w1.bias, 0.0)
        nn.init.normal_(self.w2.weight, 0, .02)
        nn.init.constant_(self.w2.bias, 0.0)

    def forward(self, x, mask):
        seq_len, batch_size, embed_dim = x.shape

        # task implement scaled dot-product attention
        # 1. Linear projections of Q, K, and V
        # Q, K, V shapes: (seq_len, batch_size, heads * k)
        Q = self.wq(x)  # Shape: (seq_len, batch_size, heads * k)
        K = self.wk(x)  # Shape: (seq_len, batch_size, heads * k)
        V = self.wv(x)  # Shape: (seq_len, batch_size, heads * k)

        # Reshape Q, K, V for multi-head attention
        # New shapes: (batch_size, heads, seq_len, k)
        Q = Q.view(seq_len, batch_size, self.heads, self.k).transpose(1, 2)  # Shape: (batch_size, heads, seq_len, k)
        K = K.view(seq_len, batch_size, self.heads, self.k).transpose(1, 2)  # Shape: (batch_size, heads, seq_len, k)
        V = V.view(seq_len, batch_size, self.heads, self.k).transpose(1, 2)  # Shape: (batch_size, heads, seq_len, k)
          
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.k)  # Shape: (batch_size, heads, seq_len, seq_len)


       # Apply mask if provided
        if mask is not None:
            # Ensure mask is the correct size: (batch_size, 1, seq_len, seq_len)
            mask = mask.unsqueeze(1).expand(-1, self.heads, -1, -1)
            attn_scores = attn_scores.masked_fill(mask == float('-inf'), float('-inf'))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # Shape: (batch_size, heads, seq_len, seq_len)
        # Apply dropout to attention weights
        attn_weights = self.dropoutatt(attn_weights)

        # Compute the weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # Shape: (batch_size, heads, seq_len, k)
        
        # Reshape back to (seq_len, batch_size, embed_dim)
        # attn_output shape: (seq_len, batch_size, heads * k)
        attn_output = attn_output.transpose(1, 2).contiguous().view(seq_len, batch_size, self.heads * self.k)

        # task implement residual connection
        out = self.wc(attn_output)
        out = out + x

        # task implement the dropout
        out = self.dropout(out)

        # task implement the layer normalization
        out = self.norm1(out)

        # task implement the posiion-wise feed forward network
        ff_out = F.relu(self.w1(out))
        ff_out = self.dropoutfc(ff_out)
        ff_out = self.w2(ff_out)
        out = out + ff_out
        out = self.norm2(out)

        # Hint: Writing efficient code is almost as important as writing correct code in ML.
        #       Avoid writing for-loops! Consider using the batch matrix multiplication operator torch.bmm
        raise NotImplementedError('Implement a transformer block')

        return out

class Transformer(nn.Module):
    def __init__(self, seq_len, tokens, d, k, m, heads, layers, tied_weights=False, dropout=0., dropoutio=0.):
        super(Transformer, self).__init__()
        self.mask = None
        self.pos = None
        self.dims = d
        self.tied_weights = tied_weights
        self.dropout=dropout

        self.positional_embedding = nn.Embedding(seq_len, d)
        self.dropi = nn.Dropout(dropoutio)
        self.word_embedding = nn.Embedding(tokens, d)
        self.transformer = nn.ModuleList()
        for i in range(layers):
            self.transformer.append(TransformerBlock(heads, d, k, m, dropout))

        if not tied_weights: self.decoder = nn.Linear(d, tokens)
        self.dropo = nn.Dropout(dropoutio)
        self.bias = nn.Parameter(torch.ones(tokens))

        nn.init.normal_(self.positional_embedding.weight, 0, .02)
        nn.init.normal_(self.word_embedding.weight, 0, .02)
        if not self.tied_weights: nn.init.normal_(self.decoder.weight, 0, .02)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        if self.mask is None or self.mask.shape[0] != x.shape[0]:
            self.mask = torch.triu(torch.ones(len(x), len(x)))
            self.mask.masked_fill_(self.mask == 0, float('-inf')).masked_fill_(self.mask == 1, float(0.0))
            self.mask = self.mask.transpose(0,1).to(x.device)
            self.pos = torch.arange(0, x.shape[0], dtype=torch.long).to(x.device)

        x = self.word_embedding(x) * math.sqrt(self.dims)
        p = self.positional_embedding(self.pos)[:,None,:]
        z = F.relu(self.dropi(x) + self.dropi(p))
        for layer in self.transformer:
            z = layer(z, self.mask)

        z = self.dropo(z)
        outputs = torch.matmul(z, self.word_embedding.weight.t()) if self.tied_weights else self.decoder(z)
        return F.log_softmax(outputs + self.bias, dim=-1)import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, heads, d, k, m, dropout=0.):
        super(TransformerBlock, self).__init__()
        self.k = k
        self.heads = heads

# Linear Transformations for Queries, Keys, and Values
# d-dimensional vectors
        self.wq = nn.Linear(d, heads*k, bias=False)
        self.wk = nn.Linear(d, heads*k, bias=False)
        self.wv = nn.Linear(d, heads*k, bias=False)
# Output Linear Layer
        self.wc = nn.Linear(heads*k, d, bias=False)
        self.dropoutatt = nn.Dropout(dropout)

# Feedforward Neural Network (FFN) 
# m (the hidden layer size).
        self.w1 = nn.Linear(d, m)
        self.dropoutfc = nn.Dropout(dropout)
# Output Layer of FFN
        self.w2 = nn.Linear(m, d)

        # task define the dropout
        self.dropout = nn.Dropout(dropout) 

        # task define the layer normalization
        self.norm1 = nn.LayerNorm(d)  # Layer norm after attention
        self.norm2 = nn.LayerNorm(d)  # Layer norm after feed-forward network

      

        nn.init.normal_(self.wq.weight, 0, .02)
        nn.init.normal_(self.wk.weight, 0, .02)
        nn.init.normal_(self.wv.weight, 0, .02)
        nn.init.normal_(self.wc.weight, 0, .02)

        nn.init.normal_(self.w1.weight, 0, .02)
        nn.init.constant_(self.w1.bias, 0.0)
        nn.init.normal_(self.w2.weight, 0, .02)
        nn.init.constant_(self.w2.bias, 0.0)

    def forward(self, x, mask):
        seq_len, batch_size, embed_dim = x.shape

        # task implement scaled dot-product attention
        # 1. Linear projections of Q, K, and V
        # Q, K, V shapes: (seq_len, batch_size, heads * k)
        Q = self.wq(x)  # Shape: (seq_len, batch_size, heads * k)
        K = self.wk(x)  # Shape: (seq_len, batch_size, heads * k)
        V = self.wv(x)  # Shape: (seq_len, batch_size, heads * k)

        # Reshape Q, K, V for multi-head attention
        # New shapes: (batch_size, heads, seq_len, k)
        Q = Q.view(seq_len, batch_size, self.heads, self.k).transpose(1, 2)  # Shape: (batch_size, heads, seq_len, k)
        K = K.view(seq_len, batch_size, self.heads, self.k).transpose(1, 2)  # Shape: (batch_size, heads, seq_len, k)
        V = V.view(seq_len, batch_size, self.heads, self.k).transpose(1, 2)  # Shape: (batch_size, heads, seq_len, k)
          
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.k)  # Shape: (batch_size, heads, seq_len, seq_len)


       # Apply mask if provided
        if mask is not None:
            # Ensure mask is the correct size: (batch_size, 1, seq_len, seq_len)
            mask = mask.unsqueeze(1).expand(-1, self.heads, -1, -1)
            attn_scores = attn_scores.masked_fill(mask == float('-inf'), float('-inf'))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # Shape: (batch_size, heads, seq_len, seq_len)
        # Apply dropout to attention weights
        attn_weights = self.dropoutatt(attn_weights)

        # Compute the weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # Shape: (batch_size, heads, seq_len, k)
        
        # Reshape back to (seq_len, batch_size, embed_dim)
        # attn_output shape: (seq_len, batch_size, heads * k)
        attn_output = attn_output.transpose(1, 2).contiguous().view(seq_len, batch_size, self.heads * self.k)

        # task implement residual connection
        out = self.wc(attn_output)
        out = out + x

        # task implement the dropout
        out = self.dropout(out)

        # task implement the layer normalization
        out = self.norm1(out)

        # task implement the posiion-wise feed forward network
        ff_out = F.relu(self.w1(out))
        ff_out = self.dropoutfc(ff_out)
        ff_out = self.w2(ff_out)
        out = out + ff_out
        out = self.norm2(out)

        # Hint: Writing efficient code is almost as important as writing correct code in ML.
        #       Avoid writing for-loops! Consider using the batch matrix multiplication operator torch.bmm
        raise NotImplementedError('Implement a transformer block')

        return out

class Transformer(nn.Module):
    def __init__(self, seq_len, tokens, d, k, m, heads, layers, tied_weights=False, dropout=0., dropoutio=0.):
        super(Transformer, self).__init__()
        self.mask = None
        self.pos = None
        self.dims = d
        self.tied_weights = tied_weights
        self.dropout=dropout

        self.positional_embedding = nn.Embedding(seq_len, d)
        self.dropi = nn.Dropout(dropoutio)
        self.word_embedding = nn.Embedding(tokens, d)
        self.transformer = nn.ModuleList()
        for i in range(layers):
            self.transformer.append(TransformerBlock(heads, d, k, m, dropout))

        if not tied_weights: self.decoder = nn.Linear(d, tokens)
        self.dropo = nn.Dropout(dropoutio)
        self.bias = nn.Parameter(torch.ones(tokens))

        nn.init.normal_(self.positional_embedding.weight, 0, .02)
        nn.init.normal_(self.word_embedding.weight, 0, .02)
        if not self.tied_weights: nn.init.normal_(self.decoder.weight, 0, .02)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        if self.mask is None or self.mask.shape[0] != x.shape[0]:
            self.mask = torch.triu(torch.ones(len(x), len(x)))
            self.mask.masked_fill_(self.mask == 0, float('-inf')).masked_fill_(self.mask == 1, float(0.0))
            self.mask = self.mask.transpose(0,1).to(x.device)
            self.pos = torch.arange(0, x.shape[0], dtype=torch.long).to(x.device)

        x = self.word_embedding(x) * math.sqrt(self.dims)
        p = self.positional_embedding(self.pos)[:,None,:]
        z = F.relu(self.dropi(x) + self.dropi(p))
        for layer in self.transformer:
            z = layer(z, self.mask)

        z = self.dropo(z)
        outputs = torch.matmul(z, self.word_embedding.weight.t()) if self.tied_weights else self.decoder(z)
        return F.log_softmax(outputs + self.bias, dim=-1)
