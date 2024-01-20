import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # 假设d_model可以被num_heads整除
        self.depth = d_model // num_heads

        self.Wq1 = nn.Linear(d_model, d_model)
        self.Wk1 = nn.Linear(d_model, d_model)
        self.Wv1 = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, coder, d_model):
        batch_size = x.size(0)

        
        # 第一层self attention
        query1 = self.split_heads(self.Wq1(x), batch_size)
        key1 = self.split_heads(self.Wk1(x), batch_size)
        value1 = self.split_heads(self.Wv1(x), batch_size)

        scores1 = torch.matmul(query1, key1.transpose(-2, -1)) / math.sqrt(self.depth)
        weights1 = F.softmax(scores1, dim=-1)
        
        context1 = torch.matmul(weights1, value1)
        context1 = context1.permute(0, 2, 1, 3).contiguous()
        
        context1 = context1.view(batch_size, -1, self.d_model)
        output = self.dense(context1)
        
        

        
        return output

class dVAE(nn.Module):
    def __init__(self, num_inputs, num_slots, slot_size, d_model, num_heads=8, dropout_rate=0.3):
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.d_model = d_model

        self.encoder = SelfAttention(d_model, num_heads)
        self.encoder_linear2 = nn.Linear(num_inputs,  num_slots)
        self.dropout2 = nn.Dropout(dropout_rate) # 添加dropout层
        self.encoder_linear1 = nn.Linear(d_model, slot_size)
        self.dropout1 = nn.Dropout(dropout_rate) # 添加dropout层
        
        self.decoder = SelfAttention(d_model, num_heads)
        self.decoder_linear1 = nn.Linear(slot_size, d_model)    
        self.dropout3 = nn.Dropout(dropout_rate) # 添加dropout层
        self.decoder_linear2 = nn.Linear(num_slots, num_inputs)
        self.dropout4 = nn.Dropout(dropout_rate) # 添加dropout

    def forward(self, x):
        batch_size = x.size(0)
        
        # 编码
        encoded = self.encoder(x,"encoder",self.d_model)
        encoded = self.dropout1(self.encoder_linear1(encoded)) # 使用dropout
        encoded = encoded.permute(0, 2, 1)
        encoded = self.dropout2(self.encoder_linear2(encoded)) # 使用dropout
        encoded_slots = encoded.permute(0, 2, 1)
        
        # 解码
        decoded = self.dropout3(self.decoder_linear1(encoded_slots)) # 使用dropout
        decoded = decoded.permute(0, 2, 1)
        decoded = self.dropout4(self.decoder_linear2(decoded)) # 使用dropout
        decoded = decoded.permute(0, 2, 1)
        decoded = self.decoder(decoded,"decoder",self.d_model)
        
        return encoded_slots, decoded
        


