# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import math
from misc import inverse_sigmoid
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from models.attention import MultiheadAttention
from utils.object_utils import *
from models.dvae import dVAE
from models.module import MLP


def gen_sineembed_for_position(pos_tensor, d_model):
    # pos_tensor [#queries, bsz, query_dim]
    scale = 2 * math.pi
    dim_t = torch.arange(d_model, dtype=torch.float32, device=pos_tensor.device)    # [d_model]
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / d_model)
    x_embed = pos_tensor[:, :, 0] * scale    # [#queries, bsz]
    pos_x = x_embed[:, :, None] / dim_t    # [#queries, bsz, d_model]
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)    # [#queries, bsz, d_model]

    w_embed = pos_tensor[:, :, 1] * scale    # [#queries, bsz]
    pos_w = w_embed[:, :, None] / dim_t    # [#queries, bsz, d_model]
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)    # [#queries, bsz, d_model]

    pos = torch.cat((pos_x, pos_w), dim=2)    # [#queries, bsz, 2*d_model]

    return pos

def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    
    m = nn.Linear(in_features, out_features, bias)
    
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)
    
    if bias:
        nn.init.zeros_(m.bias)
    
    return m

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, 
                 num_CMAencoder_layers=3,
                 num_encoder_layers=3, 
                 num_decoder_layers=3, 
                 dim_feedforward=2048, 
                 dropout=0.1,
                 activation="relu",
                 return_intermediate_dec=True, 
                 query_dim=2,
                 max_v_l=75,
                 num_queries=10,
                 num_iteration=3):
        super().__init__()

        CMA_layer = CM_TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)

        self.CMAencoder = CrossModalityEncoder(CMA_layer,num_CMAencoder_layers)

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        
        # self.globaltokenMLP = MLP(d_model,d_model,d_model,num_globaltoken_layers)

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        slot_atten = SlotAttention(num_iteration, num_queries, d_model)

        object_slot_atten = SlotAttentionVideo(num_iteration, num_queries, d_model)

        d_vae = dVAE(max_v_l, num_queries, 192, d_model)

        first_decoder_layer = TransformerDecoderFirstLayer(d_model, nhead, dim_feedforward,
                                                           dropout, activation)
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        
        decoder_norm = nn.LayerNorm(d_model)

        decoder_event_norm = nn.LayerNorm(d_model)
        
        self.decoder = TransformerDecoder(slot_atten, object_slot_atten,d_vae, first_decoder_layer, decoder_layer,
                                          num_decoder_layers, decoder_norm, decoder_event_norm,
                                          return_intermediate=return_intermediate_dec, d_model=d_model,
                                          query_dim=query_dim)


        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed, src_vid, pos_embed_vid, src_vid_mask, src_txt_global,tau = 0.8):
        video_length = src_vid.shape[1] - 1

        src = src.permute(1, 0, 2)  # (L, bsz, d)
        pos_embed = pos_embed.permute(1, 0, 2)  # (L, bsz, d)
        src_txt_global = src_txt_global.unsqueeze(1).permute(1,0,2)  # (1, bsz, d)

        ats = None 

        src , ats = self.CMAencoder(src,src_key_padding_mask=mask, 
                                    pos=pos_embed,video_length=video_length)

        memory , ats2 = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # (L, bsz, d)

        if ats is None:
            ats = ats2[:,1:video_length+1,video_length+1:]
        else:
            ats += ats2[:,1:video_length+1,video_length+1:]

        global_token=memory[0]
        memory=memory[1:]
        src_vid=src_vid[:,1:]
        pos_embed = pos_embed[1:]
        pos_embed_vid=pos_embed_vid[:,1:]
        src_vid_mask=src_vid_mask[:,1:]
        mask=mask[:,1:]
        # 这里需要处理decoder的问题

        hs, references, cross_entropy,dvae_mse = self.decoder(memory, tau, memory_key_padding_mask=mask, 
                                      pos=pos_embed, 
                                      src_vid=src_vid,
                                      pos_vid=pos_embed_vid,
                                      src_vid_mask=src_vid_mask,                                      
                                      src_txt_global=src_txt_global)  # (#layers, bsz, #qeries, d)

        memory = memory.transpose(0, 1)  # (bsz, L, d)
        return hs, memory, references , ats , global_token, cross_entropy, dvae_mse

# =====WYY Part===== 

class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        """
        x: B, N, vocab_size
        """

        tokens = torch.argmax(x, dim=-1)  # batch_size x N
        token_embs = self.dictionary(tokens)  # batch_size x N x emb_size
        return token_embs
    

class TransformerDecoder(nn.Module):

    def __init__(self, slot_atten, object_slot_atten, d_vae,first_decoder_layer, decoder_layer, num_layers, norm=None, event_norm=None, return_intermediate=False,
                d_model=512, query_dim=2):
        super().__init__()

        self.slot_atten = slot_atten
                # New slot attention for object detection
        self.object_slot_atten = object_slot_atten
        self.d_vae = d_vae
        self.layers = nn.ModuleList([])
        self.layers.append(first_decoder_layer)
        self.layers.extend(_get_clones(decoder_layer, num_layers-1))

        self.num_layers = num_layers
        self.norm = norm
        self.event_norm = event_norm
        self.return_intermediate = return_intermediate
        self.query_dim = query_dim
        self.d_model = d_model
        self.merge_slots = MergeSlotInformation(d_model=d_model)

        # DAB-DETR
        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)  # position embedding dimension reduction
        self.query_scale = MLP(d_model, d_model, d_model, 2)  # center scaling
        self.ref_anchor_head = MLP(d_model, d_model, 1, 2)  # width modulation

        self.event_span_embed = None
        self.moment_span_embed = None
        self.dict = OneHotDictionary(192,192)
        
        # Create a linear transformation layer without overwriting object_slots tensor
        self.linear_transformation = nn.Linear(192, d_model)
        # 余弦相似度损失函数
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
    def forward(self, memory, tau,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                src_vid: Optional[Tensor] = None,
                pos_vid: Optional[Tensor] = None,
                src_vid_mask: Optional[Tensor] = None,
                src_txt_global: Optional[Tensor] = None):

        intermediate = []
        ref_points = []

        # Event reasoning
        src_slot = src_vid+pos_vid
        output = self.slot_atten(src_slot, src_vid_mask)  # [bsz, #queries, d_model]
        
        output = output.permute(1,0,2)  # [#queries, bsz, d_model]
        queries = src_slot.shape[1]
        d_model = src_slot.shape[2]
        # Object reasoning
        object_slots, attns = self.object_slot_atten(src_slot, src_vid_mask)  
        # object_slots: B, num_slots, slot_size :  32*10*192

        encoded_slots, decoded = self.d_vae(src_slot)
        B = object_slots.shape[0]
        num_slots = object_slots.shape[1]
        slot_size = object_slots.shape[2]

        # 将每个slot视为一个独立的样本来处理:
        #object_slots_reshaped = object_slots.view(-1, object_slots.size(-1))
        #encoded_slots_reshaped = encoded_slots.view(-1, encoded_slots.size(-1)).detach()
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 创建一个全是1的target张量，表示所有样本都是正样本（相似）。
        #target = torch.ones(B * num_slots, dtype=torch.float32)
        # 计算损失 (注意 tensor 必须是归一化的)
        #loss = self.cosine_loss(F.normalize(object_slots_reshaped, dim=-1).to(device), 
        #           F.normalize(encoded_slots_reshaped, dim=-1).to(device), 
        #          target.to(device))
        #实际上是余弦相似度，先试试看，成功了就改

        #z_hard = gumbel_softmax(encoded_slots, tau, True, dim=1).detach()
        #cross_entropy= -loss #.item()

        z_hard = gumbel_softmax(encoded_slots, tau, True, dim=1).detach()
        cross_entropy = -(z_hard * torch.log_softmax(object_slots, dim=-1)).sum() / (B * num_slots * slot_size)  

        dvae_mse = ((src_slot - decoded) ** 2).sum() / (B * queries * d_model)
        
        

        # Apply the linear layer to each slot (assuming object_slots is a tensor with shape [B, num_slots, slot_size])
        transformed_slots = self.linear_transformation(object_slots)

        # Now, you can permute the transformed tensor, not the layer.
        transformed_slots = transformed_slots.permute(1, 0, 2)

        output = output + transformed_slots

        if self.event_span_embed:
            tmp = self.event_span_embed(output)
            new_reference_points = tmp[..., :self.query_dim].sigmoid()
            ref_points.append(new_reference_points)
            reference_points = new_reference_points.detach()

        if self.return_intermediate:
            intermediate.append(self.event_norm(output))

        

        # Moment reasoning
        for layer_id, layer in enumerate(self.layers):
            ref_pt = reference_points[..., :self.query_dim]  # [#queries, bsz, 2] (xw)
            #print("ref_pt:",ref_pt)
            query_sine_embed = gen_sineembed_for_position(ref_pt, self.d_model)  # [#queries, bsz, 2*d_model] - (:d_model)은 center, (d_model:)은 width (xw)
            query_pos = self.ref_point_head(query_sine_embed)  # [#queries, bsz, d_model] (xw)

            # Conditional-DETR
            pos_transformation = self.query_scale(output)
            query_sine_embed = query_sine_embed[...,:self.d_model] * pos_transformation  # [#queries, bsz, d_model] (x)
            # modulated w attentions
            refW_cond = self.ref_anchor_head(output).sigmoid()  # [#queries, bsz, 1] (w)
            query_sine_embed *= (refW_cond[..., 0] / ref_pt[..., 1]).unsqueeze(-1)  # (x *= w/w)

            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, 
                           query_sine_embed=query_sine_embed,
                           src_txt_global=src_txt_global)
            #if layer_id == 0:
             #   output += transformed_slots
            if self.moment_span_embed:

                tmp = self.moment_span_embed(output)
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.moment_span_embed:
                return [
                    torch.stack(intermediate).transpose(1,2),
                    torch.stack(ref_points).transpose(1,2),
                    cross_entropy, 
                    dvae_mse
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1,2),
                    reference_points.unsqueeze(0).transpose(1,2),
                    cross_entropy, 
                    dvae_mse
                ]

        return output.unsqueeze(0), cross_entropy, dvae_mse


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, dropout=0., gain=1.):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        self.proj_q = linear(d_model, d_model, bias=False)
        self.proj_k = linear(d_model, d_model, bias=False)
        self.proj_v = linear(d_model, d_model, bias=False)
        self.proj_o = linear(d_model, d_model, bias=False, gain=gain)
    
    
    def forward(self, q, k, v, attn_mask=None):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """
        B, T, _ = q.shape
        _, S, _ = k.shape
        
        q = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2)
        k = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2)
        v = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2)
        
        q = q * (q.shape[-1] ** (-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        output = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        output = self.proj_o(output)
        output = self.output_dropout(output)
        return output


class TransformerEncoderBlock(nn.Module):
    
    def __init__(self, d_model, num_heads, dropout=0., gain=1., is_first=False):
        super().__init__()
        
        self.is_first = is_first
        
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout, gain)
        
        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            linear(d_model, 4 * d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(4 * d_model, d_model, gain=gain),
            nn.Dropout(dropout))
    
    
    def forward(self, input):
        """
        input: batch_size x source_len x d_model
        return: batch_size x source_len x d_model
        """
        if self.is_first:
            input = self.attn_layer_norm(input)
            x = self.attn(input, input, input)
            input = input + x
        else:
            x = self.attn_layer_norm(input)
            x = self.attn(x, x, x)
            input = input + x
        
        x = self.ffn_layer_norm(input)
        x = self.ffn(x)
        return input + x
    

class TransformerEncoder_Video(nn.Module):
    
    def __init__(self, num_blocks, d_model, num_heads, dropout=0.):
        super().__init__()
        
        if num_blocks > 0:
            gain = (2 * num_blocks) ** (-0.5)
            self.blocks = nn.ModuleList(
                [TransformerEncoderBlock(d_model, num_heads, dropout, gain, is_first=True)] +
                [TransformerEncoderBlock(d_model, num_heads, dropout, gain, is_first=False)
                 for _ in range(num_blocks - 1)])
        else:
            self.blocks = nn.ModuleList()
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    
    def forward(self, input):
        """
        input: batch_size x source_len x d_model
        return: batch_size x source_len x d_model
        """
        for block in self.blocks:
            input = block(input)
        
        return self.layer_norm(input)
    

class TransformerDecoderBlock(nn.Module):
    
    def __init__(self, max_len, d_model, num_heads, dropout=0., gain=1., is_first=False):
        super().__init__()
        
        self.is_first = is_first
        
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, gain)
        
        mask = torch.triu(torch.ones((max_len, max_len), dtype=torch.bool), diagonal=1)
        self.self_attn_mask = nn.Parameter(mask, requires_grad=False)
        
        self.encoder_decoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_decoder_attn = MultiHeadAttention(d_model, num_heads, dropout, gain)
        
        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            linear(d_model, 4 * d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(4 * d_model, d_model, gain=gain),
            nn.Dropout(dropout))
    
    
    def forward(self, input, encoder_output):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        T = input.shape[1]
        
        if self.is_first:
            input = self.self_attn_layer_norm(input)
            x = self.self_attn(input, input, input, self.self_attn_mask[:T, :T])
            input = input + x
        else:
            x = self.self_attn_layer_norm(input)
            x = self.self_attn(x, x, x, self.self_attn_mask[:T, :T])
            input = input + x
        
        x = self.encoder_decoder_attn_layer_norm(input)
        x = self.encoder_decoder_attn(x, encoder_output, encoder_output)
        input = input + x
        
        x = self.ffn_layer_norm(input)
        x = self.ffn(x)
        return input + x


class TransformerDecoder_Video(nn.Module):
    
    def __init__(self, num_blocks, max_len, d_model, num_heads, dropout=0.):
        super().__init__()
        
        if num_blocks > 0:
            gain = (3 * num_blocks) ** (-0.5)
            self.blocks = nn.ModuleList(
                [TransformerDecoderBlock(max_len, d_model, num_heads, dropout, gain, is_first=True)] +
                [TransformerDecoderBlock(max_len, d_model, num_heads, dropout, gain, is_first=False)
                 for _ in range(num_blocks - 1)])
        else:
            self.blocks = nn.ModuleList()
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    
    def forward(self, input, encoder_output):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        for block in self.blocks:
            input = block(input, encoder_output)
        
        return self.layer_norm(input)


class SlotAttentionVideo(nn.Module):
    
    def __init__(self, num_iterations, num_slots,
                 input_size,  mlp_hidden_size = 192,
                 num_predictor_blocks=1,
                 num_predictor_heads=4,
                 dropout=0.1,
                 epsilon=1e-8):
        super().__init__()
        
        self.num_iterations = 3
        self.num_slots = num_slots
        self.input_size = input_size
        slot_size=192
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        # parameters for Gaussian initialization (shared by all slots).
        self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        # norms
        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        
        # linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)
        
        # slot update functions.
        self.gru = gru_cell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            linear(slot_size, mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_hidden_size, slot_size))
        self.predictor = TransformerEncoder_Video(num_predictor_blocks, slot_size, num_predictor_heads, dropout)
        self.attn_holder = nn.Identity()

    def forward(self, inputs, mask):
        B, num_inputs, input_size = inputs.size()

        # initialize slots
        slots = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
        slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots
        #print("slots.shape",slots.shape)
        #print("inputs.shape",inputs.shape)
        # setup key and value
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        #print("k.shape",k.shape)
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        #print("v.shape",v.shape)
        k = (self.slot_size ** (-0.5)) * k
        
        # corrector iterations
        for i in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            #print("q.shape",q.shape)                
            dots = torch.einsum('bid,bjd->bij', q, k)   # [bsz, num_slots, n_inputs]

            max_neg_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(mask.unsqueeze(1), max_neg_value)

            attn = dots.softmax(dim=1)
            attn_vis = attn.clone()
            attn = self.attn_holder(attn) #???!
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.epsilon)  # softmax over slots
            updates = torch.einsum('bjd,bij->bid', v, attn)  # [bsz, num_slots, d_model].
            #print("updates.shape",updates.shape)
            # Slot update.
            slots = self.gru(updates.reshape(-1, self.slot_size),
                                 slots_prev.view(-1, self.slot_size))
            slots = slots.view(-1, self.num_slots, self.slot_size)

            # use MLP only when more than one iterations
            if i < self.num_iterations - 1:
                slots = slots + self.mlp(self.norm_mlp(slots))

        # predictor
        slots = self.predictor(slots)

        return slots, attn_vis


class MergeSlotInformation(nn.Module):
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model
        
        # 一个简单的两层MLP
        self.fc1 = nn.Linear(2 * d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        
        # 激活函数, 例如 ReLU
        self.act_fn = nn.ReLU()

    def forward(self, event_output, object_output):
        # 将来自事件槽与对象槽的表示拼接在一起
        combined = torch.cat((event_output, object_output), dim=-1)
        
        # 通过两层MLP传递拼接后的信息
        x = self.act_fn(self.fc1(combined))
        x = self.fc2(x)
        return x
    

class SlotAttention(nn.Module):
    """Slot Attention module."""

    def __init__(self, num_iterations, num_slots, d_model,
                epsilon=1e-8):
        """Builds the Slot Attention module.
        Args:
            num_iterations: Number of iterations.
            num_slots: Number of slots.
            d_model: Hidden layer size of MLP.
            epsilon: Offset for attention coefficients before normalization.
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.d_model = d_model
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(d_model)
        self.norm_slots = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

        self.slots = nn.Parameter(torch.randn(num_slots, d_model))
        nn.init.xavier_normal_(self.slots)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(d_model, d_model)
        self.project_k = nn.Linear(d_model, d_model)
        self.project_v = nn.Linear(d_model, d_model)

        self.attn_holder = nn.Identity()

        # Slot update functions.
        self.mlp = MLP(d_model, d_model, d_model, 2)

    def forward(self, inputs, mask):
        b = inputs.shape[0]  # [bsz, n_inputs, d_model]

        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input
        k = self.project_k(inputs)  # [bsz, n_inputs, d_model]
        v = self.project_v(inputs)   # [bsz, n_inputs, d_model]

        slots = self.slots.repeat(b,1,1)

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)   # [bsz, num_slots, d_model]
            scale = self.d_model ** -0.5    # Normalization

            dots = torch.einsum('bid,bjd->bij', q, k) * scale  # [bsz, num_slots, n_inputs]

            max_neg_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(mask.unsqueeze(1), max_neg_value)

            attn = dots.softmax(dim=1)
            attn = self.attn_holder(attn)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.epsilon)  # softmax over slots
            updates = torch.einsum('bjd,bij->bid', v, attn)  # [bsz, num_slots, d_model].

            # Slot update.
            slots = slots_prev + updates
            slots = slots + self.mlp(self.norm_mlp(slots))

        return self.norm_out(slots)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()

        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     src_txt_global: Optional[Tensor] = None,):
                     
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # [#queries, batch_size, d_model]
        q_content = self.sa_qcontent_proj(tgt)
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        q = q_content + q_pos
        k = k_content + k_pos

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # [#queries, batch_size, d_model]
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        q = q_content
        k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                               key=k,
                               value=v, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]               
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoderFirstLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()

        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Gated Fusion
        self.gate_cross_attn = nn.MultiheadAttention(d_model, 1, dropout=0.1)
        self.gate_self_attn = nn.MultiheadAttention(d_model, 1, dropout=0.1)
        self.gate_dropout = nn.Dropout(dropout)
        self.gate_norm = nn.LayerNorm(d_model)
        self.gate_linear = nn.Linear(d_model, d_model)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     src_txt_global: Optional[Tensor] = None,):
        
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # [#queries, batch_size, d_model]
        q_content = self.sa_qcontent_proj(tgt)
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        q = q_content + q_pos
        k = k_content + k_pos

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)   # C'

        # ========== Begin of Gated Fusion =============
        
        tgt2 = self.gate_cross_attn(query=tgt,
                                    key=src_txt_global,
                                    value=src_txt_global)[0]  # \hat{C}
        gate = (tgt*tgt2).sigmoid()
        tgt2 = tgt+tgt2
        tgt2 = self.gate_self_attn(query=tgt2, 
                                   key=tgt2, 
                                   value=tgt2)[0]
        tgt = self.gate_dropout(self.activation(self.gate_linear(gate*(tgt2)))) + tgt

        # ========== End of Gated Fusion =============
        tgt = self.gate_norm(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # [#queries, batch_size, d_model]
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)
        q_pos = self.ca_qpos_proj(query_pos)

        q = q_content + q_pos
        k = k_content + k_pos

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                                key=k,
                                value=v, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0]               
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

# =====CHZ Part=====

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        
        output = src
        ats = None
        intermediate = []

        for layer in self.layers:
            output,tmp = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            
            if ats is None:
                ats = tmp
            else :
                ats += tmp
            if self.return_intermediate:
                intermediate.append(output)

        ats /= self.num_layers

        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output,ats


class CrossModalityEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                video_length=75):
        
        output = src
        ats = None

        intermediate = []

        for layer in self.layers:
            output,ats = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask,
                             pos=pos,video_length=video_length)
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output,ats
 

class CM_TransformerEncoderLayer(nn.Module):
    def __init__(self,d_model,nhead,dim_feedforward=2048,dropout=0.1,activation="relu"):
        super().__init__()
        self.nhead=nhead
        self.self_attn = nn.MultiheadAttention(d_model,nhead,dropout=dropout)

        self.linear1 = nn.Linear(d_model,dim_feedforward)
        self.activation = _get_activation_fn(activation)

        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward,d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def with_pos_embed(self,tensor,pos:Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self,src,src_mask:Optional[Tensor]=None,
                src_key_padding_mask:Optional[Tensor]=None,
                pos:Optional[Tensor]=None,video_length = 75):
        # src.shape 121,32,256 (L,bsz,d)
    
        pos_src = self.with_pos_embed(src,pos)
        
        global_token = src[0]
        q = pos_src[1:video_length+1]
        k = pos_src[video_length+1:]
        v = src[video_length+1:]
        # q = pos_src[:video_length]
        # k = pos_src[video_length:]
        # v = src[video_length:]
        qmask = src_key_padding_mask[: , 1:video_length+1].unsqueeze(2)
        kmask = src_key_padding_mask[: , video_length+1:].unsqueeze(1)
        
        attn_mask = torch.matmul(qmask.float(), kmask.float()).bool().repeat(self.nhead, 1, 1)
        
        src2,ats = self.self_attn(q,k,value=v,attn_mask=attn_mask,
                                  key_padding_mask=src_key_padding_mask[:, video_length+1:],need_weights=True)
        
        src2 = src[1:video_length+1] + self.dropout1(src2)
        # src2 = src[:video_length] + self.dropout1(src2)
        src3 = self.norm1(src2)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3)))) 
        src2 = src2 + self.dropout2(src3)
        src2 = self.norm2(src2)
        
        src = torch.cat([global_token.unsqueeze(0),src2,src[video_length+1:]])
        # src = torch.cat([src2,src[video_length:]])

        return src,ats


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.nhead=nhead

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_mask: Optional[Tensor] = None, # 根本就没指定mask
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
    
        q = k = self.with_pos_embed(src, pos)
        src2,ats = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask,need_weights=True)
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src,ats


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
