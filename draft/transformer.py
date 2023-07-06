import torch
from torch import nn as nn

# sample = torch.randn((64, 512, 32, 8))
# sample = torch.flatten(sample, 2).permute(0, 2, 1)
# input(sample.size())
# encoder_layer = nn.TransformerEncoderLayer(
#     d_model=512, nhead=6, batch_first=True)
# transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
# decoder_layer = nn.TransformerDecoderLayer(
#     d_model=512, nhead=6, batch_first=True)
# transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

# transformer = nn.Transformer(
#     d_model=512, nhead=8, num_encoder_layers=4, num_decoder_layers=4, batch_first=True)
# tgt = torch.randn((64, 256, 512))

# print(transformer(sample, tgt).size())


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model # 256
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1) # 91 + 1 -> 92
        self.bbox_embed = MLP(hidden_dim, hidden_dim, output_dim=4, num_layers=3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, features, pos):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # if isinstance(samples, (list, torch.Tensor)):
        #     samples = nested_tensor_from_tensor_list(samples)
        # features, pos = self.backbone(samples)  # torch.Size([2, 2048, 34, 34]) / torch.Size([2, 256, 34, 34])
        # features[0].tensors.shape = torch.Size([2, 2048, 28, 38]) features[0].mask 있음
        # pos[0].shape = torch.Size([2, 256, 28, 38])




        src= features
        hs = self.transformer(self.input_proj(src),  self.query_embed.weight, pos) 
        # hs = tuple(decoder 결과 : Size([6, 100, 2, 256]), encoder 결과 : torch.Size([2, 256, 28, 38]))[0]
        print(hs[0].size(), hs[1].size())
        print(hs[0][-1].size())
        return hs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]



class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape # batch size , channel
        print(f"query embed {query_embed.size()} pos_embed : {pos_embed.size()}")
        src = src.flatten(2).permute(2, 0, 1) # torch.Size([1064개의 vector, 2 batch, 256 vector dim]) 
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) # torch.Size([1064개의 vector, 2 batch, 256 vector dim])
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) # detr.py에서 검색해보면 init 되어있음. torch.Size([100개의 vector, 2, 256 vector dim]])
        # mask = mask.flatten(1) # torch.Size([2, 1064])

        tgt = torch.zeros_like(query_embed) # output 이 저장되는 공간 
        memory = self.encoder(src, pos=pos_embed) # torch.Size([1156, 2, 256])
        print(f"after encoding : {memory.size()}")
        # output의 shape가 가변적이기 때문에, 아래와 같이 tgt를 넘겨줌으로써 적절한 사이즈의 output 결과를 뽑아내게 한다. 
        hs = self.decoder(tgt, memory, 
                          pos=pos_embed, query_pos=query_embed) 
                          # torch.Size([6개의 decoder layer에서 나온 중간 결과(head아님), 100개의 vector, 2장, 256 vecter dim])
                          # query_embed는 Object queries
        print(f"after decoding : {hs.size()}")
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w) 


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for i, layer in enumerate(self.layers):
            output, score = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm # class Transformer: nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            output, score = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            print(f"decoding step {i} size {output.size()}")
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()          # 가장 마지막 6번째 layer에서 나온 결과 일단 버리고,
                intermediate.append(output) # LayerNorm한 값 넣음
                # intermediate.Size([100, 2, 256]) x 6개의 decoder layer에서 나온 모든 결과

        if self.return_intermediate:
            return torch.stack(intermediate) # torch.Size([6, 100, 2, 256])

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
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
        self.normalize_before = normalize_before
        # dropout은 임의로 노드를 잠궈버리는 것. 즉 Zero로 만들어 버리는 것이다. 
        # 따라서 dropout의 forward를 위한 파라미터는 노드를 넣어줘야한다. (아래 forward 구현부분 참조)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # 그냥 Element-wise-sum 해버리기 
        
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # 1. Image Self-Attention
        q = k = self.with_pos_embed(src, pos)
        src2, score = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # 2. Feed Forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, score

    # normalize_before
    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        # 1. Image Self-Attention
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos) # 그냥 더하기
        src2, score = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask) # attention score값이 [1]에 나오는데 그거 무시
        src = src + self.dropout1(src2)
        # 2. Feed Forward
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, score

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before: # false
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None): 
                     # Object queries와 같이 새로운 query를 넣어주고 싶다면 query_pos를 정의해주면 된다. 안 넣어주면 
        # 1. only Object queries self-attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # 2. Object queries and Encoder_output(memory) Self-attention
        tgt2, score = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # 3. Feed Forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, score

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        # 1. only Object queries self-attention    
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        # 2. Object queries and Encoder_output(memory) Self-attention
        tgt2 = self.norm2(tgt)
        tgt2, score = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        # 3. Feed Forward
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, score

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")




def build() :
    num_classes = 4 # num_slots
    
    transformer = Transformer(
        d_model=512,
        dropout=0.1,
        nhead=8,
        dim_feedforward=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        normalize_before=False,
        return_intermediate_dec=True,
    )
    
    
    model = DETR(
        backbone=None,
        transformer=transformer,
        num_classes=num_classes,
        num_queries=4,
        aux_loss=False,
    )
    
    return model



class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list):
        x = tensor_list
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos
    
if __name__ == "__main__" :
    # model = build()
    sample = torch.randn((64, 512, 32, 8))
    model = build()
    pos = PositionEmbeddingLearned(256)
    model(sample,pos(sample))
    
#output size : [64,4,512]