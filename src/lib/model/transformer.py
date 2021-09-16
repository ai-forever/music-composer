import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import random

from lib.constants import *
from .positional_encoding import PositionalEncoding
from .rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR


# MusicTransformer
class MusicTransformer(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Music Transformer reproduction from https://arxiv.org/abs/1809.04281. Arguments allow for
    tweaking the transformer architecture (https://arxiv.org/abs/1706.03762) and the rpr argument
    toggles Relative Position Representations (RPR - https://arxiv.org/abs/1803.02155).

    Supports training and generation using Pytorch's nn.Transformer class with dummy decoder to
    make a decoder-only transformer architecture

    For RPR support, there is modified Pytorch 1.2.0 code in rpr.py. Modified source will be
    kept up to date with Pytorch revisions only as necessary.
    ----------
    """

    def __init__(self, device, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1,
                 max_sequence=2048, rpr=False, vocab_size=VOCAB_SIZE, cond_vocab_size=None, reduce_qk=False):
        super(MusicTransformer, self).__init__()

        self.vocab_size = vocab_size
        self.cond_vocab_size = cond_vocab_size
        self.device = device
        self.dummy      = DummyDecoder()

        self.nlayers    = n_layers
        self.nhead      = num_heads
        self.d_model    = d_model
        self.d_ff       = dim_feedforward
        self.dropout    = dropout
        self.max_seq    = max_sequence
        self.rpr        = rpr
        self.reduce_qk  = reduce_qk

        # Input embedding
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        if self.cond_vocab_size is not None:
            self.cond_embedding = nn.Embedding(self.cond_vocab_size, self.d_model)
        else:
            self.cond_embedding = None

        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)

        # Base transformer
        if(not self.rpr):
            # To make a decoder-only transformer we need to use masked encoder layers
            # Dummy decoder to essentially just return the encoder output
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout, dim_feedforward=self.d_ff, custom_decoder=self.dummy
            )
        # RPR Transformer
        else:
            encoder_norm = LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout,
                                                       er_len=self.max_seq, reduce_qk=self.reduce_qk, device=self.device)
            encoder = TransformerEncoderRPR(encoder_layer, self.nlayers, encoder_norm)
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout, dim_feedforward=self.d_ff, custom_decoder=self.dummy,
                custom_encoder=encoder
            )

        # Final output is a softmaxed linear layer
        self.Wout       = nn.Linear(self.d_model, self.vocab_size)
        self.softmax    = nn.Softmax(dim=-1)
        self.mask = self.transformer.generate_square_subsequent_mask(max_sequence).to(self.device)

    # forward
    def forward(self, x, condition=None, mask=True):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Takes an input sequence and outputs predictions using a sequence to sequence method.

        A prediction at one index is the "next" prediction given all information seen previously.
        ----------
        """
        if (mask is True):
            mask = self.mask[..., :x.shape[1], :x.shape[1]]
        else:
            mask = None
        x = self.embedding(x)
        if condition is not None and self.cond_embedding is not None:
            x_cond = self.cond_embedding(condition)
            x = x + x_cond[:, None]

        # Input shape is (max_seq, batch_size, d_model)
        x = x.permute(1,0,2)

        x = self.positional_encoding(x)

        # Since there are no true decoder layers, the tgt is unused
        # Pytorch wants src and tgt to have some equal dims however
        x_out = self.transformer(src=x, tgt=x, src_mask=mask)

        # Back to (batch_size, max_seq, d_model)
        x_out = x_out.permute(1,0,2)

        y = self.Wout(x_out)

        # They are trained to predict the next note in sequence (we don't need the last one)
        return y

    def get_norms(self):
        norm_dict = {'embedding_weight_norm': torch.norm(self.embedding.weight).item(),
                     'embedding_grad_norm': torch.norm(self.embedding.weight.grad).item(),
                     'output_weight_norm': torch.norm(self.Wout.weight).item(),
                     'output_grad_norm': torch.norm(self.Wout.weight.grad).item()}
        return norm_dict

    def get_parameters(self):
        return {'device': self.device,
                'n_layers': self.nlayers,
                'num_heads': self.nhead,
                'd_model': self.d_model,
                'dim_feedforward': self.d_ff,
                'dropout': self.dropout,
                'max_sequence': self.max_seq,
                'rpr': self.rpr,
                'vocab_size': self.vocab_size,
                'cond_vocab_size': self.cond_vocab_size,
                'reduce_qk': self.reduce_qk,
        }

# Used as a dummy to nn.Transformer
# DummyDecoder
class DummyDecoder(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    A dummy decoder that returns its input. Used to make the Pytorch transformer into a decoder-only
    architecture (stacked encoders with dummy decoder fits the bill)
    ----------
    """

    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask,tgt_key_padding_mask,memory_key_padding_mask):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Returns the input (memory)
        ----------
        """

        return memory
