#!/usr/bin/env python3

# Copyright 2020 Songxiang Liu
# Apache 2.0

from typing import Dict
from typing import Tuple
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import check_argument_types

import math
import numpy as np

from .abs_model import AbsMelDecoder
from .basic_layers import Linear, Conv1d
from .rnn_decoder_mol import Decoder
from .cnn_postnet import Postnet
from .vc_utils import get_mask_from_lengths


class MelDecoderMOL(AbsMelDecoder):
    def __init__(
        self,
        num_speakers: int,
        spk_embed_dim: int = 256,
        bottle_neck_feature_dim: int = 1024,
        attention_rnn_dim: int = 256,
        attention_dim: int = 256,
        decoder_rnn_dim: int = 256,
        num_decoder_rnn_layer: int = 1,
        concat_context_to_last: bool = True,
        prenet_dims: List = [256, 128],
        prenet_dropout: float = 0.5,
        num_mixtures: int = 5,
        frames_per_step: int = 2,
        postnet_num_layers: int = 5,
        postnet_hidden_dim: int = 512,
        mask_padding: bool = True,
        use_bnf_prenet: bool = False,
        use_pitch_info: bool = False,
        pitch_embed_dim: int = 256,
        use_spk_dvec: bool = False,
        use_instance_norm: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        
        self.mask_padding = mask_padding
        self.bottle_neck_feature_dim = bottle_neck_feature_dim
        self.num_mels = 80
        self.frames_per_step = frames_per_step
        self.multi_speaker = True if num_speakers > 1 or self.use_spk_dvec else False
        self.use_bnf_prenet = use_bnf_prenet
        self.use_pitch_info = use_pitch_info
        self.use_spk_dvec = use_spk_dvec
        self.use_instance_norm = use_instance_norm

        input_dim = bottle_neck_feature_dim

        if self.use_bnf_prenet:
            self.bnf_prenet = torch.nn.GRU(input_dim,
                                           256//2,
                                           num_layers=2,
                                           bidirectional=True,
                                           batch_first=True)
            if self.use_instance_norm:
                self.norm_layer = torch.nn.InstanceNorm1d(256, affine=False)
        decoder_enc_dim = 256
        if self.use_pitch_info:
            # Downsampling convolution
            if self.use_instance_norm:
                self.pitch_convs = torch.nn.Sequential(
                    torch.nn.Conv1d(
                        2, pitch_embed_dim, kernel_size=3, stride=1, 
                        padding=1, bias=False),
                    torch.nn.ReLU(),
                    torch.nn.InstanceNorm1d(pitch_embed_dim, affine=False),
                    torch.nn.Conv1d(
                        pitch_embed_dim, pitch_embed_dim, kernel_size=3, stride=1, 
                        padding=1, bias=False),
                    torch.nn.ReLU(),
                    torch.nn.InstanceNorm1d(pitch_embed_dim, affine=False),
                )
            else:
                self.pitch_convs = torch.nn.Sequential(
                    torch.nn.Conv1d(
                        2, pitch_embed_dim, kernel_size=3, stride=1, 
                        padding=1, bias=False),
                    torch.nn.ReLU(),
                    torch.nn.Conv1d(
                        pitch_embed_dim, pitch_embed_dim, kernel_size=3, stride=1, 
                        padding=1, bias=False),
                    torch.nn.ReLU(),
                )
        
        if self.multi_speaker:
            if not self.use_spk_dvec:
                self.speaker_embedding_table = nn.Embedding(num_speakers, spk_embed_dim)
            self.reduce_proj = torch.nn.Linear(256 + spk_embed_dim, 256)
            # decoder_enc_dim += spk_embed_dim 

        # Decoder
        self.decoder = Decoder(
            enc_dim=decoder_enc_dim,
            num_mels=self.num_mels,
            frames_per_step=frames_per_step,
            attention_rnn_dim=attention_rnn_dim,
            decoder_rnn_dim=decoder_rnn_dim,
            num_decoder_rnn_layer=num_decoder_rnn_layer,
            prenet_dims=prenet_dims,
            num_mixtures=num_mixtures,
            use_stop_tokens=True,
            concat_context_to_last=concat_context_to_last,
        )

        # Mel-Spec Postnet: some residual CNN layers
        self.postnet = Postnet()
    
    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths, outputs[0].size(1))
            mask = mask.unsqueeze(2).expand(mask.size(0), mask.size(1), self.num_mels)
            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
        return outputs

    def forward(
        self,
        bottle_neck_features: torch.Tensor,
        feature_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        logf0_uv: torch.Tensor = None,
        spembs: torch.Tensor = None,
        styleembs: torch.Tensor = None,
        output_att_ws: bool = False,
    ):
        if self.use_bnf_prenet:
            decoder_inputs, _ = self.bnf_prenet(bottle_neck_features)
            if self.use_instance_norm:
                decoder_inputs = self.norm_layer(decoder_inputs.transpose(1,2)).transpose(1,2)
        if self.use_pitch_info:
            logf0_uv = self.pitch_convs(logf0_uv.transpose(1, 2)).transpose(1, 2)
            # decoder_inputs = torch.cat([decoder_inputs, logf0_uv], dim=-1) 
            decoder_inputs = decoder_inputs + logf0_uv
            
        if self.multi_speaker:
            assert spembs is not None
            if not self.use_spk_dvec:
                spk_embeds = self.speaker_embedding_table(spembs)
                spk_embeds = F.normalize(
                    spk_embeds).unsqueeze(1).expand(-1, bottle_neck_features.size(1), -1)
            else:
                spk_embeds = F.normalize(
                    spembs).unsqueeze(1).expand(-1, bottle_neck_features.size(1), -1)
            decoder_inputs = torch.cat([decoder_inputs, spk_embeds], dim=-1)
            decoder_inputs = self.reduce_proj(decoder_inputs)
        
        # (B, num_mels, T_dec)
        mel_outputs, predicted_stop, alignments = self.decoder(
            decoder_inputs, speech, feature_lengths)
        ## Post-processing
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(1, 2)).transpose(1, 2)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        if output_att_ws: 
            return self.parse_output(
                [mel_outputs, mel_outputs_postnet, predicted_stop, alignments], speech_lengths)
        else:
            return self.parse_output(
                [mel_outputs, mel_outputs_postnet, predicted_stop], speech_lengths)

        # return mel_outputs, mel_outputs_postnet

    def inference(
        self,
        bottle_neck_features: torch.Tensor,
        logf0_uv: torch.Tensor = None,
        spembs: torch.Tensor = None,
        use_stop_tokens: bool = True,
    ):
        if self.use_bnf_prenet:
            bottle_neck_features, _ = self.bnf_prenet(bottle_neck_features)
            if self.use_instance_norm:
                decoder_inputs = self.norm_layer(bottle_neck_features.transpose(1,2)).transpose(1,2)
                # self.norm_layer(bottle_neck_features.transpose(1,2)).transpose(1,2)
        if self.use_pitch_info:
            logf0_uv = self.pitch_convs(logf0_uv.transpose(1, 2)).transpose(1, 2)
            bottle_neck_features = bottle_neck_features + logf0_uv
        if self.multi_speaker:
            assert spembs is not None
            # spk_embeds = self.speaker_embedding_table(spembs)
            # spk_embeds = F.normalize(
                # spk_embeds).unsqueeze(1).expand(-1, bottle_neck_features.size(1), -1)
            if not self.use_spk_dvec:
                spk_embeds = self.speaker_embedding_table(spembs)
                spk_embeds = F.normalize(
                    spk_embeds).unsqueeze(1).expand(-1, bottle_neck_features.size(1), -1)
            else:
                spk_embeds = F.normalize(
                    spembs).unsqueeze(1).expand(-1, bottle_neck_features.size(1), -1)
            bottle_neck_features = torch.cat([bottle_neck_features, spk_embeds], dim=-1)
            bottle_neck_features = self.reduce_proj(bottle_neck_features)

        ## Decoder
        if bottle_neck_features.size(0) > 1:
            mel_outputs, alignments = self.decoder.inference_batched(bottle_neck_features)
        else:
            mel_outputs, alignments = self.decoder.inference(bottle_neck_features,)
        ## Post-processing
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(1, 2)).transpose(1, 2)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        # outputs = mel_outputs_postnet[0]
        
        return mel_outputs[0], mel_outputs_postnet[0], alignments[0]
