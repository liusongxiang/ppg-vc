"""Sequential implementation of Recurrent Neural Network Duration Model."""
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
from typeguard import check_argument_types


class BiRnnPpg2MelModel(torch.nn.Module):
    """ Bidirectional RNN-based PPG-to-Mel Model for voice conversion tasks.
        RNN could be LSTM-based or GRU-based.
    """
    def __init__(
        self,
        input_size: int, 
        multi_spk: bool = False,    
        num_speakers: int = 1,
        spk_embed_dim: int = 256,
        use_spk_dvec: bool = False,
        multi_styles: bool =  False,
        num_styles: int = 3,
        style_embed_dim: int = 256,
        dense_layer_size: int = 256,
        num_layers: int = 4,
        bidirectional: bool = True,
        hidden_dim: int = 256,
        dropout_rate: float = 0.5,
        output_size: int = 80,
        rnn_type: str = "lstm"
    ):
        assert check_argument_types()
        super().__init__()

        self.multi_spk = multi_spk
        self.spk_embed_dim = spk_embed_dim
        self.use_spk_dvec= use_spk_dvec
        self.multi_styles = multi_styles
        self.style_embed_dim = style_embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        
        self.ppg_dense_layer = nn.Linear(input_size - 2, hidden_dim)
        self.logf0_uv_layer = nn.Linear(2, hidden_dim)

        projection_input_size = hidden_dim
        if self.multi_spk:
            if not self.use_spk_dvec:         
                self.spk_embedding = nn.Embedding(num_speakers, spk_embed_dim)
            projection_input_size += self.spk_embed_dim
        if self.multi_styles:
            self.style_embedding = nn.Embedding(num_styles, style_embed_dim)
            projection_input_size += self.style_embed_dim

        self.reduce_proj = nn.Sequential(
            nn.Linear(projection_input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        rnn_type = rnn_type.upper()
        if rnn_type in ["LSTM", "GRU"]:
            rnn_class = getattr(nn, rnn_type)
            self.rnn = rnn_class(
                hidden_dim, hidden_dim, num_layers, 
                bidirectional=bidirectional,
                dropout=dropout_rate,
                batch_first=True)
        else:
            # Default: use BiLSTM
            self.rnn = nn.LSTM(
                hidden_dim, hidden_dim, num_layers, 
                bidirectional=bidirectional,
                dropout_rate=dropout_rate,
                batch_first=True)
        # Fully connected layers
        self.hidden2out_layers = nn.Sequential(
            nn.Linear(self.num_direction * hidden_dim, dense_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_layer_size, output_size)
        )
    
    def forward(
        self, 
        ppg: torch.Tensor,
        ppg_lengths: torch.Tensor,
        logf0_uv: torch.Tensor,
        spembs: torch.Tensor = None,
        styleembs: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            ppg (tensor): [B, T, D_ppg]
            ppg_lengths (tensor): [B,]
            logf0_uv (tensor): [B, T, 2], concatented logf0 and u/v flags.
            spembs (tensor): [B,] index-represented speaker.
            styleembs (tensor): [B,] index-repreented speaking style (e.g. emotion). 
        """
        ppg = self.ppg_dense_layer(ppg)
        logf0_uv = self.logf0_uv_layer(logf0_uv)

        ## Concatenate/add ppg and logf0_uv
        x = ppg + logf0_uv
        B, T, _ = x.size()

        if self.multi_spk:
            assert spembs is not None
            # spk_embs = self.spk_embedding(torch.LongTensor([0,]*ppg.size(0)).to(ppg.device))
            if not self.use_spk_dvec:
                spk_embs = self.spk_embedding(spembs)
                spk_embs = torch.nn.functional.normalize(
                    spk_embs).unsqueeze(1).expand(-1, T, -1)
            else:
                spk_embs = torch.nn.functional.normalize(
                    spembs).unsqueeze(1).expand(-1, T, -1)
            x = torch.cat([x, spk_embs], dim=2)
        
        if self.multi_styles and styleembs is not None:
            style_embs = self.style_embedding(styleembs)
            style_embs = torch.nn.functional.normalize(
                style_embs).unsqueeze(1).expand(-1, T, -1)
            x = torch.cat([x, style_embs], dim=2)
        ## FC projection
        x = self.reduce_proj(x)

        if ppg_lengths is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, ppg_lengths,
                                                       batch_first=True,
                                                       enforce_sorted=False)
        x, _ = self.rnn(x)
        if ppg_lengths is not None:
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.hidden2out_layers(x)
        
        return x
