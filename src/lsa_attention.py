import torch
from torch import nn
from torch.nn import functional as F
from .basic_layers import Linear, Conv1d



class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super().__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = Conv1d(2, attention_n_filters,
                                    kernel_size=attention_kernel_size,
                                    padding=padding, bias=False,
                                    stride=1, dilation=1)
        self.location_dense = Linear(attention_n_filters, attention_dim,
                                     bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention

class LocationSensitiveAttention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super().__init__()
        self.query_layer = Linear(attention_rnn_dim, attention_dim,
                                  bias=False, w_init_gain='tanh')
        self.memory_layer = Linear(embedding_dim, attention_dim,
                                   bias=False, w_init_gain='tanh')
        self.v = Linear(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float('inf')

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        Args:
            query: attention rnn output (B, att_rnn_dim)
            processed_memory: processed encoder outputs (B, T_in, enc_dim)
            attention_weights_cat: cumulative and previous attention weights (B, 2, T_in)
        Returns:
            attention alignment (B, T_in)
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))
        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory, attention_weights_cat, mask):
        """
        Args:
            attention_hidden_state: attention rnn last output
            memory: encoder outputs
            processed_memory: processed encoder outputs
            attention_weights_cat: previous and cumulative attention weights
            mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class ForwardAttentionV2(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(ForwardAttentionV2, self).__init__()
        self.query_layer = Linear(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = Linear(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = Linear(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float(1e20)

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat:  prev. and cumulative att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask, log_alpha):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        log_energy = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        #log_energy = 

        if mask is not None:
            log_energy.data.masked_fill_(mask, self.score_mask_value)

        #attention_weights = F.softmax(alignment, dim=1)

        #content_score = log_energy.unsqueeze(1) #[B, MAX_TIME] -> [B, 1, MAX_TIME]
        #log_alpha = log_alpha.unsqueeze(2) #[B, MAX_TIME] -> [B, MAX_TIME, 1]

        #log_total_score = log_alpha + content_score

        #previous_attention_weights = attention_weights_cat[:,0,:]
        
        log_alpha_shift_padded = []
        max_time = log_energy.size(1)
        for sft in range(2):
            shifted = log_alpha[:,:max_time-sft]
            shift_padded = F.pad(shifted, (sft,0), 'constant', self.score_mask_value)
            log_alpha_shift_padded.append(shift_padded.unsqueeze(2))

        biased = torch.logsumexp(torch.cat(log_alpha_shift_padded,2), 2)
        
        log_alpha_new = biased +  log_energy

        attention_weights =  F.softmax(log_alpha_new, dim=1)

        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights, log_alpha_new

