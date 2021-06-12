import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='none')

    def get_mask(self, lengths):
        # lengths: [B,]
        max_len = torch.max(lengths)
        batch_size = lengths.size(0)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len).to(lengths.device)
        seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand)
        return (seq_range_expand < seq_length_expand).float()

    def forward(self, mel_pred, mel_trg, lengths):
        # (B, T, 1)
        mask = self.get_mask(lengths).unsqueeze(-1)
        # (B, T, D)
        mask_ = mask.expand_as(mel_trg)
        loss = self.loss(mel_pred, mel_trg)
        return ((loss * mask_).sum()) / mask_.sum()
