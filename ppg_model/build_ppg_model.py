import argparse
import torch
from pathlib import Path
import yaml


from .utterance_mvn import UtteranceMVN
from .vgg_rnn_encoder_bottle_neck import VGGRNNBNEncoder


class PPGModel(torch.nn.Module):
    def __init__(
        self,
        normalizer,
        encoder,
    ):
        super().__init__()
        self.normalize = normalizer
        self.encoder = encoder

    def forward(self, mel, mel_lengths):
        """

        Args:
            mel (tensor): (B, L, 80)
            mel_lengths (tensor): (B, )

        Returns:
            bottle_neck_feats (tensor): (B, L//4, 144)

        """
        feats, feats_lengths = self.normalize(mel, mel_lengths)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
        return encoder_out
        

def build_model(args):
    normalizer = UtteranceMVN(**args.normalize_conf)
    encoder = VGGRNNBNEncoder(input_size=80, **args.encoder_conf)
    
    model = PPGModel(normalizer, encoder)
    
    return model


def load_ppg_model(train_config, model_file, device):
    config_file = Path(train_config)
    with config_file.open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)

    args = argparse.Namespace(**args)

    model = build_model(args)
    model_state_dict = model.state_dict()

    ckpt_state_dict = torch.load(model_file, map_location='cpu')
    ckpt_state_dict = {k:v for k,v in ckpt_state_dict.items() if 'encoder' in k}

    model_state_dict.update(ckpt_state_dict)
    model.load_state_dict(model_state_dict)

    return model.eval().to(device)
