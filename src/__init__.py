from .mel_decoder_mol_encAddlf0 import MelDecoderMOL
from .mel_decoder_mol_v2 import MelDecoderMOLv2
from .rnn_ppg2mel import BiRnnPpg2MelModel
from .mel_decoder_lsa import MelDecoderLSA


def build_model(model_name: str):
    if model_name == "seq2seqmol":
        return MelDecoderMOL
    elif model_name == "seq2seqmolv2":
        return MelDecoderMOLv2
    elif model_name == "bilstm":
        return BiRnnPpg2MelModel
    elif model_name == "seq2seqlsa":
        return MelDecoderLSA
    else:
        raise ValueError(f"Unknown model name: {model_name}.")
