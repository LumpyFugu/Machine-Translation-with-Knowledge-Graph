import torch

from fairseq.tokenizer import tokenize_line
from fairseq.data import Dictionary, LanguagePairDataset
from fairseq.data import data_utils

'''
# Merge normal dictionary and entity dictionary for unified indexing/management, so plain tokens and named entities are handled together.
# Supports common operations such as indexing, lookup, string conversion, and encoding.
# Makes it convenient to handle regular tokens and entity tokens together in NLP tasks.
'''

class LangWithEntityDictionary(object):
    def __init__(self, lang_dict: Dictionary, ne_dict: Dictionary):
        self.lang_dict = lang_dict# language dictionary
        self.ne_dict = ne_dict# entity dictionary

    def __eq__(self, other):#
        return self.lang_dict == other.lang_dict and self.ne_dict == other.ne_dict

    def __getitem__(self, idx):
        if idx < len(self.lang_dict):
            return self.lang_dict[idx]
        return self.ne_dict[idx - len(self.lang_dict)]

    def __len__(self):
        return len(self.lang_dict) + len(self.ne_dict)

    def __contains__(self, sym):
        return sym in self.lang_dict or sym in self.ne_dict

    def index(self, sym):
        if sym in self.lang_dict:
            return self.lang_dict.index(sym)
        return self.ne_dict.index(sym) + len(self.lang_dict)

    def string(self, tensor, bpe_symbol=None, escape_unk=False, extra_symbols_to_ignore=None):
    # Support 2D tensor: each row is one sentence
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(
                self.string(t, bpe_symbol, escape_unk, extra_symbols_to_ignore)
                for t in tensor
            )

        if extra_symbols_to_ignore is None:
            extra_symbols_to_ignore = set()
        else:
            # Convert to set for faster checks
            extra_symbols_to_ignore = set(extra_symbols_to_ignore)

        def token_string(i):
            if i == self.unk():
                return self.unk_string(escape_unk)
            else:
                return self[i]

        # Keep original logic: drop <eos> and (if present) <bos>,
        # and skip symbols in extra_symbols_to_ignore
        if hasattr(self, 'bos_index'):
            sent = ' '.join(
                token_string(i)
                for i in tensor
                if (i != self.eos())
                and (i != self.bos())
                and (i not in extra_symbols_to_ignore)
            )
        else:
            sent = ' '.join(
                token_string(i)
                for i in tensor
                if (i != self.eos())
                and (i not in extra_symbols_to_ignore)
            )

    # Replacement for the original data_utils.process_bpe_symbol

    # Return directly if bpe_symbol is not set or is sentencepiece
        if bpe_symbol is None or bpe_symbol == "sentencepiece":
            return sent
        else:
            # For BPE marks like "@@ " or "@@", just strip them
            return sent.replace(bpe_symbol, "")


    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return '<{}>'.format(self.lang_dict.unk_word)
        else:
            return self.lang_dict.unk_word

    def bos(self):
        return self.lang_dict.bos_index

    def pad(self):
        return self.lang_dict.pad_index

    def eos(self):
        return self.lang_dict.eos_index

    def unk(self):
        return self.lang_dict.unk_index

    def encode_line(self, line, line_tokenizer=tokenize_line, add_if_not_exist=True,
                    consumer=None, append_eos=True, reverse_order=False):
        return self.lang_dict.encode_line(line, line_tokenizer=tokenize_line, add_if_not_exist=True,
                                          consumer=None, append_eos=True, reverse_order=False)
