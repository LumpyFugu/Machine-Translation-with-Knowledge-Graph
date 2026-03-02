from typing import List
import torch

from fairseq.tokenizer import tokenize_line
from fairseq.data import Dictionary, FairseqDataset, LanguagePairDataset, data_utils
from collections import defaultdict

from .entity_dictionary import LangWithEntityDictionary
from .utils import combine_ne_with_text, tag_entity

import random

"""
For the training time, we need to consider two things
1. What to put in 'net_input'
    Method 1, adjacent tokens in x are replaced by placeholder.
    Method 2, raw x.
    Method 3, raw x.
2. What other things need to compute the loss.
    Method 1, adjacent tokens in y are replaced by placeholder.
    Method 2, x entity sequence, y entity sequence, y tokens
    Method 3, x entity sequence, y entity sequence (adjacent tokens combined), BERT processed entity (should be provided by task, not here).
"""

'''
This file defines EntityTranslationDataset, a custom Fairseq dataset (inheriting from FairseqDataset) for entity-enhanced MT tasks.
It handles both plain text and named-entity labels, and organizes batches by training mode.
'''

class EntityTranslationDataset(FairseqDataset):
    def __init__(self,
                 lang_pair: LanguagePairDataset,
                 ne_pair: LanguagePairDataset,
                 mode: int,
                 max_ne_id: int,
                 src_dict: LangWithEntityDictionary,
                 tgt_dict: LangWithEntityDictionary,
                 ignore_entity_type: List[str],
                 ne_drop_rate: float,
                 is_train: bool):
        self.lang_pair = lang_pair
        self.ne_pair = ne_pair
        self.mode = mode
        self.max_ne_id = max_ne_id
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        assert len(self.lang_pair) == len(self.ne_pair), f'Language data lenth {len(self.lang_pair)} != NE data length {len(self.ne_pair)}'

        self.collater_methods = [
            self.collater_mode_0,
            self.collater_mode_1,
            self.collater_mode_2,
            self.collater_mode_3,
            self.collater_mode_4,
            self.collater_mode_5
        ]

        assert 0 <= self.mode < len(self.collater_methods), f'Invalid mode {self.mode}'

        self.ignore_entity_type = ignore_entity_type
        self.ignore_entity_idx = [self.tgt_dict.ne_dict.index(f'B-{x}') for x in ignore_entity_type]
        self.o_index = self.tgt_dict.ne_dict.index('O')

        self.ne_drop_rate = ne_drop_rate

        assert 0 <= self.ne_drop_rate <= 1, f'{self.ne_drop_rate}'

        self.is_train = is_train


    def _sanitize_merged_tokens(self, t: torch.Tensor, lang_dict: LangWithEntityDictionary) -> torch.Tensor:
        """
        t: 1D tensor (source token sequence with merged NE ids)
        lang_dict: LangWithEntityDictionary (contains lang_dict and ne_dict)

        Clamp all ids >= len(lang_dict) to len(lang_dict)-1;
        Only affects the high-id NE segment, not the normal vocab segment.
        Also prints the first bad position for debugging.
        """
    # language vocab size and final embedding size
        lang_vocab = len(lang_dict.lang_dict)   # e.g. 18056
        num_emb   = len(lang_dict)              # e.g. 18080 (= 18056 + 24)
        bad = (t >= num_emb) | (t < 0)
        if bad.any():
            bad_pos = torch.nonzero(bad, as_tuple=False)
            b, = bad_pos[0]
            head = t.tolist()[:50]
            print(f"[SANITIZE] num_emb={num_emb}, first_bad_pos={int(b)}, token={int(t[b].item())}, head={head}")
            # Clamp only NE segment (ids >= language-vocab start)
            ne_zone = t >= lang_vocab
            t[ne_zone] = t[ne_zone].clamp(min=lang_vocab, max=num_emb - 1)
            # Final fallback (extreme case)
            t = t.clamp(min=0, max=num_emb - 1)
        return t

    def _ignore_entity(self, tensor):
        mask = torch.zeros_like(tensor, dtype=torch.bool, device=tensor.device)
        o = torch.ones_like(tensor, dtype=tensor.dtype, device=tensor.device) * self.o_index

        for x in self.ignore_entity_idx:
            mask |= tensor == x  # B-XX
            mask |= tensor == (x + 1) # I-XX
        
        return torch.where(mask, o, tensor)

    def _drop_entity(self, tensor, eos):
        no_entity = torch.ones_like(tensor, dtype=tensor.dtype, device=tensor.device) * self.o_index
        mask = tensor != eos
        return torch.where(mask, no_entity, tensor)

    def __getitem__(self, index):
        """
        {   id: index,
            lang_pair:
            {
                'id': index,
                'source': source,
                'target': target,
            },
            ne_pair:
            {
                'id': index,
                'source': source,
                'target': target,
            }
        }
        """
        ne_pair = self.ne_pair[index]
        ne_pair['source'] = self._ignore_entity(ne_pair['source'])
        ne_pair['target'] = self._ignore_entity(ne_pair['target'])
        entity_sent_mask = True

        if self.is_train:
            entity_sent_mask = random.random() >= self.ne_drop_rate

            if not entity_sent_mask: # Not a entity sentence, only keep 'O' during the training
                ne_pair['source'] = self._drop_entity(ne_pair['source'], self.src_dict.eos())
                ne_pair['target'] = self._drop_entity(ne_pair['target'], self.tgt_dict.eos())

        return {
            'id': index,
            'lang_pair': self.lang_pair[index],
            'ne_pair': ne_pair,
            'entity_sent_mask': entity_sent_mask
        }

    def __len__(self):
        return len(self.lang_pair)

    def collater_mode_0(self, samples):
        """
        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
            'target': target,

            # below are custom items
            'src_ne_pos': [None, Slice, xxx],
            'origin_src': '' # the text before merge ne
            'origin_tgt': ''
        }
        """
        combined_lang_pair = []
        src_ne_pos_map = {}
        origin_sent_map = {
            'source': {},
            'target': {}
        }
        for sample in samples:
            sample_id = sample['id']
            combined_src_tokens, _, src_alignment = combine_ne_with_text(sample['lang_pair']['source'], sample['ne_pair']['source'], self.src_dict, self.max_ne_id)
            combined_tgt_tokens, _, _ = combine_ne_with_text(sample['lang_pair']['target'], sample['ne_pair']['target'], self.tgt_dict, self.max_ne_id)
            # Apply one range correction before stacking (only touches merged high-id NE tokens)
            combined_src_tokens = [self._sanitize_merged_tokens(x, self.src_dict) for x in combined_src_tokens]    
            combined_tgt_tokens = [self._sanitize_merged_tokens(x, self.tgt_dict) for x in combined_tgt_tokens]

            combined_lang_pair.append({
                'id': sample_id,
                'source': torch.stack(combined_src_tokens),
                'target': torch.stack(combined_tgt_tokens)
            })
            src_ne_pos_map[sample_id] = src_alignment

            origin_sent_map['source'][sample_id] = sample['lang_pair']['source']
            origin_sent_map['target'][sample_id] = sample['lang_pair']['target']

        batch = self.lang_pair.collater(combined_lang_pair)
        batch_ids = batch['id'].tolist()

        batch['src_ne_pos'] = [src_ne_pos_map[x] for x in batch_ids]
        batch['origin_src'] = [origin_sent_map['source'][x] for x in batch_ids]
        batch['origin_tgt'] = [origin_sent_map['target'][x] for x in batch_ids]

        return batch

    def collater_mode_1(self, samples):
        """
        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
            'target': target,

            'ne_pair': ne_pair,
            'ne_source': [],
            'ne_target': []
        }
        """
        batch = self.lang_pair.collater([sample['lang_pair'] for sample in samples])

        id2ne = {sample['id']: sample['ne_pair'] for sample in samples}
        batch_ids = batch['id'].tolist()

        batch['ne_pair'] = [id2ne[x] for x in batch_ids]
        batch['ne_source'] = data_utils.collate_tokens(
            [s['source'] for s in batch['ne_pair']],
            self.ne_pair.src_dict.pad(), self.ne_pair.src_dict.eos(), self.ne_pair.left_pad_source
        )
        batch['ne_target'] = data_utils.collate_tokens(
            [s['target'] for s in batch['ne_pair']],
            self.ne_pair.tgt_dict.pad(), self.ne_pair.tgt_dict.eos(), self.ne_pair.left_pad_target
        )
        return batch

    def collater_mode_2(self, samples):
        """
        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
            'target': target,

            'ne_pair': ne_pair,
            'ne_source':
            'tgt_ne_pos':
            'origin_tgt':
            'entity_sent_mask'
        }
        """
        combined_lang_pair = []
        tgt_ne_pos_map = {}
        origin_tgt_map = {}
        entity_sent_mask_map = {}
        for sample in samples:
            sample_id = sample['id']

            combined_tgt_tokens, _, tgt_alignment = combine_ne_with_text(sample['lang_pair']['target'], sample['ne_pair']['target'], self.tgt_dict, self.max_ne_id)
            combined_lang_pair.append({
                'id': sample_id,
                'source': sample['lang_pair']['source'],
                'target': torch.stack(combined_tgt_tokens)
            })
            tgt_ne_pos_map[sample_id] = tgt_alignment

            origin_tgt_map[sample_id] = sample['lang_pair']['target']
            entity_sent_mask_map[sample_id] = sample['entity_sent_mask']

        batch = self.lang_pair.collater(combined_lang_pair)
        id2ne = {sample['id']: sample['ne_pair'] for sample in samples}
        batch_ids = batch['id'].tolist()
        batch['ne_pair'] = [id2ne[x] for x in batch_ids]
        batch['ne_source'] = data_utils.collate_tokens(
            [s['source'] for s in batch['ne_pair']],
            self.ne_pair.src_dict.pad(), self.ne_pair.src_dict.eos(), self.ne_pair.left_pad_source
        )
        batch['ne_target'] = data_utils.collate_tokens(
            [s['target'] for s in batch['ne_pair']],
            self.ne_pair.tgt_dict.pad(), self.ne_pair.tgt_dict.eos(), self.ne_pair.left_pad_target
        )

        batch['tgt_ne_pos'] = [tgt_ne_pos_map[x] for x in batch_ids]
        batch['origin_tgt'] = [origin_tgt_map[x] for x in batch_ids]

        batch['entity_sent_mask'] = torch.BoolTensor([entity_sent_mask_map[x] for x in batch_ids], device=batch['net_input']['src_tokens'].device)
        return batch

    def collater_mode_3(self, samples):
        data = []
        for sample in samples:
            src = tag_entity(sample['lang_pair']['source'], sample['ne_pair']['source'], self.src_dict)
            tgt = tag_entity(sample['lang_pair']['target'], sample['ne_pair']['target'], self.tgt_dict)
            data.append({
                'id': sample['id'],
                'source': src,
                'target': tgt
            })

        return self.lang_pair.collater(data)

    def collater_mode_4(self, samples):
        return self.collater_mode_1(samples)
    
    def collater_mode_5(self, samples):
        return self.collater_mode_1(samples)

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        return self.collater_methods[self.mode](samples)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.lang_pair.num_tokens(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        lang_pair_size = self.lang_pair.size(index)
        # The NE should have same size as src
        return (lang_pair_size[0], lang_pair_size[1], lang_pair_size[0])

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return self.lang_pair.ordered_indices()

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return self.lang_pair.supports_prefetch and self.ne_pair.supports_prefetch

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        self.lang_pair.prefetch(indices)
        self.ne_pair.prefetch(indices)
