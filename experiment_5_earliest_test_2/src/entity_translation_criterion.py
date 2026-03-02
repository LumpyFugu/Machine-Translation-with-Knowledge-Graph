import math

import torch

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss
from torch.nn import functional as F

'''
This file defines loss functions for entity-aware translation, including:
1. entity label-smoothed cross-entropy loss
2. entity loss
3. translation loss
4. entity-enhanced loss
'''


@register_criterion('entity_label_smoothed_cross_entropy')
class EntityLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    '''
    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.entity_eps = args.entity_label_smoothing
        self.mode = args.mode
        self.tgt_ne_start_id = task.tgt_ne_start_id

        self.forward_methods = [
            self.forward_0,
            self.forward_1,
            self.forward_2,
            self.forward_3,
            self.forward_4,
            self.forward_5
        ]
 
    def __init__(self, cfg, task):
       # New FairseqCriterion only accepts task
        super().__init__(task)
        self.args = cfg
        self.eps = getattr(cfg, "label_smoothing", 0.1)
        self.entity_eps = getattr(cfg, "entity_label_smoothing", 0.0)
        self.mode = getattr(cfg, "mode", 0)
        self.tgt_ne_start_id = task.tgt_ne_start_id
        if hasattr(task, "target_dictionary") and task.target_dictionary is not None:
             self.padding_idx = task.target_dictionary.pad()
        else:
             # Fallback: older task may store it in task.tgt_dict
             self.padding_idx = getattr(getattr(task, "tgt_dict", None), "pad", lambda: 1)()
'''
    def __init__(self, cfg, task):
        super().__init__(task)
        self.args = cfg
        self.eps = getattr(cfg, "label_smoothing", 0.1)
        self.entity_eps = getattr(cfg, "entity_label_smoothing", 0.0)
        self.mode = int(getattr(cfg, "criterion_mode", getattr(cfg, "mode", 0)))
        self.tgt_ne_start_id = task.tgt_ne_start_id
        if hasattr(task, "target_dictionary") and task.target_dictionary is not None:
            self.padding_idx = task.target_dictionary.pad()
        else:
            self.padding_idx = getattr(getattr(task, "tgt_dict", None), "pad", lambda: 1)()

        # Added: register forward_* methods in the dispatch table
        self.forward_methods = [
            self.forward_0,  # 0
            self.forward_1,  # 1
            self.forward_2,  # 2
            self.forward_3,  # 3
            self.forward_4,  # 4
            self.forward_5,  # 5
        ]

    @staticmethod
    def add_args(parser):
        '''
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--ner-loss-weight', type=float, default=1.0)
        parser.add_argument('--src-ner-loss-weight', type=float)
        parser.add_argument('--tgt-ner-loss-weight', type=float)
        parser.add_argument('--tgt-ne-lookup-weight', type=float, default=1.0)
        parser.add_argument('--entity-label-smoothing', type=float, default=0)

        parser.add_argument('--loss-gamma', type=float, default=0)
        parser.add_argument('--ne-token-weight', type=float, default=1)
        '''
        # Newly added content below
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--ner-loss-weight', type=float, default=1.0)
        parser.add_argument('--src-ner-loss-weight', type=float)
        parser.add_argument('--tgt-ner-loss-weight', type=float)
        parser.add_argument('--tgt-ne-lookup-weight', type=float, default=1.0)
        parser.add_argument('--entity-label-smoothing', type=float, default=0)
        parser.add_argument('--loss-gamma', type=float, default=0)
        parser.add_argument('--ne-token-weight', type=float, default=1)
        # Added: training mode selector (forward_0 ~ forward_5)
        parser.add_argument('--criterion-mode', type=int, default=0,
                            help='loss mode: 0|1|2|3|4|5 (for entity criterion)')

    @classmethod
    def build_criterion(cls, cfg, task):
        return cls(cfg, task)

    def forward_0(self, model, sample, reduce):
        # mode 0 is same as LabelSmoothedCrossEntropyCriterion

        model_out = model(**sample['net_input'])
        loss, nll_loss = self.compute_translation_loss(model, model_out.decoder_out, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def forward_1(self, model, sample, reduce):
        """
        mode 1 has three loss.
        1. translation loss
        2. source ne loss
        3. tgt ne loss
        """
        '''
        model_out = model(**sample['net_input'])
        translation_loss, translation_nll_loss = self.compute_translation_loss(model, model_out.decoder_out, sample, reduce=reduce)
        src_ne_nll_loss = self.compute_ne_loss(model_out.encoder_ne_logit, sample['ne_source'], reduce=reduce)
        tgt_ne_nll_loss = self.compute_ne_loss(model_out.decoder_ne_logit, sample['ne_target'], reduce=reduce)

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        src_weight = self.args.ner_loss_weight if self.args.src_ner_loss_weight is None else self.args.src_ner_loss_weight
        tgt_weight = self.args.ner_loss_weight if self.args.tgt_ner_loss_weight is None else self.args.tgt_ner_loss_weight
        loss = translation_loss + src_ne_nll_loss * src_weight + tgt_ne_nll_loss * tgt_weight

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            't_loss': utils.item(translation_loss.data) if reduce else translation_loss.data,
            'nll_loss': utils.item(translation_nll_loss.data) if reduce else translation_nll_loss.data,
            'src_ne_loss': utils.item(src_ne_nll_loss.data) if reduce else src_ne_nll_loss.data,
            'tgt_ne_loss': utils.item(tgt_ne_nll_loss.data) if reduce else tgt_ne_nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output
        '''
        model_out = model(**sample['net_input'])
        translation_loss, translation_nll_loss = self.compute_translation_loss(
            model, model_out.decoder_out, sample, reduce=reduce
        )

        sample_size = (
            sample['target'].size(0)
            if self.args.sentence_avg
            else sample['ntokens']
        )

        src_weight = (
            self.args.ner_loss_weight
            if getattr(self.args, "src_ner_loss_weight", None) is None
            else self.args.src_ner_loss_weight
        )
        tgt_weight = (
            self.args.ner_loss_weight
            if getattr(self.args, "tgt_ner_loss_weight", None) is None
            else self.args.tgt_ner_loss_weight
        )

        zero = translation_loss.new_zeros([])

        # src NE loss
        src_ne_nll_loss = zero
        if "ne_source" in sample and src_weight != 0:
            src_ne_nll_loss = self.compute_ne_loss(
                model_out.encoder_ne_logit,
                sample["ne_source"],
                reduce=reduce,
            )

        # tgt NE loss
        tgt_ne_nll_loss = zero
        if "ne_target" in sample and tgt_weight != 0:
            tgt_ne_nll_loss = self.compute_ne_loss(
                model_out.decoder_ne_logit,
                sample["ne_target"],
                reduce=reduce,
            )

        if not torch.isfinite(src_ne_nll_loss):
            # You can also print a debug line
            # print("WARNING: src_ne_nll_loss is not finite, set to 0", flush=True)
            src_ne_nll_loss = src_ne_nll_loss.new_zeros([])

        if not torch.isfinite(tgt_ne_nll_loss):
            # print("WARNING: tgt_ne_nll_loss is not finite, set to 0", flush=True)
            tgt_ne_nll_loss = tgt_ne_nll_loss.new_zeros([])

        loss = translation_loss
        if src_weight != 0:
            loss = loss + src_ne_nll_loss * src_weight
        if tgt_weight != 0:
            loss = loss + tgt_ne_nll_loss * tgt_weight

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            't_loss': utils.item(translation_loss.data) if reduce else translation_loss.data,
            'nll_loss': utils.item(translation_nll_loss.data) if reduce else translation_nll_loss.data,
            'src_ne_loss': (
                utils.item(src_ne_nll_loss.data)
                if (reduce and torch.isfinite(src_ne_nll_loss))
                else 0.0
            ),
            'tgt_ne_loss': (
                utils.item(tgt_ne_nll_loss.data)
                if (reduce and torch.isfinite(tgt_ne_nll_loss))
                else 0.0
            ),
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def forward_2(self, model, sample, reduce):
        """
        mode 3 has the four loss
        1. src ne loss
        2. tgt ne loss. (merged entity)
        3. tgt non-entity translation loss
        4. tgt entity translation loss.(compare with BERT key)
        """
        entity_sent_mask = sample['entity_sent_mask']
        model_out = model(ne_source=sample['ne_source'], tgt_ne_pos=sample['tgt_ne_pos'], origin_tgt=sample['origin_tgt'], target=sample['target'], **sample['net_input'])

        entity_mask = model.get_targets(sample, model_out.decoder_out) > self.tgt_ne_start_id

        translation_loss, translation_nll_loss = self.compute_translation_loss(model, model_out.decoder_out, sample, reduce=reduce, position_mask=~entity_mask)

        src_ne_nll_loss = self.compute_ne_loss(model_out.encoder_ne_logit[entity_sent_mask], sample['ne_source'][entity_sent_mask], reduce=reduce)

        # No need to use entity_sent_mask here, as the entity mask already covered that case
        tgt_ne_nll_loss = self.compute_ne_loss(model_out.decoder_out[0], sample['target'], reduce=reduce, position_mask=entity_mask)

        # No need to use entity_sent_mask here, either. As there will be no entity generated.
        entity_lookup_count = model_out.entity_label.nelement()
        entity_lookup_loss = self.compute_entity_lookup_loss(model_out.entity_out, model_out.entity_label, reduce=reduce)

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        src_weight = self.args.ner_loss_weight if self.args.src_ner_loss_weight is None else self.args.src_ner_loss_weight
        tgt_weight = self.args.ner_loss_weight if self.args.tgt_ner_loss_weight is None else self.args.tgt_ner_loss_weight
        tgt_lookup_weight = self.args.tgt_ne_lookup_weight

        loss = translation_loss + src_ne_nll_loss * src_weight + tgt_ne_nll_loss * tgt_weight + entity_lookup_loss * tgt_lookup_weight

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            't_loss': utils.item(translation_loss.data) if reduce else translation_loss.data,
            'nll_loss': utils.item(translation_nll_loss.data) if reduce else translation_nll_loss.data,
            'src_ne_loss': utils.item(src_ne_nll_loss.data) if reduce else src_ne_nll_loss.data,
            'tgt_ne_loss': utils.item(tgt_ne_nll_loss.data) if reduce else tgt_ne_nll_loss.data,
            'tgt_lookup_loss': utils.item(entity_lookup_loss.data) if reduce else entity_lookup_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'n_nesentences': entity_sent_mask.sum().item(),
            'sample_size': sample_size,
            'entity_lookup_count': entity_lookup_count
        }
        return loss, sample_size, logging_output

    def forward_3(self, model, sample, reduce):
        # Mode 3 use same loss as mode0, which is standard loss
        return self.forward_0(model, sample, reduce)
    
    def forward_4(self, model, sample, reduce):
        """
        mode 4 has 3 loss.
        1. translation loss with weight
        2. source ne loss
        3. tgt ne loss
        """
        model_out = model(**sample['net_input'])
        
        entity_mask = sample['ne_target'] > 4 # $ is for 'O'
                
        normal_translation_loss, normal_translation_nll_loss = self.compute_translation_loss(model, model_out.decoder_out, sample, reduce=reduce, position_mask=~entity_mask)
        ne_translation_loss, ne_translation_nll_loss = self.compute_translation_loss(model, model_out.decoder_out, sample, reduce=reduce, position_mask=entity_mask)
        
        # make sure the loss still in the same scale
        p = self.args.ne_token_weight
        q = (1 + (1-p) * ne_translation_loss.item() / normal_translation_loss.item())

        translation_loss = p * ne_translation_loss + q * normal_translation_loss
        translation_nll_loss = p * ne_translation_nll_loss + q * normal_translation_nll_loss

        src_ne_nll_loss = self.compute_ne_loss(model_out.encoder_ne_logit, sample['ne_source'], reduce=reduce)
        tgt_ne_nll_loss = self.compute_ne_loss(model_out.decoder_ne_logit, sample['ne_target'], reduce=reduce)

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        src_weight = self.args.ner_loss_weight if self.args.src_ner_loss_weight is None else self.args.src_ner_loss_weight
        tgt_weight = self.args.ner_loss_weight if self.args.tgt_ner_loss_weight is None else self.args.tgt_ner_loss_weight
        loss = translation_loss + src_ne_nll_loss * src_weight + tgt_ne_nll_loss * tgt_weight
        
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            't_loss': utils.item(translation_loss.data) if reduce else translation_loss.data,
            'nll_loss': utils.item(translation_nll_loss.data) if reduce else translation_nll_loss.data,
            'src_ne_loss': utils.item(src_ne_nll_loss.data) if reduce else src_ne_nll_loss.data,
            'tgt_ne_loss': utils.item(tgt_ne_nll_loss.data) if reduce else tgt_ne_nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def forward_5(self, model, sample, reduce):
        """
        mode 5 has four loss.
        1. translation loss
        2. source ne loss
        3. tgt ne loss
        4 ne enhanced loss
        """
        model_out = model(**sample['net_input'])
        target = model.get_targets(sample, sample['net_input']).view(-1, 1)
        gamma = self.args.loss_gamma

        translation_loss, translation_nll_loss = self.compute_translation_loss(model, model_out.decoder_out, sample, reduce=reduce)
        src_ne_nll_loss = self.compute_ne_loss(model_out.encoder_ne_logit, sample['ne_source'], reduce=reduce)
        tgt_ne_nll_loss = self.compute_ne_loss(model_out.decoder_ne_logit, sample['ne_target'], reduce=reduce)

        ne_focal_loss, _ = self.compute_ne_focal_loss(gamma, model, model_out.decoder_ne_logit, model_out.decoder_out, target, reduce=reduce)

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        src_weight = self.args.ner_loss_weight if self.args.src_ner_loss_weight is None else self.args.src_ner_loss_weight
        tgt_weight = self.args.ner_loss_weight if self.args.tgt_ner_loss_weight is None else self.args.tgt_ner_loss_weight
        loss = translation_loss + src_ne_nll_loss * src_weight + tgt_ne_nll_loss * tgt_weight + ne_focal_loss

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            't_loss': utils.item(translation_loss.data) if reduce else translation_loss.data,
            'nll_loss': utils.item(translation_nll_loss.data) if reduce else translation_nll_loss.data,
            'src_ne_loss': utils.item(src_ne_nll_loss.data) if reduce else src_ne_nll_loss.data,
            'tgt_ne_loss': utils.item(tgt_ne_nll_loss.data) if reduce else tgt_ne_nll_loss.data,
            'ne_focal_loss': utils.item(ne_focal_loss.data) if reduce else ne_focal_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        mode = self.mode
        if not isinstance(mode, int) or mode < 0 or mode >= len(self.forward_methods):
            mode = 0
        return self.forward_methods[self.mode](model, sample, reduce)

    def compute_entity_lookup_loss(self, logit, target, reduce=True):
        # When there is no entity in the target, just return all zero
        if target.nelement() == 0:
            return torch.zeros((1), device=logit.device)
        lprobs = F.log_softmax(logit, dim=-1)
        
        return label_smoothed_nll_loss(lprobs, target, self.entity_eps,reduce=reduce)[0]
        # return F.nll_loss(lprobs, target, reduction='sum' if reduce else 'none')

    def compute_ne_focal_loss(self, gamma, model, ne_logit, token_logit, target, reduce=True):

        non_pad_mask = target.ne(self.padding_idx)
        ne_probs =F.softmax(ne_logit, dim=-1)[:, :, 4].detach() # (B, T, C)? 'O' is 4
        weight = (1 - ne_probs) ** gamma
        weight = weight.view(-1)[non_pad_mask.view(-1)]

        lprobs = model.get_normalized_probs(token_logit, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
        )

        loss = loss * weight
        nll_loss = nll_loss * weight
        if reduce:
            nll_loss = nll_loss.sum()
            loss = loss.sum()

        return loss, nll_loss

    def compute_ne_loss(self, logit, target, reduce=True, position_mask=None):
        '''
        # TODO do we need label smooth?
        lprobs = F.log_softmax(logit, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        if position_mask is not None:
            position_mask = position_mask.view(-1)
            lprobs = lprobs[position_mask]
            target = target[position_mask]
        
        if target.nelement() > 0:
            if self.entity_eps > 0:
                # Can't use builtin smooth loss, implement here
                if target.dim() == lprobs.dim() - 1:
                    target = target.unsqueeze(-1)
                forbidden_index = (lprobs == float('-inf')).all(axis=0)
                nll_loss = -lprobs.gather(dim=-1, index=target)
                smooth_loss = -lprobs[:, ~forbidden_index].sum(dim=-1, keepdim=True)

                non_pad_mask = target.ne(self.padding_idx)
                nll_loss = nll_loss[non_pad_mask]
                smooth_loss = smooth_loss[non_pad_mask]

                if reduce:
                    nll_loss = nll_loss.sum()
                    smooth_loss = smooth_loss.sum()
                eps_i = self.entity_eps / (~forbidden_index).sum().item()
                loss = (1. - self.entity_eps) * nll_loss + eps_i * smooth_loss

                return loss
            else:
                return F.nll_loss(lprobs, target, ignore_index=self.padding_idx, reduction='sum' if reduce else 'none')
        return torch.tensor(0.0, device=target.device)
        '''
       # logit: (B, T, C)
        lprobs = F.log_softmax(logit, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)

        # 1) If position_mask exists, filter first
        if position_mask is not None:
            position_mask = position_mask.view(-1)
            if not position_mask.any():
                # This batch has no NE positions
                return lprobs.new_zeros([])
            lprobs = lprobs[position_mask]
            target = target[position_mask]

        # 2) If target is empty, return 0
        if target.nelement() == 0:
            return lprobs.new_zeros([])

        padding_idx = self.padding_idx

        # ====== branch without label smoothing ======
        if self.entity_eps <= 0:
            valid_mask = (target != padding_idx)
            if not valid_mask.any():
                return lprobs.new_zeros([])

            lprobs_valid = lprobs[valid_mask]
            target_valid = target[valid_mask]

            loss = F.nll_loss(
                lprobs_valid,
                target_valid,
                reduction='sum' if reduce else 'none'
            )
            return loss

        # ====== branch with label smoothing ======
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)  # (N,1)

        # One class is -inf at all positions
        forbidden_index = (lprobs == float('-inf')).all(dim=0)  # (C,)
        num_valid_class = (~forbidden_index).sum().item()

        # If all classes are forbidden, logits are invalid; return 0
        if num_valid_class == 0:
            return lprobs.new_zeros([])

        # Standard smoothing computation
        nll_loss = -lprobs.gather(dim=-1, index=target)  # (N,1)
        smooth_loss = -lprobs[:, ~forbidden_index].sum(dim=-1, keepdim=True)  # (N,1)

        non_pad_mask = (target.squeeze(-1) != padding_idx)
        if not non_pad_mask.any():
            return lprobs.new_zeros([])

        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]

        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = self.entity_eps / num_valid_class
        loss = (1.0 - self.entity_eps) * nll_loss + eps_i * smooth_loss

        # Final safeguard
        if not torch.isfinite(loss):
            # print("WARNING: compute_ne_loss produced non-finite, set to 0", flush=True)
            loss = lprobs.new_zeros([])

        return loss
    def compute_translation_loss(self, model, net_output, sample, reduce=True, position_mask=None):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        if position_mask is not None:
            position_mask = position_mask.view(-1)
            lprobs = lprobs[position_mask]
            target = target[position_mask]
        
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        n_nesentences = sum(log.get('n_nesentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        entity_lookup_count = sum(log.get('entity_lookup_count', 0) for log in logging_outputs)

        result = {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'n_nesentences': n_nesentences,
            'sample_size': sample_size
        } 

        if len(logging_outputs) > 0:
            if 'entity_lookup_count' in logging_outputs[0]:
                result['entity_lookup_count'] = entity_lookup_count
            for k in ['src_ne_loss', 'tgt_ne_loss', 't_loss', 'tgt_lookup_loss', 'ne_focal_loss']:
                if k in logging_outputs[0]:
                    if k == 'tgt_lookup_loss':
                        result[k] = sum(log.get(k, 0) for log in logging_outputs) / entity_lookup_count / math.log(2) if entity_lookup_count > 0 else 0.
                    else:
                        result[k] = sum(log.get(k, 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.

        return result
