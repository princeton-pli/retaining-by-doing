"""
python -m core.training.objectives
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print


def masked_whiten(values, mask, shift_mean=True):
    """
    Whiten `values` by normalizing with mean and variance computed over `mask`.

    Args:
        values (torch.Tensor): Input tensor.
        mask (torch.Tensor): Boolean tensor of same shape, selects elements for stats.
        shift_mean (bool): If True (default), output is zero-mean;
                           if False, the original mean is re-added after scaling.

    Returns:
        torch.Tensor: Whitened tensor of same shape as `values`.
    """
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def masked_sum(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    # If NaNs exist out of mask, replace NaNs in values with a value that
    # won't affect the sum (e.g., 0 for masked regions)
    valid_values = torch.where(mask.bool(), values, 0.0)
    return (valid_values * mask).sum(axis=axis)


def masked_mean(values, mask, axis=None):
    """
    Compute the mean of `values` over elements selected by `mask`.

    Args:
        values (Tensor): Input tensor.
        mask (Tensor): Boolean or numeric mask of the same shape as `values`.
        axis (int or tuple of int, optional): Dimension(s) along which to compute the mean.
            Defaults to None (over all elements).

    Returns:
        Tensor: Masked mean, with shape equal to `values` reduced over `axis`.
    """
    s = masked_sum(values, mask, axis)
    return s / (mask.sum(axis=axis) + 1e-8)


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def construct_token_level_rewards(action_mask):
    # 1) mask where x == 1
    mask = action_mask == 1
    # 2) build a “next‐col” mask: for each row, shift mask left and pad a False at the end
    next_is_one = torch.cat([mask[:, 1:], torch.zeros(mask.size(0), 1, dtype=torch.bool).to(mask.device)], dim=1)
    
    # 3) the end of a run is where mask is True but next_is_one is False
    ends = mask & (~next_is_one)
    
    # 4) convert back to float
    return ends.float()


class ModelOutput:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class BaseLoss:
    def __init__(self, config=None):
        self.config = config if config is not None else dict()
    
    def cross_entropy_loss(self, logits, labels, vocab_size, reduction='mean'):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        return F.cross_entropy(shift_logits, shift_labels, reduction=reduction)


class CrossEntropyLoss(BaseLoss):
    def __init__(self, config=None):
        super().__init__(config)

    def compute_loss(self, model, inputs, tokenizer=None, reduction='mean', **kwargs):
        inputs = {k: v.cuda() for k, v in inputs.items() if k in ('input_ids', 'attention_mask', 'labels')}
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        labels = inputs['labels'].cuda()
        action_mask = (labels[..., :-1] != -100).long()

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.cross_entropy_loss(output.logits, labels, model.config.vocab_size, reduction=reduction)
        
        logits = output.logits
        logprobs = F.log_softmax(logits[..., :-1, :], dim=-1).gather(dim=-1, index=input_ids[..., 1:].unsqueeze(-1)).squeeze(-1)
        
        kl_beta = kwargs.get('kl_beta', 0.0)
        ref_model = kwargs.get('ref_model', None)
        if kl_beta > 0.0 and ref_model is not None:
            with torch.no_grad():
                ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
                ref_logits = ref_outputs.logits
                ref_logprobs = F.log_softmax(ref_logits[..., :-1, :], dim=-1).gather(dim=-1, index=input_ids[..., 1:].unsqueeze(-1)).squeeze(-1)

            kl = 0.5 * (logprobs - ref_logprobs).pow(2)
            loss += kl_beta * (kl.sum(dim=-1) / action_mask.sum(dim=-1)).mean()

        avg_logprobs = ((logprobs * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)).unsqueeze(-1).clone().detach()
        return ModelOutput(
            loss=loss,
            logprobs=avg_logprobs,  # bsz x 1
        )


class GRPOLoss(BaseLoss):
    def __init__(self, config=None):
        super().__init__(config)
        from torch.nn.utils.rnn import pad_sequence
        from core.data import tokenize_messages
        self.tokenize_messages = tokenize_messages
        self.pad_sequence = pad_sequence
    
    def compute_loss(
        self,
        model,
        ref_model,
        reward_model,
        batch_on_policy_texts,
        batch_parsed_on_policy_texts,
        batch_ends_with_eos,
        datapoints,
        tokenizer,
        **kwargs,
    ):
        """
        batch_on_policy_texts (List[List[str]]): shape = bsz x num_generation_per_prompt
        batch_parsed_on_policy_texts (List[List[str]]): shape = bsz x num_generation_per_prompt
        batch_ends_with_eos (List[List[bool]]): shape = bsz x num_generation_per_prompt
        kl_beta (float)
        datapoints (List[Dict]): shape = bsz
        tokenizer (Tokenizer)
        max_seq_length (int): allowable training sequence length
        max_new_tokens (int): max generation tokens
        local_rank (int)

        GRPO:
        1. advantage is calculated within each group
        2. loss/reward/advantage/kl is calculated at token level
        """
        kl_beta = kwargs.get('kl_beta', None)
        max_seq_length = kwargs.get('max_seq_length', None)
        max_new_tokens = kwargs.get('max_new_tokens', None)

        # Number of groups needs to be greater than 1
        assert len(batch_on_policy_texts[0]) == len(batch_parsed_on_policy_texts[0]) == len(batch_ends_with_eos[0]) > 1

        bsz = len(datapoints)
        num_generation_per_prompt = len(batch_on_policy_texts[0])

        # tokenize on-policy texts
        batch_messages_without_last_assistant = [datapoint['messages'][:-1] if datapoint['messages'][-1]['role'] == 'assistant' else datapoint['messages'] for datapoint in datapoints]
        batch_on_policy_messages = [
            [messages_without_last_assistant + [dict(role='assistant', content=text)] for text in on_policy_texts]
            for messages_without_last_assistant, on_policy_texts in zip(batch_messages_without_last_assistant, batch_on_policy_texts)
        ]
        batch_on_policy_tokenized = [
            [
                self.tokenize_messages(
                    msg,
                    tokenizer,
                    max_seq_length + max_new_tokens
                )
                for msg in on_policy_messages
            ]
            for on_policy_messages in batch_on_policy_messages
        ]

        input_ids = [tokenized['input_ids'] for on_policy_tokenized in batch_on_policy_tokenized for tokenized in on_policy_tokenized]
        attention_mask = [tokenized['attention_mask'] for on_policy_tokenized in batch_on_policy_tokenized for tokenized in on_policy_tokenized]
        labels = [tokenized['labels'] for on_policy_tokenized in batch_on_policy_tokenized for tokenized in on_policy_tokenized]

        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).cuda()
        attention_mask = self.pad_sequence(attention_mask, batch_first=True, padding_value=0).cuda()
        labels = self.pad_sequence(labels, batch_first=True, padding_value=-100).cuda()

        action_mask = (labels[..., :-1] != -100).long()

        batch_ends_with_eos = torch.tensor(batch_ends_with_eos).float().to(input_ids.device)

        outcome_rewards = []
        for parsed_on_policy_texts, datapoint in zip(batch_parsed_on_policy_texts, datapoints):
            outcome_rewards.append([float(reward_model.compute_outcome_reward(text, datapoint)) for text in parsed_on_policy_texts])

        outcome_rewards = torch.tensor(outcome_rewards).to(input_ids.device)  # shape = bsz x num_generation_per_prompt
        outcome_rewards = outcome_rewards * batch_ends_with_eos
        or_std = outcome_rewards.std(dim=1, keepdim=True) + 1e-8
        or_mean = outcome_rewards.mean(dim=1, keepdim=True)
        orig_outcome_rewards = outcome_rewards.clone().detach()
        outcome_rewards = (outcome_rewards - or_mean) / or_std
        advantages = outcome_rewards.reshape(bsz * num_generation_per_prompt, -1)  # shape = (bsz * num_generation_per_prompt) x seq_length
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # shape = bsz x num_generation_per_prompt x seq_length x vocab_size
        logprobs = F.log_softmax(logits[..., :-1, :], dim=-1).gather(dim=-1, index=input_ids[..., 1:].unsqueeze(-1)).squeeze(-1)

        # Calculate KL divergence to the reference model ######################
        if ref_model is not None and kl_beta > 0.0:
            with torch.no_grad():
                ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
                ref_logits = ref_outputs.logits.detach()
                ref_logprobs = F.log_softmax(ref_logits[..., :-1, :], dim=-1).gather(dim=-1, index=input_ids[..., 1:].unsqueeze(-1)).squeeze(-1)
            kl = 0.5 * (logprobs - ref_logprobs).pow(2)
            avg_kl = (kl * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)
            avg_kl = avg_kl.reshape(bsz, num_generation_per_prompt).mean(dim=1, keepdim=True)
        else:
            kl = 0.0
            avg_kl = torch.zeros(bsz, 1).to(input_ids.device)
        #######################################################################

        avg_logprobs = (logprobs * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)
        avg_logprobs = avg_logprobs.reshape(bsz, num_generation_per_prompt, -1).squeeze(-1)

        loss = -(advantages * logprobs - kl_beta * kl)  # shape = (bsz * num_generation_per_prompt) x seq_length
        loss = ((loss * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)).mean()

        return ModelOutput(
            loss=loss,
            logprobs=avg_logprobs,         # bsz x num_generation_per_prompt
            kl=avg_kl,                     # bsz x 1
            rewards=orig_outcome_rewards,  # bsz x num_generation_per_prompt
        )
