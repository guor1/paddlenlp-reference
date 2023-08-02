from functools import partial

from paddle.io import DistributedBatchSampler, DataLoader
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Pad, Dict


def convert_example(example: dict, tokenizer=None, max_length=512):
    if tokenizer is None:
        raise ValueError("tokenizer is required")
    # 不用padding，由DataLoader的collate_fn参数来进行处理
    _example_encoded = tokenizer(
        text=example["text"],
        max_length=max_length,
        return_attention_mask=True
    )
    if "labels" in example:
        _example_encoded["labels"] = example["labels"]
    return _example_encoded


def __batchify_fn(samples, tokenizer):
    return Dict({
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        "attention_mask": Pad(axis=0, pad_val=0),
        "labels": Stack(dtype="int64")
    })(samples)


def create_train_dataloader(dataset: MapDataset, tokenizer, batch_size=16):
    _batchify_fn = partial(__batchify_fn, tokenizer=tokenizer)
    _batch_sampler = DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=True)
    return DataLoader(dataset=dataset, batch_sampler=_batch_sampler, collate_fn=_batchify_fn, return_list=True)
