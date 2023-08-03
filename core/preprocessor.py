from collections import OrderedDict
from functools import partial

from paddle.io import BatchSampler, DataLoader
from paddlenlp.data import Stack, Pad, Dict
from paddlenlp.datasets import MapDataset


def __convert_example(example: dict, tokenizer, max_length=512):
    # 不用padding，由DataLoader的collate_fn参数来进行处理
    _example_encoded = tokenizer(
        text=example["text"],
        max_length=max_length,
        return_attention_mask=True
    )
    if "labels" in example:
        _example_encoded["labels"] = example["labels"]
    return _example_encoded


def __batchify_fn(samples, tokenizer, with_labels: bool):
    _batchify_ops = OrderedDict()
    _batchify_ops["input_ids"] = Pad(axis=0, pad_val=tokenizer.pad_token_id)
    _batchify_ops["token_type_ids"] = Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
    _batchify_ops["attention_mask"] = Pad(axis=0, pad_val=0)
    if with_labels:
        _batchify_ops["labels"] = Stack(dtype="int64")
    return Dict(_batchify_ops)(samples)


def create_dataloader(dataset: MapDataset, tokenizer, batch_size=16, with_labels=True):
    """
    :param dataset: 数据集
    :param tokenizer: 分词器
    :param batch_size: 批量大小
    :param with_labels: 数据集是否包含labels字段，默认True
    :return:
    """
    _trans_fn = partial(__convert_example, tokenizer=tokenizer)
    dataset.map(_trans_fn)

    _batchify_fn = partial(__batchify_fn, tokenizer=tokenizer, with_labels=with_labels)
    _batch_sampler = BatchSampler(dataset, batch_size=batch_size, shuffle=True)
    return DataLoader(dataset=dataset, batch_sampler=_batch_sampler, collate_fn=_batchify_fn, return_list=True)
