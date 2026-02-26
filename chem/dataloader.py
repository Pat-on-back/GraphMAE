import torch.utils.data
from torch.utils.data.dataloader import default_collate

from batch import BatchSubstructContext, BatchMasking, BatchAE

class DataLoaderSubstructContext(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderSubstructContext, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchSubstructContext.from_data_list(data_list),
            **kwargs)

class DataLoaderMasking(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMasking, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMasking.from_data_list(data_list),
            **kwargs)


from util import MaskAtom


class DataLoaderMaskingPred(torch.utils.data.DataLoader):

    r"""数据加载器，用于将来自 :class:torch_geometric.data.dataset
    的数据对象合并为一个小批量（mini-batch）
    参数说明：
        dataset (Dataset): 要加载的数据集对象
        batch_size (int): 每个 batch 多少样本（默认1）
        shuffle (bool): 每轮训练是否打乱顺序（默认True）
    """
    # 初始化函数
    def __init__(self, dataset, batch_size=1, shuffle=True,
                 mask_rate=0.0, mask_edge=0.0, **kwargs):

        # 创建一个 MaskAtom 数据增强对象
        # 作用：对图中节点/边做 masking
        self._transform = MaskAtom(
            num_atom_type=119, # 节点类别数
            num_edge_type=5, # 边类别数
            mask_rate=mask_rate, # 节点mask比例
            mask_edge=mask_edge # 边mask比例
        )
        # 调用父类 DataLoader 初始化函数
        super(DataLoaderMaskingPred, self).__init__(
            dataset,
            batch_size,
            shuffle,
            # 指定自定义拼接函数
            collate_fn=self.collate_fn,
            # 其余参数透传给 DataLoader
            **kwargs
        )
    # 自定义 batch 拼接函数
    def collate_fn(self, batches):
        # 对 batch 中每个样本执行 transform，batches 是 list[Data]
        # 逐个调用 MaskAtom
        batchs = [self._transform(x) for x in batches]
        # 把增强后的样本拼接成大图 batch
        return BatchMasking.from_data_list(batchs)

class DataLoaderAE(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderAE, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchAE.from_data_list(data_list),
            **kwargs)



