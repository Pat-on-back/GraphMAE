import torch
from torch_geometric.data import Data, Batch

# 定义一个新的批图类 BatchMasking，继承自 Data
class BatchMasking(Data):

    r"""一个普通 Python 对象，用来把多个图拼接成一个大图（不连通图）。
    因为继承自 torch_geometric.data.Data,所以所有 Data 方法都能用。
    此外，可以通过 batch 向量恢复每个节点属于哪个原始图。
    """
    # 构造函数
    def __init__(self, batch=None, **kwargs):

        # 调用父类 Data 的初始化
        # kwargs 中通常包含 edge_index, x 等图属性
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    # 静态方法：从 Data 列表构造 BatchMasking
    @staticmethod
    def from_data_list(data_list):

        r"""从 Data 对象列表构建 batch 图
        同时自动创建 batch 向量
        """
        # Step1：收集所有字段名
        # 每个 data.keys 是该图拥有的属性名集合
        # 如 {'x','edge_index','edge_attr'}
        keys = [set(data.keys) for data in data_list]

        # 合并所有图的字段名
        keys = list(set.union(*keys))

        # 确保没有已有 batch 字段（避免冲突）
        assert 'batch' not in keys
        # Step2：创建空 batch 对象
        batch = BatchMasking()

        # 为每个字段创建空列表，用来存拼接数据
        for key in keys:
            batch[key] = []

        # batch.batch 也是列表
        batch.batch = []
        
        # Step3：初始化累计偏移量
        # 节点编号偏移量
        cumsum_node = 0
        # 边编号偏移量
        cumsum_edge = 0
        
        # Step4：遍历每个图
        for i, data in enumerate(data_list):
            # 当前图节点数
            num_nodes = data.num_nodes
            # 构造当前图节点所属图编号向量
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            # Step5：遍历当前图的每个字段
            for key in data.keys:
                # 取出字段数据
                item = data[key]
                # 如果是边索引或mask节点索引,需要加节点偏移量
                if key in ['edge_index', 'masked_atom_indices']:
                    item = item + cumsum_node
                # 如果是连接边索引,需要加边偏移量
                elif key == 'connected_edge_indices':
                    item = item + cumsum_edge

                # 加入 batch 对应字段列表
                batch[key].append(item)

            # 更新节点偏移量
            cumsum_node += num_nodes
            # 更新边偏移量
            cumsum_edge += data.edge_index.shape[1]
        # Step6：拼接所有图字段
        for key in keys:

            # torch.cat 拼接所有图该字段
            # dim 由 Data 类定义规则决定
            batch[key] = torch.cat(
                batch[key],
                dim=data_list[0].__cat_dim__(key, batch[key][0])
            )
        # 拼接 batch 向量
        batch.batch = torch.cat(batch.batch, dim=-1)
        # 返回连续内存版本（提高计算性能）
        return batch.contiguous()
        
    # 是否需要累计偏移的判断函数
    def cumsum(self, key, item):

        r"""如果返回 True，说明该字段需要在拼接前做偏移累加
        这个函数主要给内部机制调用
        """

        return key in [
            'edge_index',
            'face',
            'masked_atom_indices',
            'connected_edge_indices'
        ]
    # 属性：返回 batch 中图数量
    @property
    def num_graphs(self):

        """返回当前 batch 中图的数量"""
        # batch[-1] 是最后一个节点所属图编号
        # +1 就是图总数
        return self.batch[-1].item() + 1

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "negative_edge_index"] else 0


class BatchSubstructContext(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    """
    Specialized batching for substructure context pair!
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchSubstructContext, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        #keys = [set(data.keys) for data in data_list]
        #keys = list(set.union(*keys))
        #assert 'batch' not in keys

        batch = BatchSubstructContext()
        keys = ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct", "overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]

        for key in keys:
            #print(key)
            batch[key] = []

        #batch.batch = []
        #used for pooling the context
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []

        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0

        i = 0
        
        for data in data_list:
            #If there is no context, just skip!!
            if hasattr(data, "x_context"):
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                #batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
                batch.batch_overlapped_context.append(torch.full((len(data.overlap_context_substruct_idx), ), i, dtype=torch.long))
                batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))

                ###batching for the main graph
                #for key in data.keys:
                #    if not "context" in key and not "substruct" in key:
                #        item = data[key]
                #        item = item + cumsum_main if batch.cumsum(key, item) else item
                #        batch[key].append(item)
                
                ###batching for the substructure graph
                for key in ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct"]:
                    item = data[key]
                    item = item + cumsum_substruct if batch.cumsum(key, item) else item
                    batch[key].append(item)
                

                ###batching for the context graph
                for key in ["overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]:
                    item = data[key]
                    item = item + cumsum_context if batch.cumsum(key, item) else item
                    batch[key].append(item)

                cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct   
                cumsum_context += num_nodes_context
                i += 1

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=batch.cat_dim(key))
        #batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_overlapped_context = torch.cat(batch.batch_overlapped_context, dim=-1)
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)

        return batch.contiguous()

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "edge_index_substruct", "edge_index_context"] else 0

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ["edge_index", "edge_index_substruct", "edge_index_context", "overlap_context_substruct_idx", "center_substruct_idx"]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1
