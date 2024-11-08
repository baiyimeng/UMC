from models.inputs import (
    build_input_features,
    SparseFeat,
    DenseFeat,
    VarLenSparseFeat,
    get_varlen_pooling_list,
    create_embedding_matrix,
    varlen_embedding_lookup,
)
import torch


def flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def input_from_feature_columns(
    X, feature_columns, embedding_dict, feature_index, device, support_dense=True
):

    sparse_feature_columns = (
        list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
        if len(feature_columns)
        else []
    )
    dense_feature_columns = (
        list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))
        if len(feature_columns)
        else []
    )

    varlen_sparse_feature_columns = (
        list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns))
        if feature_columns
        else []
    )

    if not support_dense and len(dense_feature_columns) > 0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")

    sparse_embedding_list = [
        embedding_dict[feat.embedding_name](
            X[:, feature_index[feat.name][0] : feature_index[feat.name][1]].long()
        )
        for feat in sparse_feature_columns
    ]

    sequence_embed_dict = varlen_embedding_lookup(
        X, embedding_dict, feature_index, varlen_sparse_feature_columns
    )
    varlen_sparse_embedding_list = get_varlen_pooling_list(
        sequence_embed_dict, X, feature_index, varlen_sparse_feature_columns, device
    )

    dense_value_list = [
        X[:, feature_index[feat.name][0] : feature_index[feat.name][1]]
        for feat in dense_feature_columns
    ]

    return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list


def compute_input_dim(
    feature_columns, include_sparse=True, include_dense=True, feature_group=False
):
    sparse_feature_columns = (
        list(
            filter(
                lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns
            )
        )
        if len(feature_columns)
        else []
    )
    dense_feature_columns = (
        list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))
        if len(feature_columns)
        else []
    )

    dense_input_dim = sum(map(lambda x: x.dimension, dense_feature_columns))
    if feature_group:
        sparse_input_dim = len(sparse_feature_columns)
    else:
        sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
    input_dim = 0
    if include_sparse:
        input_dim += sparse_input_dim
    if include_dense:
        input_dim += dense_input_dim
    return input_dim
