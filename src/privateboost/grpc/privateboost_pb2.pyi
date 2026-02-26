from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FLOAT64: _ClassVar[DType]
    FLOAT32: _ClassVar[DType]
    INT32: _ClassVar[DType]
    INT64: _ClassVar[DType]

class Phase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WAITING_FOR_CLIENTS: _ClassVar[Phase]
    COLLECTING_GRADIENTS: _ClassVar[Phase]
    TRAINING_COMPLETE: _ClassVar[Phase]
FLOAT64: DType
FLOAT32: DType
INT32: DType
INT64: DType
WAITING_FOR_CLIENTS: Phase
COLLECTING_GRADIENTS: Phase
TRAINING_COMPLETE: Phase

class NdArray(_message.Message):
    __slots__ = ("dtype", "shape", "data")
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    dtype: DType
    shape: _containers.RepeatedScalarFieldContainer[int]
    data: bytes
    def __init__(self, dtype: _Optional[_Union[DType, str]] = ..., shape: _Optional[_Iterable[int]] = ..., data: _Optional[bytes] = ...) -> None: ...

class Share(_message.Message):
    __slots__ = ("x", "y")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: int
    y: NdArray
    def __init__(self, x: _Optional[int] = ..., y: _Optional[_Union[NdArray, _Mapping]] = ...) -> None: ...

class SubmitStatsRequest(_message.Message):
    __slots__ = ("session_id", "commitment", "share")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    COMMITMENT_FIELD_NUMBER: _ClassVar[int]
    SHARE_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    commitment: bytes
    share: Share
    def __init__(self, session_id: _Optional[str] = ..., commitment: _Optional[bytes] = ..., share: _Optional[_Union[Share, _Mapping]] = ...) -> None: ...

class SubmitStatsResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SubmitGradientsRequest(_message.Message):
    __slots__ = ("session_id", "round_id", "depth", "commitment", "share", "node_id")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    ROUND_ID_FIELD_NUMBER: _ClassVar[int]
    DEPTH_FIELD_NUMBER: _ClassVar[int]
    COMMITMENT_FIELD_NUMBER: _ClassVar[int]
    SHARE_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    round_id: int
    depth: int
    commitment: bytes
    share: Share
    node_id: int
    def __init__(self, session_id: _Optional[str] = ..., round_id: _Optional[int] = ..., depth: _Optional[int] = ..., commitment: _Optional[bytes] = ..., share: _Optional[_Union[Share, _Mapping]] = ..., node_id: _Optional[int] = ...) -> None: ...

class SubmitGradientsResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetStatsCommitmentsRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class GetStatsCommitmentsResponse(_message.Message):
    __slots__ = ("commitments",)
    COMMITMENTS_FIELD_NUMBER: _ClassVar[int]
    commitments: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, commitments: _Optional[_Iterable[bytes]] = ...) -> None: ...

class GetGradientCommitmentsRequest(_message.Message):
    __slots__ = ("session_id", "round_id", "depth")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    ROUND_ID_FIELD_NUMBER: _ClassVar[int]
    DEPTH_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    round_id: int
    depth: int
    def __init__(self, session_id: _Optional[str] = ..., round_id: _Optional[int] = ..., depth: _Optional[int] = ...) -> None: ...

class GetGradientCommitmentsResponse(_message.Message):
    __slots__ = ("commitments",)
    COMMITMENTS_FIELD_NUMBER: _ClassVar[int]
    commitments: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, commitments: _Optional[_Iterable[bytes]] = ...) -> None: ...

class GetGradientNodeIdsRequest(_message.Message):
    __slots__ = ("session_id", "round_id", "depth")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    ROUND_ID_FIELD_NUMBER: _ClassVar[int]
    DEPTH_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    round_id: int
    depth: int
    def __init__(self, session_id: _Optional[str] = ..., round_id: _Optional[int] = ..., depth: _Optional[int] = ...) -> None: ...

class GetGradientNodeIdsResponse(_message.Message):
    __slots__ = ("node_ids",)
    NODE_IDS_FIELD_NUMBER: _ClassVar[int]
    node_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, node_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class GetStatsSumRequest(_message.Message):
    __slots__ = ("session_id", "commitments")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    COMMITMENTS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    commitments: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, session_id: _Optional[str] = ..., commitments: _Optional[_Iterable[bytes]] = ...) -> None: ...

class GetStatsSumResponse(_message.Message):
    __slots__ = ("sum",)
    SUM_FIELD_NUMBER: _ClassVar[int]
    sum: NdArray
    def __init__(self, sum: _Optional[_Union[NdArray, _Mapping]] = ...) -> None: ...

class GetGradientsSumRequest(_message.Message):
    __slots__ = ("session_id", "round_id", "depth", "commitments", "node_id")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    ROUND_ID_FIELD_NUMBER: _ClassVar[int]
    DEPTH_FIELD_NUMBER: _ClassVar[int]
    COMMITMENTS_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    round_id: int
    depth: int
    commitments: _containers.RepeatedScalarFieldContainer[bytes]
    node_id: int
    def __init__(self, session_id: _Optional[str] = ..., round_id: _Optional[int] = ..., depth: _Optional[int] = ..., commitments: _Optional[_Iterable[bytes]] = ..., node_id: _Optional[int] = ...) -> None: ...

class GetGradientsSumResponse(_message.Message):
    __slots__ = ("sum",)
    SUM_FIELD_NUMBER: _ClassVar[int]
    sum: NdArray
    def __init__(self, sum: _Optional[_Union[NdArray, _Mapping]] = ...) -> None: ...

class CancelSessionRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class CancelSessionResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ResetRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class ResetResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class JoinSessionRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class JoinSessionResponse(_message.Message):
    __slots__ = ("session_id", "shareholders", "threshold", "config")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SHAREHOLDERS_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    shareholders: _containers.RepeatedCompositeFieldContainer[ShareholderInfo]
    threshold: int
    config: TrainingConfig
    def __init__(self, session_id: _Optional[str] = ..., shareholders: _Optional[_Iterable[_Union[ShareholderInfo, _Mapping]]] = ..., threshold: _Optional[int] = ..., config: _Optional[_Union[TrainingConfig, _Mapping]] = ...) -> None: ...

class ShareholderInfo(_message.Message):
    __slots__ = ("id", "address", "x_coord")
    ID_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    X_COORD_FIELD_NUMBER: _ClassVar[int]
    id: str
    address: str
    x_coord: int
    def __init__(self, id: _Optional[str] = ..., address: _Optional[str] = ..., x_coord: _Optional[int] = ...) -> None: ...

class TrainingConfig(_message.Message):
    __slots__ = ("features", "target_column", "loss", "n_bins", "n_trees", "max_depth", "learning_rate", "lambda_reg", "min_clients", "target_count", "target_fraction")
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    TARGET_COLUMN_FIELD_NUMBER: _ClassVar[int]
    LOSS_FIELD_NUMBER: _ClassVar[int]
    N_BINS_FIELD_NUMBER: _ClassVar[int]
    N_TREES_FIELD_NUMBER: _ClassVar[int]
    MAX_DEPTH_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    LAMBDA_REG_FIELD_NUMBER: _ClassVar[int]
    MIN_CLIENTS_FIELD_NUMBER: _ClassVar[int]
    TARGET_COUNT_FIELD_NUMBER: _ClassVar[int]
    TARGET_FRACTION_FIELD_NUMBER: _ClassVar[int]
    features: _containers.RepeatedCompositeFieldContainer[FeatureSpec]
    target_column: str
    loss: str
    n_bins: int
    n_trees: int
    max_depth: int
    learning_rate: float
    lambda_reg: float
    min_clients: int
    target_count: int
    target_fraction: float
    def __init__(self, features: _Optional[_Iterable[_Union[FeatureSpec, _Mapping]]] = ..., target_column: _Optional[str] = ..., loss: _Optional[str] = ..., n_bins: _Optional[int] = ..., n_trees: _Optional[int] = ..., max_depth: _Optional[int] = ..., learning_rate: _Optional[float] = ..., lambda_reg: _Optional[float] = ..., min_clients: _Optional[int] = ..., target_count: _Optional[int] = ..., target_fraction: _Optional[float] = ...) -> None: ...

class FeatureSpec(_message.Message):
    __slots__ = ("index", "name")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    index: int
    name: str
    def __init__(self, index: _Optional[int] = ..., name: _Optional[str] = ...) -> None: ...

class GetRoundStateRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class GetRoundStateResponse(_message.Message):
    __slots__ = ("phase", "round_id", "depth")
    PHASE_FIELD_NUMBER: _ClassVar[int]
    ROUND_ID_FIELD_NUMBER: _ClassVar[int]
    DEPTH_FIELD_NUMBER: _ClassVar[int]
    phase: Phase
    round_id: int
    depth: int
    def __init__(self, phase: _Optional[_Union[Phase, str]] = ..., round_id: _Optional[int] = ..., depth: _Optional[int] = ...) -> None: ...

class GetTrainingStateRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class GetTrainingStateResponse(_message.Message):
    __slots__ = ("model", "current_splits", "bins", "round_id", "depth")
    class CurrentSplitsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: SplitDecision
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[SplitDecision, _Mapping]] = ...) -> None: ...
    MODEL_FIELD_NUMBER: _ClassVar[int]
    CURRENT_SPLITS_FIELD_NUMBER: _ClassVar[int]
    BINS_FIELD_NUMBER: _ClassVar[int]
    ROUND_ID_FIELD_NUMBER: _ClassVar[int]
    DEPTH_FIELD_NUMBER: _ClassVar[int]
    model: Model
    current_splits: _containers.MessageMap[int, SplitDecision]
    bins: _containers.RepeatedCompositeFieldContainer[BinConfiguration]
    round_id: int
    depth: int
    def __init__(self, model: _Optional[_Union[Model, _Mapping]] = ..., current_splits: _Optional[_Mapping[int, SplitDecision]] = ..., bins: _Optional[_Iterable[_Union[BinConfiguration, _Mapping]]] = ..., round_id: _Optional[int] = ..., depth: _Optional[int] = ...) -> None: ...

class GetModelRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class GetModelResponse(_message.Message):
    __slots__ = ("model",)
    MODEL_FIELD_NUMBER: _ClassVar[int]
    model: Model
    def __init__(self, model: _Optional[_Union[Model, _Mapping]] = ...) -> None: ...

class BinConfiguration(_message.Message):
    __slots__ = ("feature_idx", "edges", "inner_edges", "n_bins")
    FEATURE_IDX_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    INNER_EDGES_FIELD_NUMBER: _ClassVar[int]
    N_BINS_FIELD_NUMBER: _ClassVar[int]
    feature_idx: int
    edges: NdArray
    inner_edges: NdArray
    n_bins: int
    def __init__(self, feature_idx: _Optional[int] = ..., edges: _Optional[_Union[NdArray, _Mapping]] = ..., inner_edges: _Optional[_Union[NdArray, _Mapping]] = ..., n_bins: _Optional[int] = ...) -> None: ...

class SplitDecision(_message.Message):
    __slots__ = ("node_id", "feature_idx", "threshold", "gain", "left_child_id", "right_child_id")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    FEATURE_IDX_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    GAIN_FIELD_NUMBER: _ClassVar[int]
    LEFT_CHILD_ID_FIELD_NUMBER: _ClassVar[int]
    RIGHT_CHILD_ID_FIELD_NUMBER: _ClassVar[int]
    node_id: int
    feature_idx: int
    threshold: float
    gain: float
    left_child_id: int
    right_child_id: int
    def __init__(self, node_id: _Optional[int] = ..., feature_idx: _Optional[int] = ..., threshold: _Optional[float] = ..., gain: _Optional[float] = ..., left_child_id: _Optional[int] = ..., right_child_id: _Optional[int] = ...) -> None: ...

class Model(_message.Message):
    __slots__ = ("initial_prediction", "learning_rate", "trees")
    INITIAL_PREDICTION_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    TREES_FIELD_NUMBER: _ClassVar[int]
    initial_prediction: float
    learning_rate: float
    trees: _containers.RepeatedCompositeFieldContainer[Tree]
    def __init__(self, initial_prediction: _Optional[float] = ..., learning_rate: _Optional[float] = ..., trees: _Optional[_Iterable[_Union[Tree, _Mapping]]] = ...) -> None: ...

class Tree(_message.Message):
    __slots__ = ("root",)
    ROOT_FIELD_NUMBER: _ClassVar[int]
    root: TreeNode
    def __init__(self, root: _Optional[_Union[TreeNode, _Mapping]] = ...) -> None: ...

class TreeNode(_message.Message):
    __slots__ = ("split", "leaf")
    SPLIT_FIELD_NUMBER: _ClassVar[int]
    LEAF_FIELD_NUMBER: _ClassVar[int]
    split: SplitNode
    leaf: LeafNode
    def __init__(self, split: _Optional[_Union[SplitNode, _Mapping]] = ..., leaf: _Optional[_Union[LeafNode, _Mapping]] = ...) -> None: ...

class SplitNode(_message.Message):
    __slots__ = ("feature_idx", "threshold", "left", "right")
    FEATURE_IDX_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    feature_idx: int
    threshold: float
    left: TreeNode
    right: TreeNode
    def __init__(self, feature_idx: _Optional[int] = ..., threshold: _Optional[float] = ..., left: _Optional[_Union[TreeNode, _Mapping]] = ..., right: _Optional[_Union[TreeNode, _Mapping]] = ...) -> None: ...

class LeafNode(_message.Message):
    __slots__ = ("value",)
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: float
    def __init__(self, value: _Optional[float] = ...) -> None: ...
