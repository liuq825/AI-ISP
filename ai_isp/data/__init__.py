"""RAW 数据、Condition 和 SIDD 适配模块。"""

from .condition_v2 import ConditionMetadata, encode_condition_v2, validate_ryyb_release_condition_v2
from .noise_model import synthesize_correlated_ryyb_noise
from .pack_raw import normalize_post_blc_lsc_packed_raw, pack_bayer, unpack_bayer
from .ryyb_contract import (
    AdmissionPolicy,
    BUFFER_CONTRACT_VERSION,
    RAW_DOMAIN_STATE,
    RyybFrameDescriptor,
    RyybManifestRecord,
    RyybReleaseDataRequirements,
    pack_ryyb,
    reconstruct_and_unpack_ryyb,
    unpack_ryyb,
    validate_ai_admission,
    validate_release_dataset_requirements,
)
from .ryyb_dataset import BalancedCameraSampler, RyybRawPatchDataset

__all__ = [
    "ConditionMetadata",
    "AdmissionPolicy",
    "BUFFER_CONTRACT_VERSION",
    "BalancedCameraSampler",
    "RyybRawPatchDataset",
    "RyybFrameDescriptor",
    "RyybManifestRecord",
    "RyybReleaseDataRequirements",
    "RAW_DOMAIN_STATE",
    "encode_condition_v2",
    "normalize_post_blc_lsc_packed_raw",
    "pack_bayer",
    "pack_ryyb",
    "reconstruct_and_unpack_ryyb",
    "synthesize_correlated_ryyb_noise",
    "unpack_bayer",
    "unpack_ryyb",
    "validate_ai_admission",
    "validate_release_dataset_requirements",
    "validate_ryyb_release_condition_v2",
]
