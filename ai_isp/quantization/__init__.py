"""V6.1 固定混合精度 LSQ+ 与 Dynamic Affine 审计组件。"""

from .dynamic_affine import QuantTensorSpec, audit_dynamic_affine_equivalence, integer_dynamic_affine_reference

from .lsqplus_qat import (
    # V4 compatibility diagnostic only; V6.1 pipeline never branches on it.
    FilmPrecisionGateResult,
    LearnableFakeQuant,
    QatPolicy,
    audit_film_quantization,
    audit_qat_model,
    calibrate_qat_model,
    configure_qat_phase,
    iter_quantizers,
    prepare_qat_model,
    prepare_qdq_export,
)

__all__ = [
    "LearnableFakeQuant",
    "QuantTensorSpec",
    "FilmPrecisionGateResult",
    "QatPolicy",
    "audit_film_quantization",
    "audit_dynamic_affine_equivalence",
    "audit_qat_model",
    "calibrate_qat_model",
    "configure_qat_phase",
    "iter_quantizers",
    "prepare_qat_model",
    "prepare_qdq_export",
    "integer_dynamic_affine_reference",
]
