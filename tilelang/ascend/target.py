from __future__ import annotations

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_detector, register_target_normalizer


def _target_ffi_api():
    from tilelang import _ffi_api

    return _ffi_api


def _make_ascend_target(target_dict: dict | None = None) -> Target:
    target_dict = dict(target_dict or {})
    target_dict["kind"] = "ascend"
    return Target(target_dict)


def _make_pto_target() -> Target:
    target_dict = {"kind": "ascend"}
    # The native kind supplies Ascend semantics; "pto" selects PTO codegen,
    # while "ascend" preserves the target kind's default feature key.
    target_dict["keys"] = ["pto", "ascend"]
    return Target(target_dict)


def _make_hivm_target() -> Target:
    target_dict = {"kind": "ascend"}
    # HIVM reuses the "ascend" target kind and is selected by the "hivm" key
    # (goal section 4: no new target kind). "ascend" is kept alongside it for
    # the same reason PTO keeps it -- the native kind's default feature key.
    target_dict["keys"] = ["hivm", "ascend"]
    return Target(target_dict)


def target_is_ascend(target: Target) -> bool:
    """Return whether *target* uses the Ascend architecture."""
    return _target_ffi_api().TargetIsAscend(target)


def target_is_pto(target: Target) -> bool:
    """Return whether *target* selects PTO within the Ascend architecture."""
    return target_is_ascend(target) and "pto" in target.keys


def target_is_hivm(target: Target) -> bool:
    """Return whether *target* selects the HIVM backend within Ascend.

    The HIVM predicate declines a target that also carries "pto": the two
    selectors are mutually exclusive device backends, and the resolver rejects
    that combination with a named cause before enumerating candidates. Note
    that the PTO predicate deliberately does NOT gain the mirror-image
    exclusion -- see the Challenger's ruling: PTO keeps its existing predicate,
    and the conflict is handled by the explicit guard in the resolver.
    """
    return target_is_ascend(target) and "hivm" in target.keys and "pto" not in target.keys


def target_is_plain_ascend(target: Target) -> bool:
    """Return whether *target* selects AscendC rather than PTO or HIVM codegen."""
    return target_is_ascend(target) and "pto" not in target.keys and "hivm" not in target.keys


def check_ascend_availability() -> bool:
    try:
        import torch

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def _detect_ascend_target() -> Target | None:
    if check_ascend_availability():
        return _make_ascend_target()
    return None


def normalize_ascend_target(target: TargetLike) -> Target | None:
    if not isinstance(target, str) or target.strip() != "ascend":
        return None

    try:
        return _make_ascend_target()
    except Exception:
        return None


def normalize_pto_target(target: TargetLike) -> Target | None:
    if not isinstance(target, str) or target.strip() != "pto":
        return None

    try:
        return _make_pto_target()
    except Exception:
        return None


def normalize_hivm_target(target: TargetLike) -> Target | None:
    """Accept ``hivm`` as the concise name for the HIVM backend.

    This is an EXPLICIT selection: goal req 8 requires that HIVM never replace
    the default AscendC through auto-detection. There is deliberately no HIVM
    target DETECTOR registered below, so an "auto" target can never land here;
    only a caller that literally asks for "hivm" is normalized.
    """
    if not isinstance(target, str) or target.strip() != "hivm":
        return None

    try:
        return _make_hivm_target()
    except Exception:
        return None


def normalize_asc_target(target: TargetLike) -> Target | None:
    """Accept ``asc`` as the concise name for the AscendC backend."""
    if isinstance(target, str) and target.strip() == "asc":
        return normalize_ascend_target("ascend")
    return None


register_target_detector("ascend", _detect_ascend_target, override=True)
register_target_normalizer("ascend", normalize_ascend_target, override=True)
register_target_normalizer("asc", normalize_asc_target, override=True)
register_target_normalizer("pto", normalize_pto_target, override=True)
register_target_normalizer("hivm", normalize_hivm_target, override=True)
