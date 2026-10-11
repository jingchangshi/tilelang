from pathlib import Path
import hashlib
import json

import tilelang
from tilelang import tvm
from tilelang.engine.lower import _prepare_device_codegen_mod
from examples.ascend.example_vecadd import vector_add

ROOT = (
    Path(__file__).resolve().parent
    / "golden/hivm/vadd_f32_single_tile"
)
ROOT.mkdir(parents=True, exist_ok=True)

# 1. 与实际用例保持完全一致
program = vector_add(524288, mode="simd")

# 2. 运行正常 Ascend lowering，但不编译设备二进制
artifact = tilelang.lower(
    program,
    target="ascend",
    enable_device_compile=False,
)

# 3. 两个重要边界
device_mod = artifact.device_mod
golden = _prepare_device_codegen_mod(device_mod)

# 4. 保存可读的 TIR
(ROOT / "device_mod.tir").write_text(
    device_mod.script(),
    encoding="utf-8",
)
(ROOT / "golden_codegen_input.tir").write_text(
    golden.script(),
    encoding="utf-8",
)

# 5. 保存可机器重放的 IRModule
serialized = tvm.ir.save_json(golden)
(ROOT / "golden_codegen_input.json").write_text(
    serialized,
    encoding="utf-8",
)

# 6. 验证序列化后没有丢失 IR 信息
reloaded = tvm.ir.load_json(serialized)
tvm.ir.assert_structural_equal(golden, reloaded)

# 7. 从独立重放的 Golden 直接调用 AscendC CodeGen
target = golden["main_kernel"].attrs["target"]

ascend_build = tvm.ffi.get_global_func(
    "target.build.tilelang_ascend_without_compile"
)
replay_module = ascend_build(reloaded, target)
replay_source = replay_module.inspect_source()

(ROOT / "ascend_reference.asc").write_text(
    replay_source,
    encoding="utf-8",
)

# 8. 与正常 tilelang.lower 的结果交叉检查
if replay_source != artifact.kernel_source:
    raise AssertionError(
        "Golden replay produced different AscendC source. "
        "Check the target and final CodeGen preparation passes."
    )

# 9. 保存固定用例的输入契约
manifest = {
    "case": "vadd_f32_single_tile",
    "input_boundary": "after _prepare_device_codegen_mod",
    "target": "ascend",
    "kernel": "main_kernel",
    "dtype": "float32",
    "N": 524288,
    "num_cores": 64,
    "elements_per_core": 8192,
    "vector_lanes": 64,
    "vector_iterations": 128,
    "ub_arena_bytes": 196608,
    "ub_offsets_f32": {
        "a": 0,
        "b": 16384,
        "c": 32768,
    },
    "sync_events": [
        ["MTE2_V", 0],
        ["V_MTE3", 0],
    ],
    "golden_sha256": hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest(),
}

(ROOT / "manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print(f"Golden created at: {ROOT.resolve()}")
print("Round-trip IR structural equality: PASS")
print("Independent AscendC replay: PASS")

