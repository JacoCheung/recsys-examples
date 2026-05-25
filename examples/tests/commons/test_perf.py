import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from commons.utils.perf import DeviceSpec, _calculate_peak_tflops, cal_mfu


def test_gb200_peak_tflops_uses_gb200_specs_before_b200_substring() -> None:
    peak_tflops = _calculate_peak_tflops(
        compute_capability=(10, 0),
        num_sms=160,
        clock_mhz=2100,
        device_name="NVIDIA GB200",
    )

    assert peak_tflops["bf16"] == 2500
    assert peak_tflops["fp16"] == 2500
    assert peak_tflops["fp8"] == 5000


def test_b200_peak_tflops() -> None:
    peak_tflops = _calculate_peak_tflops(
        compute_capability=(10, 0),
        num_sms=148,
        clock_mhz=2000,
        device_name="NVIDIA B200",
    )

    assert peak_tflops["bf16"] == 2250
    assert peak_tflops["fp16"] == 2250
    assert peak_tflops["fp8"] == 4500


def test_gb200_bf16_mfu_does_not_require_cuda_query() -> None:
    spec = DeviceSpec(
        device_index=0,
        device_name="NVIDIA GB200",
        compute_capability=(10, 0),
        architecture="Blackwell",
        num_sms=160,
        gpu_clock_mhz=2100,
        memory_total_gb=186,
        memory_bandwidth_gb_s=8000,
        peak_tflops={"bf16": 2500},
    )

    assert (
        cal_mfu(
            achieved_tflops=5000,
            world_size=2,
            dtype="bf16",
            device_spec=spec,
        )
        == 100.0
    )
