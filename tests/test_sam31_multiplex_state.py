"""Official SAM 3.1 bucket positions must preserve external object order."""

from __future__ import annotations

import torch

from sam3_onnx_equiv.export._equiv_loader import equiv_sam3_on_path
from sam3_onnx_equiv.path_config import sam31_cpu_source_root


def test_official_bucket_padding_and_roundtrip() -> None:
    with equiv_sam3_on_path(sam31_cpu_source_root()):
        from sam3.model.multiplex_utils import MultiplexController

        controller = MultiplexController(multiplex_count=16)
        controller.eval()
        state = controller.get_state(
            num_valid_entries=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            random=False,
            object_ids=[101, 202],
        )
    assert state.assignments == [[0, 1] + [-1] * 14]
    assert state.object_ids == [101, 202]
    values = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    bucket = state.mux(values)
    assert bucket.shape == (1, 16, 2)
    assert torch.count_nonzero(bucket[:, 2:]) == 0
    torch.testing.assert_close(state.demux(bucket), values)

    state.add_objects(list(range(2, 17)), object_ids=list(range(300, 315)), allow_new_buckets=True)
    assert state.num_buckets == 2
    assert state.assignments[1] == [16] + [-1] * 15
    values = torch.arange(17, dtype=torch.float32).unsqueeze(-1)
    torch.testing.assert_close(state.demux(state.mux(values)), values)
