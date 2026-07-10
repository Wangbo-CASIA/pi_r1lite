from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from r1lite_rtc import RtcActionBuffer


def test_action_buffer_smooths_all_dims_during_overlap():
    buffer = RtcActionBuffer()

    buffer.integrate_new_chunk(
        np.array(
            [
                [0.0, 0.0],
                [10.0, 1.0],
            ],
            dtype=np.float32,
        ),
        max_k=0,
    )
    buffer.integrate_new_chunk(
        np.array(
            [
                [100.0, 1.0],
                [200.0, 0.0],
            ],
            dtype=np.float32,
        ),
        max_k=0,
    )

    first_action = buffer.pop_next_action()
    second_action = buffer.pop_next_action()

    np.testing.assert_allclose(first_action, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(second_action, np.array([200.0, 0.0], dtype=np.float32))
