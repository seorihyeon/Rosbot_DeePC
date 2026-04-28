import math

import numpy as np
import pytest

from rosbot_deepc.deepc_solver import DeePCSolver
from rosbot_deepc.utils import (
    encode_yaw_output,
    load_dataset_csv,
    load_reference_csv,
    normalize_yaw_representation,
    unicycle_tracking_law,
    wrap_to_pi,
)


def test_yaw_representation_only_accepts_current_names():
    assert normalize_yaw_representation('wrap') == 'wrap'
    assert normalize_yaw_representation('unwrap') == 'unwrap'
    assert normalize_yaw_representation('trig') == 'trig'

    for value in ('auto', 'wrap_scalar', 'unwrap_scalar'):
        with pytest.raises(ValueError):
            normalize_yaw_representation(value)


def test_encode_yaw_output_shapes_and_wrapping():
    scalar = encode_yaw_output(
        x=1.0,
        y=2.0,
        yaw=4.0,
        v=0.3,
        w=-0.4,
        yaw_representation='wrap',
    )
    trig = encode_yaw_output(
        x=1.0,
        y=2.0,
        yaw=4.0,
        v=0.3,
        w=-0.4,
        yaw_representation='trig',
    )

    assert len(scalar) == 5
    assert scalar[2] == pytest.approx(wrap_to_pi(4.0))
    assert len(trig) == 6
    assert trig[2] == pytest.approx(math.cos(4.0))
    assert trig[3] == pytest.approx(math.sin(4.0))


def test_load_dataset_csv_aligns_shifted_outputs(tmp_path):
    csv_path = tmp_path / 'dataset.csv'
    csv_path.write_text(
        '\n'.join(
            [
                'cmd_v,cmd_w,x,y,yaw,v_meas,w_meas,sim_time_sec',
                '0.1,0.2,0.0,0.0,0.0,0.01,0.02,0.0',
                '0.3,0.4,1.0,0.0,0.5,0.03,0.04,0.1',
                '0.5,0.6,2.0,0.0,1.0,0.05,0.06,0.2',
            ]
        )
    )

    u_data, y_data = load_dataset_csv(
        str(csv_path),
        yaw_representation='wrap',
        y_shift_steps=1,
    )

    np.testing.assert_allclose(u_data, [[0.1, 0.3], [0.2, 0.4]])
    assert y_data.shape == (5, 2)
    np.testing.assert_allclose(y_data[:, 0], [1.0, 0.0, 0.5, 0.03, 0.04])


def test_load_reference_csv_infers_yaw_velocity_and_stop(tmp_path):
    csv_path = tmp_path / 'reference.csv'
    csv_path.write_text('\n'.join(['x,y', '0.0,0.0', '1.0,0.0', '1.0,1.0']))

    ref = load_reference_csv(str(csv_path), dt=0.5, append_final_stop_steps=1)

    assert len(ref) == 4
    assert ref[0].yaw == pytest.approx(0.0)
    assert ref[1].yaw == pytest.approx(math.pi / 2.0)
    assert ref[0].v == pytest.approx(2.0)
    assert ref[0].w == pytest.approx(math.pi)
    assert ref[-1].v == pytest.approx(0.0)
    assert ref[-1].w == pytest.approx(0.0)


def test_unicycle_tracking_law_uses_feedforward_and_clamps():
    cmd_v, cmd_w = unicycle_tracking_law(
        1.0,
        0.2,
        e_x=0.5,
        e_y=-0.25,
        e_psi=0.1,
        kx=0.8,
        ky=1.8,
        kpsi=2.0,
        v_min=0.0,
        v_max=2.0,
        w_min=-1.0,
        w_max=1.0,
    )

    assert cmd_v == pytest.approx(math.cos(0.1) + 0.4)
    assert cmd_w == pytest.approx(0.2 - 0.45 + 0.2)


def test_solver_diag_requires_exact_dimension():
    diag = DeePCSolver._make_diag([1.0, 2.0, 3.0, 4.0, 5.0], 5, 'Q_diag')
    np.testing.assert_allclose(np.diag(diag), [1.0, 2.0, 3.0, 4.0, 5.0])

    with pytest.raises(ValueError):
        DeePCSolver._make_diag([1.0, 2.0, 3.0], 5, 'Q_diag')
    with pytest.raises(ValueError):
        DeePCSolver._make_diag([1.0, -2.0], 2, 'R_diag')
