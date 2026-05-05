import numpy as np
import pytest

from fldp.strategies.parameters import (
    add_gaussian_noise,
    apply_delta,
    clip_by_l2_norm,
    l2_norm,
    mean_arrays,
    params_to_delta,
)


def test_params_to_delta_and_apply_delta_round_trip() -> None:
    initial = [np.array([1.0, 2.0]), np.array([[3.0]])]
    updated = [np.array([1.5, 1.0]), np.array([[5.0]])]

    delta = params_to_delta(initial, updated)
    reconstructed = apply_delta(initial, delta)

    assert [array.tolist() for array in delta] == [[0.5, -1.0], [[2.0]]]
    assert all(np.allclose(left, right) for left, right in zip(reconstructed, updated))


def test_l2_norm_spans_all_arrays() -> None:
    arrays = [np.array([3.0, 4.0]), np.array([12.0])]

    assert l2_norm(arrays) == pytest.approx(13.0)


def test_clip_by_l2_norm_scales_large_updates_only() -> None:
    arrays = [np.array([3.0, 4.0])]

    clipped = clip_by_l2_norm(arrays, clip_norm=2.5)
    unchanged = clip_by_l2_norm(arrays, clip_norm=10.0)

    assert l2_norm(clipped) == pytest.approx(2.5)
    assert np.allclose(unchanged[0], arrays[0])


def test_mean_arrays_supports_uniform_and_weighted_means() -> None:
    first = [np.array([1.0, 3.0]), np.array([[2.0]])]
    second = [np.array([3.0, 5.0]), np.array([[6.0]])]

    uniform = mean_arrays([first, second])
    weighted = mean_arrays([first, second], weights=[1, 3])

    assert np.allclose(uniform[0], [2.0, 4.0])
    assert np.allclose(uniform[1], [[4.0]])
    assert np.allclose(weighted[0], [2.5, 4.5])
    assert np.allclose(weighted[1], [[5.0]])


def test_add_gaussian_noise_is_deterministic_with_rng_seed() -> None:
    arrays = [np.zeros(3), np.ones((2, 2))]
    first = add_gaussian_noise(arrays, std=0.5, rng=np.random.default_rng(4))
    second = add_gaussian_noise(arrays, std=0.5, rng=np.random.default_rng(4))

    assert all(np.allclose(left, right) for left, right in zip(first, second))
    assert not np.allclose(first[0], arrays[0])


def test_parameter_utilities_reject_incompatible_inputs() -> None:
    with pytest.raises(ValueError, match="empty"):
        l2_norm([])

    with pytest.raises(ValueError, match="same length"):
        params_to_delta([np.array([1.0])], [np.array([1.0]), np.array([2.0])])

    with pytest.raises(ValueError, match="shapes"):
        apply_delta([np.array([1.0])], [np.array([[1.0]])])

    with pytest.raises(ValueError, match="weights"):
        mean_arrays([[np.array([1.0])]], weights=[0])

    with pytest.raises(ValueError, match="std"):
        add_gaussian_noise([np.array([1.0])], std=-1.0, rng=np.random.default_rng(1))
