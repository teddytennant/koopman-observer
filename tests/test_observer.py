"""Equation checks for the observer step, error recursion, and synthetic gap."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from koopman_observer.identify import calibrate_trajectories, selectors
from koopman_observer.observer import (
    deep_error_recursion,
    deep_increment_mse,
    observer_error,
    observer_step,
    split_blocks,
)
from koopman_observer.synthetic import generate_calibration_holdout, generate_trajectories


def _luenberger(L_d):
    rank = L_d.shape[0]
    return jnp.concatenate([L_d, jnp.eye(rank)], axis=0)


def test_correction_matches_shallow_observation():
    key = jax.random.PRNGKey(5)
    rank = 3
    dim = 2 * rank
    keys = jax.random.split(key, 4)
    A = 0.2 * jax.random.normal(keys[0], (dim, dim))
    b = 0.1 * jax.random.normal(keys[1], (dim,))
    L_d = 0.3 * jax.random.normal(keys[2], (rank, rank))
    L = _luenberger(L_d)
    hat_v_prev = jax.random.normal(keys[3], (dim,))
    y = jax.random.normal(jax.random.PRNGKey(51), (rank,))
    hat_v = observer_step(hat_v_prev, y, A, b, L)
    _, H = selectors(rank)
    assert jnp.allclose(H @ hat_v, y, atol=1e-6)


def test_error_identity_and_deep_recursion():
    key = jax.random.PRNGKey(6)
    rank = 3
    dim = 2 * rank
    keys = jax.random.split(key, 6)
    A = 0.25 * jax.random.normal(keys[0], (dim, dim))
    b = 0.05 * jax.random.normal(keys[1], (dim,))
    L_d = 0.4 * jax.random.normal(keys[2], (rank, rank))
    L = _luenberger(L_d)
    v_prev = jax.random.normal(keys[3], (dim,))
    e_u_prev = jax.random.normal(keys[4], (rank,))
    # Previous shallow error is zero, as it is after a correction step.
    hat_v_prev = v_prev - jnp.concatenate([e_u_prev, jnp.zeros(rank)])
    v_k = jax.random.normal(keys[5], (dim,))
    y = v_k[rank:]
    hat_v = observer_step(hat_v_prev, y, A, b, L)
    e_direct = v_k - hat_v
    e_formula, rho = observer_error(v_k, v_prev, hat_v_prev, A, b, L)
    _, H = selectors(rank)
    eye = jnp.eye(dim)
    rho_direct = v_k - (A @ v_prev + b)
    e_inline = (eye - L @ H) @ (A @ (v_prev - hat_v_prev) + rho_direct)
    e_u = deep_error_recursion(e_u_prev, rho_direct, A, L_d)
    A_uu, _, A_yu, _ = split_blocks(A, rank)
    e_u_inline = (A_uu - L_d @ A_yu) @ e_u_prev + (rho_direct[:rank] - L_d @ rho_direct[rank:])
    assert jnp.allclose(e_formula, e_direct, atol=1e-5)
    assert jnp.allclose(e_inline, e_direct, atol=1e-5)
    assert jnp.allclose(e_u, e_direct[:rank], atol=1e-5)
    assert jnp.allclose(e_u_inline, e_direct[:rank], atol=1e-5)
    assert jnp.allclose(e_direct[rank:], jnp.zeros(rank), atol=1e-5)
    assert jnp.allclose(rho, rho_direct, atol=1e-5)


def test_corrected_mse_below_open_loop_on_coupled_trajectories():
    key = jax.random.PRNGKey(7)
    rank = 2
    length = 6
    d_cal, s_cal, d_te, s_te, info = generate_calibration_holdout(
        key,
        n_cal=400,
        n_holdout=200,
        length=length,
        C_d=6,
        C_s=5,
        rank=rank,
        coupled=True,
    )
    _, _, info_block = generate_trajectories(
        jax.random.PRNGKey(70),
        n_traj=4,
        length=4,
        C_d=6,
        C_s=5,
        rank=rank,
        coupled=False,
    )
    assert info["A_yu"].shape == (rank, rank)
    assert float(jnp.linalg.norm(info["A_yu"])) > 0.0
    assert jnp.allclose(info_block["A_yu"], jnp.zeros((rank, rank)))
    model = calibrate_trajectories(d_cal, s_cal, rank=rank, lam=1e-3, gamma=1e-2)
    mse_corrected = deep_increment_mse(d_te, s_te, model, corrected=True)
    mse_open = deep_increment_mse(d_te, s_te, model, corrected=False)
    assert float(mse_corrected) < float(mse_open)
