"""Equation checks for joint increments, EDMD, gain, and the one-step identity."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from koopman_observer.identify import (
    empirical_second_moments,
    fit_bases_and_scales,
    identify_gain,
    identify_operator,
    joint_increment,
    one_step_reduction,
    proposition_trace,
)


def test_joint_increment_matches_hand_example():
    P_d = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    P_s = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_d = 2.0
    sigma_s = 0.5
    d_k = jnp.array([4.0, 2.0, 9.0])
    d_prev = jnp.array([0.0, 0.0, 1.0])
    s_k = jnp.array([1.0, 3.0])
    s_prev = jnp.array([0.0, 1.0])
    v = joint_increment(d_k, d_prev, s_k, s_prev, P_d, P_s, sigma_d, sigma_s)
    delta_d = d_k - d_prev
    delta_s = s_k - s_prev
    u = P_d.T @ delta_d / sigma_d
    y = P_s.T @ delta_s / sigma_s
    expected = jnp.concatenate([u, y])
    assert jnp.allclose(v, jnp.array([2.0, 1.0, 4.0, 2.0]))
    assert jnp.allclose(v, expected)


def test_large_lambda_shrinks_operator_to_identity():
    key = jax.random.PRNGKey(2)
    k1, k2 = jax.random.split(key)
    v_prev = jax.random.normal(k1, (40, 6))
    v_next = jax.random.normal(k2, (40, 6))
    A, b = identify_operator(v_prev, v_next, lam=1e8)
    assert jnp.allclose(A, jnp.eye(6), atol=1e-5)
    assert jnp.allclose(b, jnp.zeros(6), atol=1e-5)


def test_gain_matches_closed_form():
    key = jax.random.PRNGKey(3)
    rank = 3
    gamma = 0.37
    rho = jax.random.normal(key, (50, 2 * rank))
    L_d, L = identify_gain(rho, rank, gamma)
    U = rho[:, :rank]
    Ys = rho[:, rank:]
    m = rho.shape[0]
    closed = (U.T @ Ys / m) @ jnp.linalg.inv(Ys.T @ Ys / m + gamma * jnp.eye(rank))
    assert jnp.allclose(L_d, closed, atol=1e-8)
    assert jnp.allclose(L[:rank], L_d)
    assert jnp.allclose(L[rank:], jnp.eye(rank))


def test_proposition_trace_identity():
    key = jax.random.PRNGKey(4)
    k_u, k_y = jax.random.split(key)
    rank = 4
    gamma = 0.2
    eps_y = jax.random.normal(k_y, (80, rank))
    eps_u = 0.7 * eps_y + 0.3 * jax.random.normal(k_u, (80, rank))
    lhs, rhs, _ = one_step_reduction(eps_u, eps_y, gamma)
    Sigma_uu, Sigma_uy, Sigma_yy = empirical_second_moments(eps_u, eps_y)
    M = jnp.linalg.inv(Sigma_yy + gamma * jnp.eye(rank))
    L = Sigma_uy @ M
    corrected = eps_u - (L @ eps_y.T).T
    lhs_direct = jnp.mean(jnp.sum(eps_u**2, axis=-1)) - jnp.mean(jnp.sum(corrected**2, axis=-1))
    rhs_direct = jnp.trace(
        Sigma_uy @ M @ (Sigma_yy + 2.0 * gamma * jnp.eye(rank)) @ M @ Sigma_uy.T
    )
    rhs_lib, L_lib = proposition_trace(Sigma_uy, Sigma_yy, gamma)
    del Sigma_uu
    assert jnp.allclose(lhs, lhs_direct, atol=1e-5)
    assert jnp.allclose(rhs, rhs_direct, atol=1e-5)
    assert jnp.allclose(rhs_lib, rhs_direct, atol=1e-5)
    assert jnp.allclose(L_lib, L, atol=1e-5)
    assert jnp.allclose(lhs_direct, rhs_direct, atol=1e-5)
    assert float(lhs_direct) >= -1e-8


def test_bases_are_orthonormal():
    key = jax.random.PRNGKey(8)
    k_d, k_s = jax.random.split(key)
    delta_d = jax.random.normal(k_d, (60, 9))
    delta_s = jax.random.normal(k_s, (60, 5))
    rank = 3
    P_d, P_s, _, _ = fit_bases_and_scales(delta_d, delta_s, rank)
    assert P_d.shape == (9, rank)
    assert P_s.shape == (5, rank)
    assert jnp.allclose(P_d.T @ P_d, jnp.eye(rank), atol=1e-6)
    assert jnp.allclose(P_s.T @ P_s, jnp.eye(rank), atol=1e-6)
