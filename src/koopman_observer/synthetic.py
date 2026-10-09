"""Linear-Gaussian feature-increment trajectories with optional cross-coupling."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def _spectral_scale(matrix, radius):
    eigs = jnp.linalg.eigvals(matrix)
    current = jnp.max(jnp.abs(eigs))
    return matrix * (radius / jnp.maximum(current, 1e-8))


def sample_dynamics(key, rank, coupled=True, radius=0.45):
    """Joint increment operator, bias, and noise covariance.

    The state is v = concat(u, y). When coupled is False the operator is
    block diagonal, so the deep-to-shallow block A_yu is zero. When coupled
    is True, A_uy and A_yu are nonzero and the noise cross-covariance lets
    shallow residuals predict deep residuals.
    """
    k_uu, k_uy, k_yu, k_yy, k_b = jax.random.split(key, 5)
    eye = jnp.eye(rank)
    A_uu = _spectral_scale(jax.random.normal(k_uu, (rank, rank)), radius)
    A_yy = _spectral_scale(jax.random.normal(k_yy, (rank, rank)), radius)
    if coupled:
        A_uy = 0.55 * jax.random.normal(k_uy, (rank, rank)) / jnp.sqrt(rank)
        A_yu = 0.35 * jax.random.normal(k_yu, (rank, rank)) / jnp.sqrt(rank)
        cross = 0.85
    else:
        A_uy = jnp.zeros((rank, rank))
        A_yu = jnp.zeros((rank, rank))
        cross = 0.0
    top = jnp.concatenate([A_uu, A_uy], axis=1)
    bot = jnp.concatenate([A_yu, A_yy], axis=1)
    A = jnp.concatenate([top, bot], axis=0)
    if coupled:
        A = _spectral_scale(A, radius)
    b = 0.05 * jax.random.normal(k_b, (2 * rank,))
    block = jnp.concatenate(
        [jnp.concatenate([eye, cross * eye], axis=1), jnp.concatenate([cross * eye, eye], axis=1)],
        axis=0,
    )
    cov = 0.35 * block
    return A, b, cov


def _orthonormal_frame(key, channels, rank):
    raw = jax.random.normal(key, (channels, rank))
    Q, _ = jnp.linalg.qr(raw)
    return Q[:, :rank]


def generate_trajectories(
    key,
    n_traj,
    length,
    C_d,
    C_s,
    rank,
    coupled=True,
    sigma_d=1.2,
    sigma_s=0.8,
    dynamics=None,
):
    """Feature trajectories whose increments follow an affine Gaussian model.

    Returns d (n_traj, length + 1, C_d), s (n_traj, length + 1, C_s), and a
    dict with the latent operator. length is the number of increments.
    The block-diagonal variant is coupled=False, which forces A_yu = 0.
    Pass dynamics from a previous call to draw a held-out set from the same law.
    """
    k_dyn, k_pd, k_ps, k_v0, k_d0, k_s0, k_eps = jax.random.split(key, 7)
    if dynamics is None:
        A, b, cov = sample_dynamics(k_dyn, rank, coupled=coupled)
        P_d = _orthonormal_frame(k_pd, C_d, rank)
        P_s = _orthonormal_frame(k_ps, C_s, rank)
        sigma_d = jnp.asarray(sigma_d)
        sigma_s = jnp.asarray(sigma_s)
    else:
        A = dynamics["A"]
        b = dynamics["b"]
        cov = dynamics["cov"]
        P_d = dynamics["P_d"]
        P_s = dynamics["P_s"]
        sigma_d = dynamics["sigma_d"]
        sigma_s = dynamics["sigma_s"]
        coupled = dynamics["coupled"]
    chol = jnp.linalg.cholesky(cov)
    v = [jax.random.normal(k_v0, (n_traj, 2 * rank))]
    for t in range(1, length):
        k_eps, k_step = jax.random.split(k_eps)
        eps = jax.random.normal(k_step, (n_traj, 2 * rank)) @ chol.T
        v.append(v[-1] @ A.T + b + eps)
    v = jnp.stack(v, axis=1)
    u = v[:, :, :rank]
    y = v[:, :, rank:]
    d0 = jax.random.normal(k_d0, (n_traj, C_d))
    s0 = jax.random.normal(k_s0, (n_traj, C_s))
    delta_d = sigma_d * (u @ P_d.T)
    delta_s = sigma_s * (y @ P_s.T)
    d = jnp.concatenate([d0[:, None, :], d0[:, None, :] + jnp.cumsum(delta_d, axis=1)], axis=1)
    s = jnp.concatenate([s0[:, None, :], s0[:, None, :] + jnp.cumsum(delta_s, axis=1)], axis=1)
    info = {
        "A": A,
        "b": b,
        "cov": cov,
        "P_d": P_d,
        "P_s": P_s,
        "sigma_d": sigma_d,
        "sigma_s": sigma_s,
        "coupled": coupled,
        "A_yu": A[rank:, :rank],
    }
    return d, s, info


def generate_calibration_holdout(
    key,
    n_cal,
    n_holdout,
    length,
    C_d,
    C_s,
    rank,
    coupled=True,
):
    """Calibration and held-out trajectories from one increment law."""
    k_cal, k_te = jax.random.split(key)
    d_cal, s_cal, info = generate_trajectories(
        k_cal,
        n_cal,
        length,
        C_d,
        C_s,
        rank,
        coupled=coupled,
    )
    d_te, s_te, _ = generate_trajectories(
        k_te,
        n_holdout,
        length,
        C_d,
        C_s,
        rank,
        coupled=coupled,
        dynamics=info,
    )
    return d_cal, s_cal, d_te, s_te, info
