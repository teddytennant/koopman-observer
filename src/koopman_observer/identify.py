"""Bases, joint increments, ridge EDMD, and Luenberger gain."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def selectors(rank):
    """Return J and H. J keeps the deep block, H keeps the shallow block."""
    eye = jnp.eye(rank)
    zeros = jnp.zeros((rank, rank))
    J = jnp.concatenate([eye, zeros], axis=1)
    H = jnp.concatenate([zeros, eye], axis=1)
    return J, H


def joint_increment(d_k, d_prev, s_k, s_prev, P_d, P_s, sigma_d, sigma_s):
    """Joint normalized increment v = concat(u, y).

    u = P_d.T @ (d_k - d_{k-1}) / sigma_d
    y = P_s.T @ (s_k - s_{k-1}) / sigma_s
    Leading batch axes are preserved.
    """
    u = ((d_k - d_prev) @ P_d) / sigma_d
    y = ((s_k - s_prev) @ P_s) / sigma_s
    return jnp.concatenate([u, y], axis=-1)


def top_basis(deltas, rank):
    """Top-r eigenvectors of the increment second-moment matrix.

    deltas has shape (M, C). Columns of the result are orthonormal.
    """
    deltas = jnp.asarray(deltas)
    moment = (deltas.T @ deltas) / deltas.shape[0]
    _, evecs = jnp.linalg.eigh(moment)
    return evecs[:, -rank:]


def projected_rms(deltas, P):
    """sigma = sqrt(mean(||P.T @ delta||^2))."""
    projected = deltas @ P
    return jnp.sqrt(jnp.mean(jnp.sum(projected * projected, axis=-1)))


def fit_bases_and_scales(delta_d, delta_s, rank):
    """Orthonormal bases and RMS scales from calibration increments only."""
    P_d = top_basis(delta_d, rank)
    P_s = top_basis(delta_s, rank)
    sigma_d = projected_rms(delta_d, P_d)
    sigma_s = projected_rms(delta_s, P_s)
    return P_d, P_s, sigma_d, sigma_s


def identify_operator(v_prev, v_next, lam):
    """Ridge EDMD affine map (A, b) for one transition.

    W = (Y.T @ X / M + lam * W0) @ inv(X.T @ X / M + lam * I)
    with X = [v_prev, 1], Y = v_next, W0 = [I, 0], W = [A, b].
    """
    v_prev = jnp.asarray(v_prev)
    v_next = jnp.asarray(v_next)
    m, dim = v_prev.shape
    ones = jnp.ones((m, 1), dtype=v_prev.dtype)
    X = jnp.concatenate([v_prev, ones], axis=1)
    Y = v_next
    eye_dim = jnp.eye(dim, dtype=v_prev.dtype)
    W0 = jnp.concatenate([eye_dim, jnp.zeros((dim, 1), dtype=v_prev.dtype)], axis=1)
    gram = (X.T @ X) / m + lam * jnp.eye(dim + 1, dtype=v_prev.dtype)
    rhs = (Y.T @ X) / m + lam * W0
    W = jnp.linalg.solve(gram, rhs.T).T
    A = W[:, :dim]
    b = W[:, dim]
    return A, b


def residuals(v_prev, v_next, A, b):
    """rho = v_next - A @ v_prev - b, rows are samples."""
    prediction = v_prev @ A.T + b
    return v_next - prediction


def identify_gain(rho, rank, gamma):
    """Luenberger gain from deep and shallow residual blocks.

    L_d = (U.T @ Ys / M) @ inv(Ys.T @ Ys / M + gamma * I)
    L = concat([L_d; I_r]) so that H @ L = I_r.
    """
    rho = jnp.asarray(rho)
    m = rho.shape[0]
    U = rho[:, :rank]
    Ys = rho[:, rank:]
    gram = (Ys.T @ Ys) / m + gamma * jnp.eye(rank, dtype=rho.dtype)
    cross = (U.T @ Ys) / m
    L_d = jnp.linalg.solve(gram, cross.T).T
    L = jnp.concatenate([L_d, jnp.eye(rank, dtype=rho.dtype)], axis=0)
    return L_d, L


def empirical_second_moments(eps_u, eps_y):
    """Sample second moments Sigma_uu, Sigma_uy, Sigma_yy with 1/M normalization."""
    m = eps_u.shape[0]
    Sigma_uu = (eps_u.T @ eps_u) / m
    Sigma_uy = (eps_u.T @ eps_y) / m
    Sigma_yy = (eps_y.T @ eps_y) / m
    return Sigma_uu, Sigma_uy, Sigma_yy


def proposition_trace(Sigma_uy, Sigma_yy, gamma):
    """Right-hand side of the one-step residual reduction identity.

    M = inv(Sigma_yy + gamma I)
    L = Sigma_uy @ M
    trace[Sigma_uy @ M @ (Sigma_yy + 2 gamma I) @ M @ Sigma_yu]
    """
    eye = jnp.eye(Sigma_yy.shape[0], dtype=Sigma_yy.dtype)
    moment = Sigma_yy + gamma * eye
    M = jnp.linalg.inv(moment)
    Sigma_yu = Sigma_uy.T
    rhs = jnp.trace(Sigma_uy @ M @ (Sigma_yy + 2.0 * gamma * eye) @ M @ Sigma_yu)
    L = Sigma_uy @ M
    return rhs, L


def one_step_reduction(eps_u, eps_y, gamma):
    """Empirical deep residual reduction and the trace identity value."""
    Sigma_uu, Sigma_uy, Sigma_yy = empirical_second_moments(eps_u, eps_y)
    rhs, L = proposition_trace(Sigma_uy, Sigma_yy, gamma)
    corrected = eps_u - eps_y @ L.T
    lhs = jnp.mean(jnp.sum(eps_u * eps_u, axis=-1)) - jnp.mean(
        jnp.sum(corrected * corrected, axis=-1)
    )
    del Sigma_uu
    return lhs, rhs, L


def schur_deep_moment(Sigma_uu, Sigma_uy, Sigma_yy):
    """Residual second moment at gamma = 0: Sigma_uu - Sigma_uy inv(Sigma_yy) Sigma_yu."""
    Sigma_yu = Sigma_uy.T
    return Sigma_uu - Sigma_uy @ jnp.linalg.inv(Sigma_yy) @ Sigma_yu


def fit_per_transition(v, lam, gamma):
    """Fit (A_t, b_t, L_t) for each consecutive pair in v of shape (N, T, 2r).

    Operator index t predicts v[:, t + 1] from v[:, t].
    """
    n_ops = v.shape[1] - 1
    rank = v.shape[-1] // 2
    As = []
    bs = []
    Lds = []
    Ls = []
    for t in range(n_ops):
        A, b = identify_operator(v[:, t, :], v[:, t + 1, :], lam)
        rho = residuals(v[:, t, :], v[:, t + 1, :], A, b)
        L_d, L = identify_gain(rho, rank, gamma)
        As.append(A)
        bs.append(b)
        Lds.append(L_d)
        Ls.append(L)
    return jnp.stack(As), jnp.stack(bs), jnp.stack(Lds), jnp.stack(Ls)


def calibrate_trajectories(d, s, rank, lam, gamma):
    """Bases, scales, and per-transition operators from calibration trajectories.

    d has shape (N, T + 1, C_d), s has shape (N, T + 1, C_s).
    Bases and scales use calibration increments only.
    """
    delta_d = (d[:, 1:, :] - d[:, :-1, :]).reshape(-1, d.shape[-1])
    delta_s = (s[:, 1:, :] - s[:, :-1, :]).reshape(-1, s.shape[-1])
    P_d, P_s, sigma_d, sigma_s = fit_bases_and_scales(delta_d, delta_s, rank)
    v = joint_increment(
        d[:, 1:, :],
        d[:, :-1, :],
        s[:, 1:, :],
        s[:, :-1, :],
        P_d,
        P_s,
        sigma_d,
        sigma_s,
    )
    A, b, L_d, L = fit_per_transition(v, lam, gamma)
    return {
        "P_d": P_d,
        "P_s": P_s,
        "sigma_d": sigma_d,
        "sigma_s": sigma_s,
        "A": A,
        "b": b,
        "L_d": L_d,
        "L": L,
        "rank": rank,
    }
