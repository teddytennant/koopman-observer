"""Luenberger correction, error identities, and partial-observation episodes."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from koopman_observer.identify import joint_increment, selectors


def split_blocks(A, rank):
    """Split the joint operator into (A_uu, A_uy, A_yu, A_yy)."""
    A_uu = A[:rank, :rank]
    A_uy = A[:rank, rank:]
    A_yu = A[rank:, :rank]
    A_yy = A[rank:, rank:]
    return A_uu, A_uy, A_yu, A_yy


def open_loop_step(hat_v_prev, A, b):
    """One affine step without a shallow correction."""
    return A @ hat_v_prev + b


def observer_step(hat_v_prev, y, A, b, L):
    """Predict, then correct with the observed shallow increment y.

    bar_v = A @ hat_v_prev + b
    nu = y - H @ bar_v
    hat_v = bar_v + L @ nu
    After the step, H @ hat_v = y whenever H @ L = I.
    """
    rank = y.shape[-1]
    _, H = selectors(rank)
    bar_v = A @ hat_v_prev + b
    nu = y - H @ bar_v
    return bar_v + L @ nu


def integrate_deep(hat_d_prev, hat_v, P_d, sigma_d, rank):
    """hat_d = hat_d_prev + sigma_d * (P_d @ (J @ hat_v))."""
    J, _ = selectors(rank)
    return hat_d_prev + sigma_d * (P_d @ (J @ hat_v))


def observer_error(v_k, v_prev, hat_v_prev, A, b, L):
    """Error identity e = (I - L H) (A e_prev + rho).

    e_prev = v_prev - hat_v_prev and rho = v_k - A v_prev - b.
    Returns (e, rho).
    """
    rank = L.shape[1]
    _, H = selectors(rank)
    e_prev = v_prev - hat_v_prev
    rho = v_k - (A @ v_prev + b)
    eye = jnp.eye(v_k.shape[-1], dtype=v_k.dtype)
    e = (eye - L @ H) @ (A @ e_prev + rho)
    return e, rho


def deep_error_recursion(e_u_prev, rho, A, L_d):
    """Deep block of the error when the previous shallow error is zero.

    e_u = (A_uu - L_d A_yu) e_u_prev + (rho_u - L_d rho_y)
    """
    rank = e_u_prev.shape[-1]
    A_uu, _, A_yu, _ = split_blocks(A, rank)
    rho_u = rho[:rank]
    rho_y = rho[rank:]
    return (A_uu - L_d @ A_yu) @ e_u_prev + (rho_u - L_d @ rho_y)


def _operator_at(param, index):
    return param[index]


def run_episode(d, s, model, corrected=True, n_init=2):
    """Roll out one feature trajectory.

    The first n_init transitions are measured in both deep and shallow
    features and initialize hat_d and hat_v. Later steps receive the shallow
    increment only. Operators are per transition: index t maps increment t
    to increment t + 1.

    d is (T + 1, C_d), s is (T + 1, C_s).
    Returns hat_d (T + 1, C_d) and hat_v (T, 2r).
    """
    P_d = model["P_d"]
    P_s = model["P_s"]
    sigma_d = model["sigma_d"]
    sigma_s = model["sigma_s"]
    A = model["A"]
    b = model["b"]
    L = model["L"]
    rank = int(model["rank"])

    T = d.shape[0] - 1
    hat_d = d[0]
    hat_ds = [hat_d]
    hat_vs = []
    hat_v_prev = None
    for k in range(T):
        y = ((s[k + 1] - s[k]) @ P_s) / sigma_s
        if k < n_init:
            u = ((d[k + 1] - d[k]) @ P_d) / sigma_d
            hat_v = jnp.concatenate([u, y], axis=-1)
            hat_d = d[k + 1]
        else:
            op = k - 1
            if corrected:
                hat_v = observer_step(
                    hat_v_prev, y, _operator_at(A, op), _operator_at(b, op), _operator_at(L, op)
                )
            else:
                hat_v = open_loop_step(hat_v_prev, _operator_at(A, op), _operator_at(b, op))
            hat_d = integrate_deep(hat_d, hat_v, P_d, sigma_d, rank)
        hat_vs.append(hat_v)
        hat_ds.append(hat_d)
        hat_v_prev = hat_v
    return jnp.stack(hat_ds, axis=0), jnp.stack(hat_vs, axis=0)


def deep_increment_mse(d, s, model, corrected=True, n_init=2):
    """Mean squared deep-increment error on partial steps.

    d is (N, T + 1, C_d) and s is (N, T + 1, C_s). The score is the mean of
    ||u_hat - u_true||^2 over trajectories and steps k >= n_init, with u in
    the calibration basis.
    """
    P_d = model["P_d"]
    P_s = model["P_s"]
    sigma_d = model["sigma_d"]
    sigma_s = model["sigma_s"]
    rank = int(model["rank"])
    n = d.shape[0]
    total = 0.0
    count = 0
    for i in range(n):
        _, hat_v = run_episode(d[i], s[i], model, corrected=corrected, n_init=n_init)
        v_true = joint_increment(
            d[i, 1:, :],
            d[i, :-1, :],
            s[i, 1:, :],
            s[i, :-1, :],
            P_d,
            P_s,
            sigma_d,
            sigma_s,
        )
        err = hat_v[n_init:, :rank] - v_true[n_init:, :rank]
        total = total + jnp.sum(err * err)
        count = count + err.shape[0]
    return total / count
