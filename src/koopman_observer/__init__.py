"""Observation-corrected Koopman model of feature increments."""

import jax

jax.config.update("jax_enable_x64", True)

from koopman_observer.identify import (
    calibrate_trajectories,
    empirical_second_moments,
    fit_bases_and_scales,
    fit_per_transition,
    identify_gain,
    identify_operator,
    joint_increment,
    projected_rms,
    proposition_trace,
    residuals,
    selectors,
    top_basis,
)
from koopman_observer.observer import (
    deep_error_recursion,
    deep_increment_mse,
    integrate_deep,
    observer_error,
    observer_step,
    open_loop_step,
    run_episode,
    split_blocks,
)
from koopman_observer.synthetic import generate_calibration_holdout, generate_trajectories

__all__ = [
    "calibrate_trajectories",
    "deep_error_recursion",
    "deep_increment_mse",
    "empirical_second_moments",
    "fit_bases_and_scales",
    "fit_per_transition",
    "generate_calibration_holdout",
    "generate_trajectories",
    "identify_gain",
    "identify_operator",
    "integrate_deep",
    "joint_increment",
    "observer_error",
    "observer_step",
    "open_loop_step",
    "projected_rms",
    "proposition_trace",
    "residuals",
    "run_episode",
    "selectors",
    "split_blocks",
    "top_basis",
]
