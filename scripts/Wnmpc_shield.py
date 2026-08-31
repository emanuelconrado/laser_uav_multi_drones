# ==============================================================================
# WNMPC_SHIELD.PY - SOFT-CONSTRAINT CEM SHIELD (N-UAV, N=2/3/4)  [FIXED v4]
# ==============================================================================
#   1. DECENTRALIZED cheap gate now only checks TRUE termination conditions
#      (out-of-bounds, floor, tension>50, obstacle).  Swing>25° and UAV-UAV
#      separation <0.26 m are soft reward penalties, NOT terminations, so the
#      pure PPO policy learned to brush against them.  Enforcing them as hard
#      limits caused false-positive interventions on every figure-8 turn.
#   2. CENTRALIZED path uses the CEM.
# ==============================================================================
import copy
import itertools
import numpy as np
import torch

from unified_config import SAFETY_FLOOR, UAV_RADIUS, GRAVITY, CONTROL_SENSITIVITY

try:
    from unified_policy import build_local_obs_batch
except Exception:
    build_local_obs_batch = None

_NON_STATE_ATTRS = {"scene", "payload_vis"}

# --- Hard limits (only for true termination prediction) -----------------------
THETA_HARD_LIMIT   = np.deg2rad(25.0)   # still used by CENTRALIZED CEM cost
MIN_UAV_SEPARATION = 2.0 * UAV_RADIUS   # 0.24 m
TENSION_HARD_LIMIT = 50.0

# --- Safety margins -----------------------------------------------------------
SEP_SAFETY_MARGIN     = 0.02
TENSION_SAFETY_MARGIN = 5.0

# --- Soft-constraint penalty weight (centralized only) ------------------------
CONSTRAINT_PENALTY_W = 1e4


def _uav_position_pairs(world):
    positions = [getattr(world, f"uav{i}_pos") for i in range(world.num_uavs)]
    return itertools.combinations(positions, 2)


def _min_uav_separation(world):
    dists = [np.linalg.norm(pi - pj) for pi, pj in _uav_position_pairs(world)]
    return min(dists) if dists else float("inf")


# ------------------------------------------------------------------
# FAST SNAPSHOT / RESTORE
# ------------------------------------------------------------------
def _fast_copy(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], np.ndarray):
        return type(value)(v.copy() for v in value)
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _snapshot_world_state(world):
    skip = set(_NON_STATE_ATTRS)
    skip.update(f"uav{i}_vis" for i in range(world.num_uavs))
    snapshot = {}
    for key, value in world.__dict__.items():
        if key in skip:
            continue
        snapshot[key] = _fast_copy(value)
    return snapshot


def _restore_world_state(world, snapshot):
    for key, value in snapshot.items():
        if isinstance(value, np.ndarray):
            setattr(world, key, value.copy())
        elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], np.ndarray):
            setattr(world, key, type(value)(v.copy() for v in value))
        else:
            setattr(world, key, copy.deepcopy(value))


def compute_emergency_brake_action(policy_action, num_uavs):
    """Lift-preserving emergency brake."""
    brake = np.zeros_like(policy_action)
    hover_z = float(np.clip(GRAVITY / max(1, num_uavs) / CONTROL_SENSITIVITY, 0.0, 0.85))
    for i in range(num_uavs):
        brake[i * 3 + 2] = max(hover_z, policy_action[i * 3 + 2] * 0.6)
        brake[i * 3]     = policy_action[i * 3]     * 0.2
        brake[i * 3 + 1] = policy_action[i * 3 + 1] * 0.2
    return np.clip(brake, -1.0, 1.0)


def _adaptive_horizon(world):
    current_distance = np.linalg.norm(world.payload_pos - world.target)
    if current_distance > 1.5:
        return 8
    elif current_distance > 0.5:
        return 5
    else:
        return 3


def _is_dynamic_stage(world):
    _STATIC = ("free", "hover", "wind", "physics_stress", "nmpc_active_shield")
    stage = getattr(world, "training_stage", getattr(world, "stage", "free"))
    return stage not in _STATIC


# ------------------------------------------------------------------
# DECENTRALIZED PER-UAV SAFETY PROJECTION
# ------------------------------------------------------------------
def _decentralized_safety_projection(world, policy_action):
    """Fast per-UAV action repair.  No centralized optimization."""
    action = policy_action.copy().astype(np.float32)
    num_uavs = world.num_uavs

    # 1. Ensure minimum hover thrust per UAV (prevents floor crash)
    hover_z = float(np.clip(GRAVITY / max(1, num_uavs) / CONTROL_SENSITIVITY, 0.0, 0.85))
    for i in range(num_uavs):
        action[i * 3 + 2] = max(action[i * 3 + 2], hover_z)

    # 2. Gentle repulsion if UAVs are too close (only touch XY; leave Z to policy)
    positions = [getattr(world, f"uav{i}_pos") for i in range(num_uavs)]
    for i in range(num_uavs):
        for j in range(i + 1, num_uavs):
            diff = positions[i] - positions[j]
            dist = np.linalg.norm(diff) + 1e-8
            if dist < (MIN_UAV_SEPARATION + 0.10):   # 0.34 m buffer, not 0.26
                # nudge both UAVs apart in XY only
                repulse = 0.15 * (1.0 - dist / (MIN_UAV_SEPARATION + 0.10)) * diff[:2] / dist
                action[i * 3]     += repulse[0]
                action[i * 3 + 1] += repulse[1]
                action[j * 3]     -= repulse[0]
                action[j * 3 + 1] -= repulse[1]

    # 3. Soft boundary push (inward XY force if near 7 m wall)
    for i in range(num_uavs):
        p = positions[i]
        if abs(p[0]) > 6.5:
            action[i * 3] -= 0.10 * np.sign(p[0])
        if abs(p[1]) > 6.5:
            action[i * 3 + 1] -= 0.10 * np.sign(p[1])

    return np.clip(action, -1.0, 1.0)


# ------------------------------------------------------------------
# POLICY ROLLOUT HELPERS
# ------------------------------------------------------------------
def _get_decentralized_action(world, policy, device):
    """Query each UAV's actor using local observations; return joint action."""
    if build_local_obs_batch is None:
        return None
    obs = world.get_obs()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    joint = np.zeros(world.num_uavs * 3, dtype=np.float32)
    for i in range(world.num_uavs):
        local_obs = build_local_obs_batch(obs_t, i, world.num_uavs, device)
        with torch.no_grad():
            dist = policy.get_actor_dist(local_obs)
            a = torch.clamp(dist.mean, -1.0, 1.0).cpu().numpy().flatten()
        joint[i * 3:(i + 1) * 3] = a[:3]
    return joint


def _get_centralized_action(world, policy, device):
    obs = world.get_obs()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        if hasattr(policy, 'forward'):
            result = policy.forward(obs_t)
            dist = result[0] if isinstance(result, tuple) else result
        elif hasattr(policy, 'get_actor_dist'):
            dist = policy.get_actor_dist(obs_t)
        else:
            dist = policy(obs_t)
        a = torch.clamp(dist.mean, -1.0, 1.0).cpu().numpy().flatten()
    return a


def _get_policy_action(world, policy, device):
    """Route to the correct actor depending on execution_mode."""
    mode = getattr(world, 'execution_mode', 'centralized')
    if mode == 'decentralized':
        a = _get_decentralized_action(world, policy, device)
        if a is not None:
            return a
    return _get_centralized_action(world, policy, device)


def _terminal_value_cost(world, policy=None, device=None):
    if policy is not None and device is not None:
        try:
            obs = world.get_obs()
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if hasattr(policy, 'get_critic_value'):
                    value = policy.get_critic_value(obs_t)
                else:
                    _, value = policy(obs_t)
            return -float(value.squeeze().item())
        except Exception:
            pass
    return 50.0 * np.linalg.norm(world.payload_pos - world.target)


# ------------------------------------------------------------------
# CHEAP GATE
# ------------------------------------------------------------------
def check_action_trajectory(world, action, horizon=3, policy=None, device=None,
                            mode='centralized'):
    """
    If mode == 'decentralized', only check TRUE termination conditions:
      - out of bounds (|x|>8, |y|>8, z>10, z<SAFETY_FLOOR)
      - tension > TENSION_HARD_LIMIT
      - obstacle hit (penalty <= -10)
    We deliberately SKIP swing>25° and UAV separation <0.26 m because those
    are soft reward terms, not terminations, and the decentralized policy
    learned to operate with occasional violations of them.
    """
    rng_state = np.random.get_state()
    master_state = _snapshot_world_state(world)
    is_safe = True

    for step in range(horizon):
        world.step(action)

        # Hard terminations (shared)
        if (world.payload_pos[2] < SAFETY_FLOOR or
                abs(world.payload_pos[0]) > 8.0 or
                abs(world.payload_pos[1]) > 8.0 or
                world.payload_pos[2] > 10.0 or
                world.tension > TENSION_HARD_LIMIT):
            is_safe = False
            break

        # Obstacle hit
        if getattr(world, 'obstacles_enabled', False):
            obs_pen = world._check_obstacle_collision(world.payload_pos)
            if obs_pen <= -10.0:
                is_safe = False
                break

        # Centralized: also guard swing and separation (CEM can handle them)
        if mode == 'centralized':
            if (abs(world.theta_x) > THETA_HARD_LIMIT or
                    abs(world.theta_y) > THETA_HARD_LIMIT or
                    _min_uav_separation(world) < (MIN_UAV_SEPARATION + SEP_SAFETY_MARGIN) or
                    world.tension > (TENSION_HARD_LIMIT - TENSION_SAFETY_MARGIN)):
                is_safe = False
                break

        # Roll out policy for next step if provided
        if policy is not None and device is not None and step < horizon - 1:
            try:
                action = _get_policy_action(world, policy, device)
            except Exception:
                pass

    _restore_world_state(world, master_state)
    np.random.set_state(rng_state)
    return is_safe


# ------------------------------------------------------------------
# CEM SOLVER  (CENTRALIZED ONLY)
# ------------------------------------------------------------------
def run_nmpc_solver(world, policy_action, curriculum_level, physics_level, env_level,
                     num_samples=None, policy=None, device=None, horizon_override=None,
                     iterations=3):
    rng_state = np.random.get_state()

    ACTION_DIM = world.num_uavs * 3
    HORIZON = int(horizon_override) if horizon_override is not None else _adaptive_horizon(world)
    HORIZON = min(HORIZON, 10)

    if num_samples is None:
        num_samples = max(96, ACTION_DIM * 16)
    NUM_ITERATIONS = iterations
    ELITE_RATIO = 0.15
    NUM_ELITES = max(1, int(num_samples * ELITE_RATIO))
    AR_COEF = 0.65
    GAMMA = 0.98

    dynamic = _is_dynamic_stage(world)

    tracking_w          = 18.0 if dynamic else 12.0
    velocity_w          = 4.0  if dynamic else 3.0
    swing_w             = 1.5  if dynamic else 2.0
    swing_rate_w        = 0.8  if dynamic else 1.0
    jerk_w              = 0.3
    effort_w            = 0.02
    policy_deviation_w  = 8.0
    boundary_w          = 12.0 if dynamic else 15.0
    constraint_w        = 2e3  if dynamic else CONSTRAINT_PENALTY_W

    base_sigma = np.clip(
        0.03 + (0.03 * curriculum_level) + (0.03 * physics_level) + (0.02 * env_level),
        0.03, 0.25,
    )

    mean_sequence = np.tile(policy_action, (HORIZON, 1))
    sigma_sequence = np.ones((HORIZON, ACTION_DIM)) * base_sigma

    master_state = _snapshot_world_state(world)
    prev_best_cost = float('inf')

    for iteration in range(NUM_ITERATIONS):
        candidates = [mean_sequence.copy()]
        for _ in range(num_samples - 1):
            raw_noise = np.random.normal(0, sigma_sequence, size=(HORIZON, ACTION_DIM))
            smoothed_noise = np.zeros_like(raw_noise)
            smoothed_noise[0] = raw_noise[0]
            for k in range(1, HORIZON):
                smoothed_noise[k] = (AR_COEF * smoothed_noise[k - 1] +
                                     (1.0 - AR_COEF) * raw_noise[k])
            candidates.append(np.clip(mean_sequence + smoothed_noise, -1.0, 1.0))

        costs = []
        violation_scores = []

        for seq_candidate in candidates:
            _restore_world_state(world, master_state)
            total_cost = 0.0
            worst_violation = 0.0
            tension_limit = TENSION_HARD_LIMIT - TENSION_SAFETY_MARGIN

            prev_action = getattr(world, 'prev_action', None)
            u_prev = (prev_action.copy() if prev_action is not None
                      else np.array(policy_action, dtype=np.float32).copy())

            for k in range(HORIZON):
                current_u_k = seq_candidate[k]

                theta_x_prev, theta_y_prev = world.theta_x, world.theta_y
                world.step(current_u_k)
                theta_x_dot = (world.theta_x - theta_x_prev) / world.dt
                theta_y_dot = (world.theta_y - theta_y_prev) / world.dt

                # ---- SOFT CONSTRAINT VIOLATIONS ----
                v_theta   = max(max(abs(world.theta_x), abs(world.theta_y)) - THETA_HARD_LIMIT, 0.0)
                v_alt     = max(SAFETY_FLOOR - world.payload_pos[2], 0.0)
                v_sep     = max((MIN_UAV_SEPARATION + SEP_SAFETY_MARGIN) - _min_uav_separation(world), 0.0)
                v_tension = max(world.tension - tension_limit, 0.0)

                v_bound = 0.0
                if abs(world.payload_pos[0]) > 7.0:
                    v_bound += abs(world.payload_pos[0]) - 7.0
                if abs(world.payload_pos[1]) > 7.0:
                    v_bound += abs(world.payload_pos[1]) - 7.0
                if world.payload_pos[2] > 9.0:
                    v_bound += world.payload_pos[2] - 9.0

                step_violation = v_theta + v_alt + v_sep + v_tension + v_bound
                worst_violation = max(worst_violation, step_violation)

                # ---- STAGE COST ----
                tracking_error = np.linalg.norm(world.payload_pos - world.target)
                target_vel = getattr(world, "target_vel", np.zeros(3, dtype=np.float32))
                vel_err = np.linalg.norm(world.payload_vel - target_vel)

                swing_penalty       = (world.theta_x ** 2) + (world.theta_y ** 2)
                swing_rate_penalty  = (theta_x_dot ** 2) + (theta_y_dot ** 2)
                jerk_penalty        = np.linalg.norm(current_u_k - u_prev)
                effort_penalty      = np.mean(np.abs(current_u_k))

                bound_prox = 0.0
                if abs(world.payload_pos[0]) > 5.0:
                    bound_prox += abs(world.payload_pos[0]) - 5.0
                if abs(world.payload_pos[1]) > 5.0:
                    bound_prox += abs(world.payload_pos[1]) - 5.0
                if world.payload_pos[2] < 1.0:
                    bound_prox += 1.0 - world.payload_pos[2]
                if world.payload_pos[2] > 7.0:
                    bound_prox += world.payload_pos[2] - 7.0

                dev_penalty = (policy_deviation_w * (0.7 ** k) *
                               np.linalg.norm(current_u_k - policy_action))

                stage_cost = (
                    (tracking_w * tracking_error) +
                    (velocity_w * vel_err) +
                    (swing_w * swing_penalty) +
                    (swing_rate_w * swing_rate_penalty) +
                    (jerk_w * jerk_penalty) +
                    (effort_w * effort_penalty) +
                    (boundary_w * bound_prox) +
                    dev_penalty +
                    (constraint_w * step_violation)
                )
                total_cost += (GAMMA ** k) * stage_cost
                u_prev = current_u_k.copy()

            if worst_violation < 0.5:
                terminal_cost = _terminal_value_cost(world, policy=policy, device=device)
                total_cost += (GAMMA ** HORIZON) * terminal_cost * 10.0

            costs.append(total_cost)
            violation_scores.append(worst_violation)

        costs = np.array(costs)

        best_cost = np.min(costs)
        if iteration > 0:
            improvement = (prev_best_cost - best_cost) / (abs(prev_best_cost) + 1e-8)
            if improvement < 0.01:
                break
        prev_best_cost = best_cost

        sorted_indices = np.argsort(costs)
        elite_indices = sorted_indices[:NUM_ELITES]
        elite_sequences = [candidates[idx] for idx in elite_indices]
        elite_costs = costs[elite_indices]

        beta = np.min(elite_costs)
        weights = np.exp(-1.0 * (elite_costs - beta) / (np.std(elite_costs) + 1e-4))
        weights /= np.sum(weights)

        updated_mean = np.zeros_like(mean_sequence)
        for w_idx, elite_seq in enumerate(elite_sequences):
            updated_mean += weights[w_idx] * elite_seq

        updated_sigma = np.zeros_like(sigma_sequence)
        for w_idx, elite_seq in enumerate(elite_sequences):
            updated_sigma += weights[w_idx] * ((elite_seq - updated_mean) ** 2)

        mean_sequence = 0.7 * mean_sequence + 0.3 * updated_mean
        sigma_sequence = 0.8 * sigma_sequence + 0.2 * np.sqrt(updated_sigma)
        sigma_sequence = np.clip(sigma_sequence, base_sigma * 0.25, 0.50)

    _restore_world_state(world, master_state)
    np.random.set_state(rng_state)

    world.prev_optimal_seq = mean_sequence.copy()
    world.prev_variance_seq = sigma_sequence.copy()

    return mean_sequence[0]


# ------------------------------------------------------------------
# MAIN SHIELD ENTRY POINT
# ------------------------------------------------------------------
def shield_action(
    world,
    policy_action,
    curriculum_level=None,
    physics_level=None,
    env_level=None,
    num_samples=None,
    policy=None,
    device=None,
    horizon_override=None,
    is_training=False,
):
    curriculum_level = (
        getattr(
            world,
            "curriculum_level",
            1.0,
        )
        if curriculum_level is None
        else curriculum_level
    )

    physics_level = (
        getattr(
            world,
            "physics_level",
            1.0,
        )
        if physics_level is None
        else physics_level
    )

    env_level = (
        getattr(
            world,
            "env_level",
            1.0,
        )
        if env_level is None
        else env_level
    )

    mode = getattr(
        world,
        "execution_mode",
        "centralized",
    )

    action_dim = (
        world.num_uavs * 3
    )

    # =====================================================
    # CURRENT-STATE EMERGENCY
    # =====================================================
    #
    # If the centralized system is already outside a hard
    # limit, do not spend hundreds of milliseconds running
    # CEM. Return the emergency brake immediately.
    current_hard_violation = (
        world.payload_pos[2]
        < SAFETY_FLOOR
        or abs(world.payload_pos[0])
        > 8.0
        or abs(world.payload_pos[1])
        > 8.0
        or world.payload_pos[2]
        > 10.0
        or world.tension
        > TENSION_HARD_LIMIT
    )

    if (
        mode == "centralized"
        and current_hard_violation
    ):
        return compute_emergency_brake_action(
            policy_action,
            world.num_uavs,
        )

    # =====================================================
    # DECENTRALIZED PATH
    # =====================================================
    if mode == "decentralized":
        # 1. Raw policy action.
        if check_action_trajectory(
            world,
            policy_action,
            horizon=2,
            policy=None,
            device=None,
            mode="decentralized",
        ):
            return policy_action

        # 2. Per-UAV projection.
        projected_action = (
            _decentralized_safety_projection(
                world,
                policy_action,
            )
        )

        if check_action_trajectory(
            world,
            projected_action,
            horizon=2,
            policy=None,
            device=None,
            mode="decentralized",
        ):
            return projected_action

        # 3. Emergency brake.
        emergency_action = (
            compute_emergency_brake_action(
                policy_action,
                world.num_uavs,
            )
        )

        if check_action_trajectory(
            world,
            emergency_action,
            horizon=2,
            policy=None,
            device=None,
            mode="decentralized",
        ):
            return emergency_action

        # 4. Absolute last resort.
        #
        # Return the brake even if the current violation
        # cannot disappear inside the prediction horizon.
        return emergency_action

    # =====================================================
    # CENTRALIZED PATH
    # =====================================================

    # 1. Check the raw PPO action.
    if check_action_trajectory(
        world,
        policy_action,
        horizon=3,
        policy=policy,
        device=device,
        mode="centralized",
    ):
        return policy_action

    # 2. Configure and run CEM.
    if is_training:
        samples = (
            max(
                64,
                action_dim * 10,
            )
            if num_samples is None
            else num_samples
        )

        horizon = (
            5
            if horizon_override is None
            else horizon_override
        )

        iterations = (
            2
            if horizon_override is None
            else 3
        )

    else:
        samples = (
            max(
                96,
                action_dim * 16,
            )
            if num_samples is None
            else num_samples
        )

        horizon = (
            6
            if horizon_override is None
            else horizon_override
        )

        iterations = 3

    cem_action = run_nmpc_solver(
        world,
        policy_action,
        curriculum_level,
        physics_level,
        env_level,
        num_samples=samples,
        policy=policy,
        device=device,
        horizon_override=horizon,
        iterations=iterations,
    )

    # 3. Verify the CEM action.
    if check_action_trajectory(
        world,
        cem_action,
        horizon=5,
        policy=policy,
        device=device,
        mode="centralized",
    ):
        return cem_action

    # 4. Lift-preserving emergency brake.
    emergency_action = (
        compute_emergency_brake_action(
            policy_action,
            world.num_uavs,
        )
    )

    if check_action_trajectory(
        world,
        emergency_action,
        horizon=3,
        policy=policy,
        device=device,
        mode="centralized",
    ):
        return emergency_action

    # 5. Absolute last resort.
    #
    # If the initial state is already unsafe, the brake
    # may not eliminate the violation in only three steps.
    # It is still safer than restoring saturated PPO XY.
    return emergency_action
