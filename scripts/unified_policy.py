# ==============================================================================
# UNIFIED_POLICY.PY - Centralized & Decentralized Actor-Critic for Multi-UAV
# ==============================================================================
# This is the FALLBACK policy used when custom rl_policy_{c/d}{N}.py is not found.
# ==============================================================================

import torch
import torch.nn as nn
import torch.distributions as distributions
import numpy as np


class UnifiedActor(nn.Module):
    """Decentralized Actor: parameter-shared, processes local 15D obs per UAV."""

    def __init__(self, obs_dim=15, action_dim=3, hidden_dim=256, num_uavs=2):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_uavs = num_uavs

        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.mean_net = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim),
            nn.Tanh(),
        )

        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -1.0)

    def forward(self, obs):
        features = self.feature_net(obs)
        mean = self.mean_net(features)

        # FIX: Clamp log_std to prevent std explosion or near-zero stagnation
        log_std = torch.clamp(self.actor_logstd, min=-3.0, max=-0.5)
        std = torch.exp(log_std).expand_as(mean)
        return mean, std

    def get_dist(self, obs):
        mean, std = self.forward(obs)
        return distributions.Normal(mean, std)


class UnifiedCritic(nn.Module):
    """Centralized Critic: sees full global state."""

    def __init__(self, global_state_dim=41, hidden_dim=512):
        super().__init__()
        self.global_state_dim = global_state_dim

        self.value_net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, global_state):
        return self.value_net(global_state)


class CentralizedActor(nn.Module):
    """Centralized Actor: single network sees all UAV states, outputs joint actions."""

    def __init__(self, global_state_dim=41, total_action_dim=12, hidden_dim=512, num_uavs=4):
        super().__init__()
        self.global_state_dim = global_state_dim
        self.total_action_dim = total_action_dim
        self.num_uavs = num_uavs

        self.feature_net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.mean_net = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, total_action_dim),
            nn.Tanh(),
        )

        self.actor_logstd = nn.Parameter(torch.ones(1, self.total_action_dim) * -1.0)

    def forward(self, global_state):
        features = self.feature_net(global_state)
        mean = self.mean_net(features)

        # FIX: Added clamp for centralized mode as well
        log_std = torch.clamp(self.actor_logstd, min=-3.0, max=-0.5)
        std = torch.exp(log_std).expand_as(mean)
        return mean, std

    def get_dist(self, global_state):
        mean, std = self.forward(global_state)
        return distributions.Normal(mean, std)


class UnifiedPPOPolicy(nn.Module):
    """
    Complete PPO Policy supporting centralized and decentralized execution.
    Dimensions auto-configure from unified_config.
    """

    def __init__(self, num_uavs=None, execution_mode=None,
                 local_obs_dim=None, hidden_dim=256):
        super().__init__()

        # Auto-detect from unified_config if not provided
        try:
            from unified_config import NUM_UAVS as CFG_NUM_UAVS
            from unified_config import EXECUTION_MODE as CFG_EXECUTION_MODE
        except ImportError:
            CFG_NUM_UAVS = 2
            CFG_EXECUTION_MODE = "decentralized"

        self.num_uavs = num_uavs if num_uavs is not None else CFG_NUM_UAVS
        self.execution_mode = execution_mode if execution_mode is not None else CFG_EXECUTION_MODE
        # 15 base dims + 6 per OTHER UAV (relative pos+vel) so decentralized
        # actors can perceive teammates — see build_local_obs_batch. Must match
        # unified_config.LOCAL_ACTOR_OBS_DIM.
        self.local_obs_dim = (
            local_obs_dim if local_obs_dim is not None
            else 15 + 6 * (self.num_uavs - 1)
        )
        local_obs_dim = self.local_obs_dim
        self.local_action_dim = 3
        self.total_action_dim = self.num_uavs * 3
        # Global state: 6*N + 17 (matches unified_world.py get_obs)
        self.global_state_dim = 6 * self.num_uavs + 17
        self.obs_dim = self.global_state_dim

        if self.execution_mode == "centralized":
            self.actor = CentralizedActor(
                global_state_dim=self.global_state_dim,
                total_action_dim=self.total_action_dim,
                hidden_dim=hidden_dim * 2,
                num_uavs=self.num_uavs
            )
        else:
            self.actor = UnifiedActor(
                obs_dim=local_obs_dim,
                action_dim=self.local_action_dim,
                hidden_dim=hidden_dim,
                num_uavs=self.num_uavs
            )

        self.critic = UnifiedCritic(self.global_state_dim, hidden_dim * 2)
        self._init_weights()
        self._apply_hover_bias()

    def _compute_hover_bias(self):
        """Estimate the per-UAV steady-state upward action (in the [-1, 1]
        action range, pre-tanh bias) needed just to hold the coupled
        UAV+payload system level once the cable is loaded — i.e. the control
        thrust needed to counter the cable-tension reaction, on top of the
        env's own static hover thrust which already cancels the UAV's own
        weight (see unified_world._simulate_uavs).

        This ONLY biases where the actor's action distribution starts;
        gradients still update it freely during training, and nothing about
        the environment's physics changes. That's deliberate — see the note
        in unified_world._simulate_uavs on why doing this via a static
        physics-side thrust term is unstable (it can run away whenever the
        cable goes slack). Biasing the policy's initial *output* instead
        just gives exploration a much better starting point.
        """
        try:
            from unified_config import CONTROL_SENSITIVITY, GRAVITY
        except ImportError:
            CONTROL_SENSITIVITY, GRAVITY = 6.0, 9.81
        # Nominal payload mass matches physics_level 0 (the curriculum's
        # starting point, ~0.9-1.1 kg) — exactly when this head start
        # matters most for exploration; later curriculum stages widen the
        # mass range and the policy will have already learned to adapt by
        # then.
        nominal_payload_mass = 1.0
        tension_share = nominal_payload_mass * GRAVITY / max(1, self.num_uavs)
        # Leave headroom below the actor's max thrust for the correction PPO
        # still needs to learn (tracking, lateral control, etc.), and clip
        # away from ±1 so tanh isn't saturated at the bias itself.
        target_action = float(np.clip(tension_share / CONTROL_SENSITIVITY, 0.0, 0.85))
        return float(np.arctanh(target_action))

    def _apply_hover_bias(self):
        """Set the Fz component(s) of the actor's final output-layer bias to _compute_hover_bias()."""
        try:
            bias_val = self._compute_hover_bias()

            # --- FIX: Recursively find the absolute final linear layer in the actor architecture ---
            final_linear = None
            for module in self.actor.modules():
                if isinstance(module, nn.Linear):
                    # Check if this layer outputs to our primary action dims
                    if module.out_features in (self.local_action_dim, self.total_action_dim):
                        final_linear = module

            if final_linear is not None and final_linear.bias is not None:
                with torch.no_grad():
                    if self.execution_mode == "centralized":
                        # Joint action is [Fx0, Fy0, Fz0, Fx1, Fy1, Fz1, ...]
                        final_linear.bias[2::3] = bias_val
                        print(f">>> [POLICY] Successfully applied Centralized hover-bias: {bias_val:.4f}")
                    else:
                        # Local action is [Fx, Fy, Fz] for this one UAV
                        final_linear.bias[2] = bias_val
                        print(f">>> [POLICY] Successfully applied Decentralized hover-bias: {bias_val:.4f}")
            else:
                print(">>> [POLICY] Target output layer not found via recursive scan!")
        except Exception as e:
            print(f">>> [POLICY] Could not apply hover-bias init, leaving default zero-init. ({e})")

    def _init_weights(self):
        """
        Orthogonal initialization with layer-aware gains.
        Hidden layers: gain=sqrt(2) for ReLU stability.
        Final actor output layers: gain=0.01 for near-zero initial actions.
        """
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                # Identify final actor output layers (before Tanh)
                # Pattern: actor.mean_net.N where N is the last layer index
                is_final_actor = (
                        'mean_net' in name
                        and isinstance(m, nn.Linear)
                        and m.out_features in (self.local_action_dim, self.total_action_dim)
                )

                if is_final_actor:
                    # Small gain: start policy near zero actions for better exploration
                    nn.init.orthogonal_(m.weight, gain=0.01)
                else:
                    # Standard ReLU gain for hidden layers
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_actor_dist(self, obs):
        """Get action distribution (required by unified_eval)."""
        return self.actor.get_dist(obs)

    def get_critic_value(self, global_state):
        """Get state value (required by unified_eval)."""
        return self.critic(global_state)

    def evaluate_actor(self, obs):
        """
        Evaluate actor for a SINGLE UAV's local observation.
        Returns (dist, entropy) where entropy is pre-computed.
        """
        dist = self.get_actor_dist(obs)
        try:
            entropy = dist.entropy().sum(dim=-1)
        except (AttributeError, NotImplementedError):
            entropy = torch.zeros(obs.shape[0], device=obs.device)
        return dist, entropy

    def evaluate_critic(self, global_state):
        return None, self.critic(global_state)

    def forward(self, global_state):
        """
        Forward pass returning (dist, value).
        For centralized: dist is joint action distribution over all UAVs.
        For decentralized: dist is joint distribution built from per-UAV actors.
        """
        value = self.critic(global_state)

        if self.execution_mode == "centralized":
            dist = self.actor.get_dist(global_state)
        else:
            batch_size = global_state.shape[0]
            joint_mean = global_state.new_zeros(batch_size, self.total_action_dim)
            joint_std = global_state.new_zeros(batch_size, self.total_action_dim)

            for uav_idx in range(self.num_uavs):
                local_obs = build_local_obs_batch(
                    global_state, uav_idx, self.num_uavs, global_state.device
                )
                local_dist = self.actor.get_dist(local_obs)
                start = uav_idx * self.local_action_dim
                end = start + self.local_action_dim
                joint_mean[:, start:end] = local_dist.mean
                joint_std[:, start:end] = local_dist.stddev

            # Block-diagonal joint covariance: UAVs are independent given global state
            dist = distributions.Normal(joint_mean, joint_std)

        return dist, value


# ==============================================================================
# OBSERVATION BUILDER HELPERS
# ==============================================================================

def build_local_obs_batch(global_obs_batch, uav_idx, num_uavs, device='cpu'):
    batch_size = global_obs_batch.shape[0]
    num_neighbors = num_uavs - 1
    local_dim = 15 + 6 * num_neighbors
    local_obs = torch.zeros((batch_size, local_dim), dtype=torch.float32, device=device)

    start_col = uav_idx * 6
    own_pos = global_obs_batch[:, start_col:start_col + 3]
    own_vel = global_obs_batch[:, start_col + 3:start_col + 6]

    payload_start = 6 * num_uavs
    payload_pos = global_obs_batch[:, payload_start:payload_start + 3]

    # --- FIX: Desired 120-degree formation geometry around payload ---
    # NOTE: These hard-coded offsets (0.6, 0.8) are what the policy was trained on.
    # They intentionally differ from unified_config.FORMATION_TARGET_OFFSETS.
    # Do not change without retraining.
    formation_radius = 0.6  # Horizontal distance from payload center (meters)
    angle = uav_idx * (2.0 * np.pi / num_uavs)
    target_offset_x = formation_radius * np.cos(angle)
    target_offset_y = formation_radius * np.sin(angle)
    target_offset_z = 0.8   # Vertical cable height (meters)

    # Compute error relative to nominal hover formation rather than payload center
    local_obs[:, 0] = (own_pos[:, 0] - payload_pos[:, 0]) - target_offset_x
    local_obs[:, 1] = (own_pos[:, 1] - payload_pos[:, 1]) - target_offset_y
    local_obs[:, 2] = (own_pos[:, 2] - payload_pos[:, 2]) - target_offset_z

    # Velocity and remaining payload features stay standard
    local_obs[:, 3:6] = own_vel
    local_obs[:, 6:15] = global_obs_batch[:, payload_start:payload_start + 9]

    # Neighbor relative offsets remain unchanged...
    write_idx = 15
    for j in range(num_uavs):
        if j == uav_idx:
            continue
        other_col = j * 6
        other_pos = global_obs_batch[:, other_col:other_col + 3]
        other_vel = global_obs_batch[:, other_col + 3:other_col + 6]
        local_obs[:, write_idx:write_idx + 3] = own_pos - other_pos
        local_obs[:, write_idx + 3:write_idx + 6] = own_vel - other_vel
        write_idx += 6

    return local_obs


def build_all_local_obs(global_obs_batch, num_uavs, device='cpu'):
    """Build local observations for ALL UAVs at once."""
    return [build_local_obs_batch(global_obs_batch, i, num_uavs, device)
            for i in range(num_uavs)]


def build_joint_action_decentralized(policy, global_obs_batch, num_uavs, device='cpu'):
    """
    Build joint action vector from decentralized actors.

    Returns:
        env_actions:      Clamped actions to send to environment [B, N*3]
        sampled_actions:  Raw sampled actions for PPO log_prob   [B, N*3]
        total_log_prob:   Sum of log probs across UAVs           [B]
        total_entropy:    Sum of entropies across UAVs           [B]
    """
    batch_size = global_obs_batch.shape[0]

    # Two separate tensors: raw samples for PPO, clamped for environment
    sampled_actions = global_obs_batch.new_zeros(batch_size, num_uavs * 3)
    env_actions = global_obs_batch.new_zeros(batch_size, num_uavs * 3)
    total_log_prob = torch.zeros(batch_size, device=device)
    total_entropy = torch.zeros(batch_size, device=device)

    for uav_idx in range(num_uavs):
        local_obs = build_local_obs_batch(global_obs_batch, uav_idx, num_uavs, device)
        result = policy.evaluate_actor(local_obs)

        # Handle both (dist, entropy) and (dist,) return patterns
        if isinstance(result, tuple) and len(result) >= 2:
            dist, entropy = result[0], result[1]
        else:
            dist = result[0] if isinstance(result, tuple) else result
            entropy = None

        action = dist.sample()  # Raw sample from policy
        log_prob = dist.log_prob(action).sum(dim=-1)  # Log prob of RAW sample
        action_env = torch.clamp(action, -1.0, 1.0)  # Clamped for environment

        start = uav_idx * 3
        sampled_actions[:, start:start + 3] = action  # Store raw for PPO
        env_actions[:, start:start + 3] = action_env  # Store clamped for env
        total_log_prob += log_prob

        # Defensive entropy accumulation
        if entropy is not None:
            total_entropy += entropy
        else:
            try:
                total_entropy += dist.entropy().sum(dim=-1)
            except (AttributeError, NotImplementedError):
                pass  # entropy unavailable, stays zero

    return env_actions, sampled_actions, total_log_prob, total_entropy


# ==============================================================================
# DECENTRALIZED PPO UPDATE
# ==============================================================================

def ppo_update_decentralized(
        policy, global_obs_batch, sampled_actions, old_log_prob, advantages, returns,
        num_uavs, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01
):
    """
    Compute PPO loss for decentralized execution.

    Args:
        policy: UnifiedPPOPolicy instance
        global_obs_batch: [B, global_state_dim] global states from rollout
        sampled_actions: [B, N*3] raw (unclamped) actions stored during rollout
        old_log_prob: [B] log probabilities from rollout
        advantages: [B] advantage estimates
        returns: [B] discounted returns
        num_uavs: number of UAVs
    """
    batch_size = global_obs_batch.shape[0]
    device = global_obs_batch.device

    # Critic: centralized, takes global state
    values = policy.get_critic_value(global_obs_batch).squeeze(-1)
    vf_loss = nn.functional.mse_loss(values, returns)

    # Actor: decentralized, must loop over UAVs and reconstruct local observations
    total_new_log_prob = torch.zeros(batch_size, device=device)
    total_entropy = torch.zeros(batch_size, device=device)

    for uav_idx in range(num_uavs):
        # Reconstruct local observation for this UAV
        local_obs = build_local_obs_batch(global_obs_batch, uav_idx, num_uavs, device)

        # Evaluate this UAV's actor
        dist, entropy = policy.evaluate_actor(local_obs)

        # Extract this UAV's action slice from the joint sampled_actions
        start = uav_idx * 3
        end = start + 3
        uav_action = sampled_actions[:, start:end]

        # Accumulate
        total_new_log_prob += dist.log_prob(uav_action).sum(dim=-1)
        total_entropy += entropy

    # Normalize advantages (standard practice for stability)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO clipped objective
    ratio = torch.exp(total_new_log_prob - old_log_prob)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    pg_loss = -torch.min(surr1, surr2).mean()  # Negative because we minimize loss

    # Total loss: policy + value - entropy bonus
    loss = pg_loss + vf_coef * vf_loss - ent_coef * total_entropy.mean()

    return loss, pg_loss, vf_loss, total_entropy.mean()
