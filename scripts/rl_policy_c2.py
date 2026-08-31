# =========================================================
# rl_policy.py5
# FINAL PPO POLICY
# FIXED OBSERVATION DIMENSION
# =========================================================

import torch
import torch.nn as nn

from torch.distributions import Normal

# =========================================================
# PPO POLICY
# =========================================================
class PPOPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # --- FIX 1: Harmonize with centralized get_obs framework (6 * 2 + 17 = 29) ---
        self.obs_dim = 29

        # --- FIX 2: Harmonize with 2-UAV 3D force specifications (2 * 3 = 6) ---
        self.action_dim = 6

        # Actor Network
        self.actor_mean = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
            nn.Tanh(),  # FIXED: Keeps forces bounded cleanly within [-1, 1] for stable updates
        )

        # --- FIX 3: Boost initial PPO exploration variance factor ---
        self.actor_logstd = nn.Parameter(
            torch.ones(self.action_dim) * -0.2
        )

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.FloatTensor(obs)

        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        mean = self.actor_mean(obs)

        # --- FIX 4: Loosen clamping window to protect exploratory entropy ---
        log_std = torch.clamp(self.actor_logstd, -1.5, 0.5)
        std = torch.exp(log_std)

        assert not torch.isnan(std).any()
        assert not torch.isinf(std).any()
        assert not torch.isnan(mean).any()
        assert not torch.isinf(mean).any()

        dist = Normal(mean, std)
        value = self.critic(obs)
        return dist, value

    # =====================================================
    # SAMPLE ACTION
    # =====================================================

    def get_action(

        self,

        obs
    ):

        # =================================================
        # FORWARD
        # =================================================

        dist, value = self.forward(obs)

        # =================================================
        # SAMPLE
        # =================================================

        action = dist.sample()

        # =================================================
        # LOG PROB
        # =================================================

        log_prob = dist.log_prob(
            action
        ).sum(dim=-1)

        # =================================================
        # RETURN
        # =================================================

        return (

            action.detach().cpu().numpy(),

            log_prob.detach(),

            value.detach()
        )

    # =====================================================
    # PPO EVALUATION
    # =====================================================

    def evaluate_action(

        self,

        obs,

        action
    ):

        # =================================================
        # FORWARD
        # =================================================

        dist, value = self.forward(obs)

        # =================================================
        # LOG PROB
        # =================================================

        log_prob = dist.log_prob(
            action
        ).sum(dim=-1)

        # =================================================
        # ENTROPY
        # =================================================

        entropy = dist.entropy().sum(dim=-1)

        return (

            log_prob,

            entropy,

            value
        )