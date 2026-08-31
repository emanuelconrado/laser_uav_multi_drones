# ==============================================================================
# UNIFIED_CONFIG.PY - Multi-UAV System Configuration (Centralized + Decentralized)
# ==============================================================================
# FUSION OF: All 2/3/4 UAV configurations with execution mode selection
# ==============================================================================

# ═══════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION SECTION — EDIT THESE VALUES
# ═══════════════════════════════════════════════════════════════════════════════
import os

# --- CORE UAV CONFIGURATION ---
NUM_UAVS = int(os.environ.get("NUM_UAVS", 2)) # Set to 2, 3, or 4
EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "decentralized")
# --- EXECUTION MODE SELECTION ---
# "centralized"    : Single policy sees all UAV states, outputs joint actions (N×3)
# "decentralized"  : CTDE - Parameter-shared actors, centralized critic (local obs only)
#EXECUTION_MODE = "decentralized"  # <-- USER CONFIGURABLE

# --- SAFETY & TRAJECTORY SWITCHES (SE = Safety/Execution) ---
# Set to True to enable, False to disable
SE_NMPC_SHIELD = False              # <-- NMPC Safety Shield: intervenes during danger zones
SE_ENABLE_PPO_TRAJECTORY = False    # <-- PPO Trajectory Planning: macro-action lookahead
SE_USE_FCDEKF = True                # <-- FCDEKF state estimator: swaps in estimated payload
                                     #     state (see FCDEKF_START_EPISODE below) instead of
                                     #     ground truth, once training has progressed enough.

# Alias for backward compatibility (do not modify)
USE_NMPC_SHIELD = SE_NMPC_SHIELD
ENABLE_PPO_TRAJECTORY = SE_ENABLE_PPO_TRAJECTORY

# --- MODULE SELECTION ---
# AUTO_SELECT_MODULE = True  → auto-pick based on EXECUTION_MODE + NUM_UAVS
# AUTO_SELECT_MODULE = False → use CUSTOM_MODULE_NAME below
AUTO_SELECT_MODULE = True

# Manual override (only used when AUTO_SELECT_MODULE = False)
CUSTOM_FCDEKF_MODULE = "fcdekf_c2"      # e.g., "fcdekf_d3", "fcdekf_c4"
CUSTOM_POLICY_MODULE = "rl_policy_c2"   # e.g., "rl_policy_d3", "rl_policy_c4"

# Auto-derived module names (do not modify)
_MODE_LETTER = "c" if EXECUTION_MODE == "centralized" else "d"
AUTO_FCDEKF_MODULE = f"fcdekf_{_MODE_LETTER}{NUM_UAVS}"
AUTO_POLICY_MODULE = f"rl_policy_{_MODE_LETTER}{NUM_UAVS}"

# Final selected modules
FCDEKF_MODULE = AUTO_FCDEKF_MODULE if AUTO_SELECT_MODULE else CUSTOM_FCDEKF_MODULE
POLICY_MODULE = AUTO_POLICY_MODULE if AUTO_SELECT_MODULE else CUSTOM_POLICY_MODULE

# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-DERIVED DIMENSIONS — DO NOT MODIFY BELOW
# ═══════════════════════════════════════════════════════════════════════════════

LOCAL_ACTOR_OBS_DIM = 15 + 6 * (NUM_UAVS - 1)
# Own relative-pos-to-payload(3) + own_vel(3) + payload_pos(3) + payload_vel(3) +
# theta(2) + tension(1) = 15, PLUS each other UAV's relative_pos(3) + relative_vel(3).
# Without this, decentralized actors have zero information about teammates and
# cannot condition on avoiding them (see unified_world.get_local_obs).
# Global state dim = N UAVs * (pos 3D + vel 3D) + payload_pos(3) + payload_vel(3) +
#                    tracking_error(3) + target_vel(3) + theta(2) + tension(1) + wind(1) + time(1)
#                    = 6*N + 17
GLOBAL_CRITIC_STATE_DIM = 6 * NUM_UAVS + 17  # 29 for N=2, 35 for N=3, 41 for N=4
POLICY_ACTION_DIM = NUM_UAVS * 3  # 6 for N=2, 9 for N=3, 12 for N=4
LOCAL_ACTION_DIM = 3  # Per UAV: [Fx, Fy, Fz]

# Execution-mode-specific dimensions
if EXECUTION_MODE == "centralized": #Fixed 
    ACTOR_OBS_DIM = GLOBAL_CRITIC_STATE_DIM  # Actor sees everything
else:
    ACTOR_OBS_DIM = LOCAL_ACTOR_OBS_DIM  # Actor sees only local state

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

NUM_ENVS = 1024
EPISODES = 10000
MAX_STEPS = 512
GAMMA = 0.99
PPO_EPOCHS = 10
BATCH_SIZE = 16384    # was 256 — scale with NUM_ENVS*MAX_STEPS, ~32 minibatches/epoch
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.25


TARGET_KL = 0.06  # Increased from 0.015 to allow proper policy updates

# Smoothed initial learning rates to keep KL divergence under control
if NUM_UAVS == 2:
    INITIAL_LR = 1.5e-4  # Lowered from 3e-4 to prevent immediate KL spikes
    LR_MIN     = 1e-5
elif NUM_UAVS == 3:
    INITIAL_LR = 8e-5
    LR_MIN     = 1e-5
elif NUM_UAVS == 4:
    INITIAL_LR = 5e-5
    LR_MIN     = 5e-6
else:
    INITIAL_LR = 1.5e-4
    LR_MIN     = 1e-5

# --- STATE ESTIMATION (FCDEKF) ---
# Episode at which rollouts switch from ground-truth payload state to
# FCDEKF-estimated state (only takes effect if SE_USE_FCDEKF is True and the
# FCDEKF_MODULE import succeeds). Training starts on clean ground truth so
# the policy can first learn the task itself, then the estimator's noise/lag
# is introduced once the policy is competent enough to absorb it.
FCDEKF_START_EPISODE = int(0.2 * EPISODES)  # e.g. episode 2000 of 10000


# --- EXPLORATION ---
INITIAL_LOGSTD = 0.0      # log(1.0) = 0
LOGSTD_MIN = -1.0         # std ~0.37
LOGSTD_DECAY_RATE = 0.002

# ═══════════════════════════════════════════════════════════════════════════════
# CURRICULUM CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

INITIAL_CURRICULUM_LEVEL = 0.2
MAX_CURRICULUM_LEVEL = 1.5
CURRICULUM_STEP = 0.05
SUCCESS_WINDOW = 100

# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT & PHYSICS
# ═══════════════════════════════════════════════════════════════════════════════

DT = 0.02
MAX_EPISODE_TIME = 20.0
MAX_EPISODE_STEPS = int(MAX_EPISODE_TIME / DT)

UAV_BASE_MASS = 0.5
GRAVITY = 9.81
CABLE_K = 200.0   # was 10.0 — stiffened for realistic cm-scale strain
CABLE_D = 60.0    # was 8.5 — re-tuned for near-critical damping at new k
CONTROL_SENSITIVITY = 15.0  # <-- reverted to Copia value; must stay paired with formation_scale=0.5 in unified_world.py (see note below)

# --- ACTUATOR DAMPER ---
DAMPER_STEPS = 15
DAMPER_FACTOR = 0.30

# --- FORMATION ---
# BASE_RADIUS is the actual formation_scale used by unified_world.py
# (UnifiedGenesisWorld.formation_scale). It MUST stay paired with
# CONTROL_SENSITIVITY above, per that constant's comment. Previously
# unified_world.py hardcoded formation_scale = 1.5 directly, ignoring this
# constant entirely (BASE_RADIUS was dead config) — that silently detuned
# the force scale vs. formation size pairing and both hurt formation
# tightness/safety and confused the tracking task. Now unified_world.py
# imports and uses this value directly (see formation_scale assignment).
BASE_RADIUS = 0.5

# Nominal taut-cable length used to derive each UAV's target vertical
# offset above the payload. Matches UnifiedGenesisWorld.cable_length's
# default. Used both by unified_world.py's own per-episode vertical_offset
# calc (dynamic, uses the live cable_length) and by FORMATION_TARGET_OFFSETS
# below (static approximation, for the decentralized actor's observation).
NOMINAL_CABLE_LENGTH = 1.0

# Unitless formation templates — same numbers as
# UnifiedGenesisWorld.FORMATION_OFFSETS, duplicated here (as plain tuples,
# no numpy/torch dependency) so this module stays a lightweight, dependency-
# free single source of truth that both unified_world.py and
# unified_policy.py can agree on without importing each other.
_FORMATION_OFFSETS_UNITLESS = {
    2: [(-0.5, 0.0), (0.5, 0.0)],
    3: [(-0.5, -0.288675), (0.5, -0.288675), (0.0, 0.57735)],
    4: [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5)],
}


def _formation_target_offset(uav_idx, num_uavs):
    """(dx, dy, dz) target offset of UAV `uav_idx` relative to the payload,
    matching unified_world's real formation geometry (BASE_RADIUS-scaled
    template) and its taut-cable vertical offset formula
    (vert = sqrt(cable_length^2 - horiz^2))."""
    ox, oy = _FORMATION_OFFSETS_UNITLESS[num_uavs][uav_idx]
    dx, dy = ox * BASE_RADIUS, oy * BASE_RADIUS
    horiz = (dx ** 2 + dy ** 2) ** 0.5
    dz = max(0.0, NOMINAL_CABLE_LENGTH ** 2 - horiz ** 2) ** 0.5
    return (dx, dy, dz)


# FORMATION_TARGET_OFFSETS[num_uavs][uav_idx] -> (dx, dy, dz) offset from
# payload. Consumed by unified_policy.build_local_obs_batch so the
# decentralized actor's "am I in formation" observation feature is always
# consistent with the actual geometry the reward function scores against —
# previously that function used a hardcoded generic circular formation
# (radius=0.6, z=0.8) that didn't match the real N=2/4 formations at all,
# which fed the decentralized actor a wrong/noisy self-position error signal
# and was a likely cause of poor decentralized convergence.
FORMATION_TARGET_OFFSETS = {
    n: [_formation_target_offset(i, n) for i in range(n)]
    for n in (2, 3, 4)
}

# --- OBSTACLE CONFIGURATION ---
CORRIDOR_WALL_FREQ = 0.8
CORRIDOR_WIDTH = 2.0
GATE_SPACING = 4.0
NUM_GATES = 8

# --- TRAJECTORY ---
BASE_OMEGA = 2.0 * 3.14159 / 24.0
FIGURE8_WARMUP = 4.0

# ═══════════════════════════════════════════════════════════════════════════════
# REWARD WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

W_TRACKING = 1.0
W_FORMATION = 1.0
W_SWING = 1.0
W_JERK = 1.0
W_OVERSHOOT = 1.0
W_PROGRESS = 8.0

# ═══════════════════════════════════════════════════════════════════════════════
# SAFETY & BOUNDARIES
# ═══════════════════════════════════════════════════════════════════════════════

BOUNDARY_LIMIT = 8.0
BOUNDARY_MARGIN = 3.0
SAFETY_FLOOR = 0.1
CEILING_LIMIT = 10.0

# Episode "success" (see unified_world._check_termination) fires when
# tracking error drops below max(SUCCESS_THRESHOLD_FLOOR, BASE_SUCCESS_THRESHOLD
# - PHYSICS_THRESHOLD_DECAY * physics_level) and holds with low velocity for
# >2s. Previously this was hardcoded in unified_world.py as
# max(0.03, 0.3 - 0.05*physics_level) — i.e. 30cm at physics_level 0 — and
# didn't read these config constants at all, so episodes (and the docking
# bonus) rewarded "close enough" at 20-30cm instead of the ~10cm precision
# actually wanted. Both are now tied to config and tightened to a 10cm target.
BASE_SUCCESS_THRESHOLD = 0.10   # 10 cm, was 0.3 (30 cm)
PHYSICS_THRESHOLD_DECAY = 0.01  # was 0.05 — a 0.3-scale decay is too aggressive against a 0.10 base
SUCCESS_THRESHOLD_FLOOR = 0.05  # was a bare 0.03 magic number inline in unified_world.py

# Distance at which the one-time docking bonus fires in unified_world._compute_reward.
# Was hardcoded to 0.2 (20cm); tightened to match the 10cm precision target.
DOCKING_BONUS_THRESHOLD = 0.10

# Reported by unified_eval.py as an explicit "within target" success-rate
# metric (separate from the stage-relative CURRICULUM_TRACKING_THRESHOLD
# below, which governs curriculum pacing, not final precision).
PRECISION_TARGET_M = 0.10

# ═══════════════════════════════════════════════════════════════════════════════
# NMPC SHIELD CONFIGURATION (active when SE_NMPC_SHIELD = True)
# ═══════════════════════════════════════════════════════════════════════════════

NMPC_ENTERING_DANGER_ZONE = {
    "theta_threshold": 0.1745,  # 10 degrees in radians
    "altitude_threshold": 0.5,
    "min_step": 5,
}

# Trajectory planning horizon (active when SE_ENABLE_PPO_TRAJECTORY = True)
TRAJECTORY_HORIZON_STEPS = 5

# ═══════════════════════════════════════════════════════════════════════════════
# FILE PATHS & LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = "models"
CHECKPOINT_PREFIX = f"unified_{EXECUTION_MODE}_{NUM_UAVS}uav"

LOG_EVERY_N_EPISODES = 2
CHECKPOINT_EVERY_N_EPISODES = 50

# --- PLAIN-LANGUAGE EPISODE EXPLANATIONS ---
# Every N episodes, print a short, human-readable narrative of where the
# payload/UAVs ended up and which reward terms are driving the score.
# This only prints occasionally and reuses data already produced by the
# rollout, so it does not add meaningful overhead to training.
ENABLE_EPISODE_EXPLANATIONS = True
EXPLAIN_EVERY_N_EPISODES = 100

# --- EPISODE MOVIE RECORDING ---
# Every N episodes, run one extra rollout with the current (deterministic)
# policy and save it as a short video showing the payload/UAV trajectories.
# Runs in a separate single env outside the training rollout, so it never
# touches the training buffers and only costs time on the rare episode it
# actually fires on.
SAVE_EPISODE_MOVIES = True          # <-- ON by default
MOVIE_EVERY_N_EPISODES = 500
MOVIE_DIR = os.path.join(MODEL_DIR, "movies")
MOVIE_FPS = 20

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE CURRICULUM ORDER
# ═══════════════════════════════════════════════════════════════════════════════

STAGE_ORDER = [
    "free",
    "wind",
    "physics_stress",
    "slow_line",
    "fast_line",
    "large_line",
    "small_circle",
    "slow_circle",
    "fast_circle",
    "figure-8",
    "lemniscate",
    "spline",
    "nmpc_active_shield"
]

STAGE_SUCCESS_RATES = {
    "free": 0.70,
    "wind": 0.70,
    "physics_stress": 0.75,
    "slow_line": 0.80,
    "fast_line": 0.80,
    "large_line": 0.80,
    "small_circle": 0.75,
    "slow_circle": 0.80,
    "fast_circle": 0.80,
    "figure-8": 0.85,
    "lemniscate": 0.85,
    "spline": 0.85,
    "nmpc_active_shield": 0.90
}

# ═══════════════════════════════════════════════════════════════════════════════
# CURRICULUM SUCCESS METRIC (stage-relative, used by unified_train.py)
# ═══════════════════════════════════════════════════════════════════════════════
# A rollout counts as a curriculum "success" when its mean tracking error over
# the whole rollout is below the stage's threshold below. This replaces a flat
# reward>50 check, since reward scale/weights differ a lot per stage (see
# STAGE_REWARD_CONFIG / the weight blocks in unified_world._compute_reward),
# so a single absolute reward number is not comparable across stages.
CURRICULUM_TRACKING_THRESHOLD = {
    "free": 2.5,
    "hover": 2.5,
    "wind": 1.5,
    "physics_stress": 1.2,
    "slow_line": 0.35,
    "fast_line": 0.40,
    "large_line": 0.45,
    "small_circle": 0.50,
    "slow_circle": 0.55,
    "fast_circle": 0.60,
    "figure-8": 0.65,
    "lemniscate": 0.65,
    "spline": 0.65,
    "nmpc_active_shield": 0.40,
}
DEFAULT_CURRICULUM_TRACKING_THRESHOLD = 0.50

STAGE_REWARD_CONFIG = {
    "free": {"tracking": 5.0, "formation": 0.2, "swing": 0.01, "jerk": 0.0, "overshoot": 0.0, "progress": 15.0},
    "hover": {"tracking": 5.0, "formation": 0.2, "swing": 0.01, "jerk": 0.0, "overshoot": 0.0, "progress": 15.0},
    "wind": {"tracking": 1.5, "formation": 0.5, "swing": 0.2, "jerk": 0.1, "overshoot": 0.0, "progress": 8.0},
    "physics_stress": {"tracking": 1.2, "formation": 0.8, "swing": 0.8, "jerk": 0.5, "overshoot": 0.2, "progress": 8.0},
    "default": {"tracking": 4.0, "formation": 0.3, "swing": 0.05, "jerk": 0.05, "overshoot": 0.05, "progress": 12.0}
}

# ═══════════════════════════════════════════════════════════════════════════════
# PACKET LOSS SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

PACKET_LOSS_PROB = 0.3
PACKET_LOSS_PROTECTED_CHANNELS = {
    "uav_pos_vel": True,      # Always protect UAV position/velocity
    "tracking_error": True,   # Always protect tracking error
}

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

UAV_COLORS = [
    (1.0, 0.0, 0.0),    # Red - UAV 0
    (0.0, 0.0, 1.0),    # Blue - UAV 1
    (1.0, 1.0, 0.0),    # Yellow - UAV 2
    (0.0, 1.0, 1.0),    # Cyan - UAV 3
]
PAYLOAD_COLOR = (0.0, 1.0, 0.0)  # Green
UAV_RADIUS = 0.12
PAYLOAD_SIZE = (0.25, 0.25, 0.25)
