# ==============================================================================
# UNIFIED_WORLD.PY - Multi-UAV Environment (Centralized & Decentralized)
# ==============================================================================
# FUSION OF: genesis_worldP20.py (2-UAV), Tgenesis_worldP14.py (3-UAV),
#            Wgenesis_worldP14.py (4-UAV) + best features from all
# ==============================================================================

import numpy as np
import genesis as gs
import torch

from unified_config import (
    CABLE_K, CABLE_D, UAV_RADIUS, BASE_RADIUS,
    BASE_SUCCESS_THRESHOLD, PHYSICS_THRESHOLD_DECAY, SUCCESS_THRESHOLD_FLOOR,
    DOCKING_BONUS_THRESHOLD,
)


class UnifiedGenesisWorld:
    """
    Unified Multi-UAV Transport Environment supporting 2, 3, or 4 UAVs.

    Supports both centralized and decentralized observation modes.

    Formation geometries:
        N=2: Horizontal line (±0.5 template, scaled by formation_scale)
        N=3: Equilateral triangle (120° separation, template vertices)
        N=4: Regular tetrahedron / square pyramid (template vertices)

    Action space: N UAVs × 3D continuous forces [Fx, Fy, Fz]
    """

    # =========================================================================
    # FORMATION GEOMETRY FACTORY
    # =========================================================================
    # These are UNITLESS templates. Actual spawn positions are computed as:
    #   offset = formation_offsets[i] * formation_scale
    # This makes formation size explicitly configurable without editing templates.
    FORMATION_OFFSETS = {
        2: [
            np.array([-0.5, 0.0, 0.0]),   # Template: unitless
            np.array([0.5, 0.0, 0.0]),    # Template: unitless
        ],
        3: [
            np.array([-0.5, -0.288675, 0.0]),   # Template: unitless
            np.array([0.5, -0.288675, 0.0]),    # Template: unitless
            np.array([0.0, 0.57735, 0.0]),      # Template: unitless
        ],
        4: [
            np.array([0.5, 0.0, 0.0]),    # Template: unitless
            np.array([-0.5, 0.0, 0.0]),   # Template: unitless
            np.array([0.0, 0.5, 0.0]),    # Template: unitless
            np.array([0.0, -0.5, 0.0]),   # Template: unitless
        ]
    }

    def __init__(
        self,
        num_uavs=2,                    # CONFIGURABLE: 2, 3, or 4
        execution_mode="decentralized",  # "centralized" or "decentralized"
        scene=None,
        stage="free",
        visualize=True,
        start_at_target=False,
        physics_level=0,
        env_level=0,
        curriculum_level=0.2
    ):
        # --- Core Configuration ---
        self.num_uavs = max(2, min(4, int(num_uavs)))  # Clamp to 2-4
        self.execution_mode = execution_mode
        self.scene = scene
        self.stage = stage
        self.visualize = visualize
        self.start_at_target = start_at_target
        self.physics_level = physics_level
        self.env_level = env_level
        self.curriculum_level = curriculum_level
        self.is_training = True

        # --- Simulation Parameters ---
        self.dt = 0.02
        self.prev_dist = 5.0
        self.max_episode_time = 20.0
        self.max_episode_steps = int(self.max_episode_time / self.dt)
        self.current_step = 0
        self.elapsed_time = 0.0
        self.speed_multiplier = 1.0
        self.scale_multiplier = 1.0

        # --- Formation Geometry ---
        self.formation_offsets = self.FORMATION_OFFSETS[self.num_uavs]
        # formation_scale: multiplies unitless templates to get actual meter offsets.
        # BUGFIX: this was hardcoded to 1.5 here, silently ignoring
        # unified_config.BASE_RADIUS (which sat unused) and contradicting the
        # comment right below it — CONTROL_SENSITIVITY was tuned assuming
        # formation_scale=0.5 (N=2 separation = 0.5m). Running at 1.5 (3x too
        # wide) desynced control authority from formation size, which hurts
        # both formation-keeping and tracking convergence. Now sourced
        # directly from config so there is exactly one place to change it.
        self.formation_scale = BASE_RADIUS

        # Ideal pairwise separation for every UAV pair, derived directly from the
        # unitless formation geometry and formation_scale. This replaces a single
        # fixed "1.0m for every pair" target, which is only correct for N=2/N=3
        # layouts and is geometrically infeasible for N=4's planar formation.
        self.formation_pair_targets = {}
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                ideal_dist = float(np.linalg.norm(
                    (self.formation_offsets[i] - self.formation_offsets[j]) * self.formation_scale
                ))
                self.formation_pair_targets[(i, j)] = ideal_dist

        # --- Initialize UAV States (dynamic based on N) ---
        for i in range(self.num_uavs):
            setattr(self, f"uav{i}_pos", np.zeros(3, dtype=np.float32))
            setattr(self, f"uav{i}_vel", np.zeros(3, dtype=np.float32))
            setattr(self, f"uav{i}_quat", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
            setattr(self, f"uav{i}_acc", np.zeros(3, dtype=np.float32))
            setattr(self, f"tether{i}_dir", np.zeros(3, dtype=np.float32))

        # --- Payload State ---
        self.payload_pos = np.zeros(3, dtype=np.float32)
        self.payload_vel = np.zeros(3, dtype=np.float32)
        self.payload_acc = np.zeros(3, dtype=np.float32)
        self.payload_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # --- Target & Trajectory ---
        self.target = np.array([1.0, 1.0, 2.5], dtype=np.float32)
        self.prev_target = np.zeros(3, dtype=np.float32)
        self.target_vel = np.zeros(3, dtype=np.float32)
        self.target_acc = np.zeros(3, dtype=np.float32)
        self.start_payload_pos = np.array([-1.0, 0.0, 1.5], dtype=np.float32)
        self.initial_training_target = np.copy(self.target)

        # --- Swing & Tension ---
        self.theta_x = 0.0
        self.theta_y = 0.0
        self.prev_theta_x = 0.0
        self.prev_theta_y = 0.0
        self.swing_x = 0.0
        self.swing_y = 0.0
        self.tension = 9.81

        # --- Physics Parameters ---
        self.payload_mass = 1.0
        self.cable_length = 1.0
        self.cable_k = CABLE_K  # imported from unified_config.py
        self.cable_d = CABLE_D  # imported from unified_config.py
        from unified_config import CONTROL_SENSITIVITY
        self.control_sensitivity = CONTROL_SENSITIVITY
        self.vertical_offsets = [1.0] * num_uavs  # per-UAV; recomputed in reset()

        # --- Environment Flags ---
        self.wind_enabled = False
        self.wind_strength = 0.0
        self.wind_vector = np.zeros(3, dtype=np.float32)
        self.obstacles_enabled = False
        self.failure_enabled = False
        self.packet_loss = 0.0

        # --- State Estimator (FCDEKF) ---
        # Attach an FCDEKF instance externally (env.state_estimator = FCDEKF(...))
        # and flip use_state_estimator on/off to switch obs between ground truth
        # and the estimator's output. See _refresh_state_estimate() /
        # _parse_estimate() below. Left disabled unless explicitly wired up, so
        # this is a no-op for anyone not using the estimator.
        self.state_estimator = None
        self.use_state_estimator = False
        self._current_estimate = None  # cached (pos, vel, theta_x, theta_y, tension) for this step

        # --- Action History ---
        self.prev_action = np.zeros(self.num_uavs * 3, dtype=np.float32)
        self.prev_tracking_error = 5.0
        self.docking_bonus_claimed = False

        # --- Stage Configuration ---
        self._configure_stage()

        self._apply_env_curriculum()
        self._apply_physics_curriculum()

        # --- Visual Objects ---
        self.viewer_initialized = False
        if self.visualize and self.scene is not None:
            self.build_visual_entities()

        # --- Obstacles ---
        self.obstacles = []
        self.obstacle_visuals = []


    # =====================================================================
    # STAGE CONFIGURATION
    # =====================================================================
    def _configure_stage(self):
        """Configure environment based on stage string."""
        if self.stage == "wind":
            self.wind_enabled = True
            self.wind_strength = 0.4
        elif self.stage == "corridor":
            self.wind_enabled = True
            self.wind_strength = 0.3
            self.obstacles_enabled = True
        elif self.stage == "gates":
            self.wind_enabled = True
            self.wind_strength = 0.5
            self.obstacles_enabled = True
        elif self.stage == "communication":
            self.packet_loss = 0.3
        elif self.stage == "failures":
            self.failure_enabled = True

    # =====================================================================
    # SENSOR MEASUREMENT (9D - from 3-UAV/4-UAV best practice)
    # =====================================================================
    def get_measurement(self):
        """Return sensor measurement for EKF (9D unified)."""
        # Build tether force vector
        tether_forces = np.zeros(3, dtype=np.float32)
        for i in range(self.num_uavs):
            t_dir = getattr(self, f"tether{i}_dir", np.zeros(3))
            tether_forces += t_dir

        tether_forces *= (self.tension / max(1, self.num_uavs))

        return np.array([
            self.payload_pos[0], self.payload_pos[1], self.payload_pos[2],
            self.theta_x, self.theta_y, self.tension,
            tether_forces[0], tether_forces[1], tether_forces[2]
        ], dtype=np.float32)

    # =====================================================================
    # FCDEKF STATE ESTIMATION
    # =====================================================================
    # ASSUMED FCDEKF INTERFACE (adjust to match your actual fcdekf_{c/d}{N}.py
    # if it differs — everything else in this file only depends on
    # _refresh_state_estimate()/_parse_estimate(), so you only need to edit
    # the two calls below):
    #   est = FCDEKF(dt=DT)                     # or FCDEKF() with no args
    #   est.reset(payload_pos=..., payload_vel=...)   # or reset() with no args
    #   estimate = est.update(measurement)      # measurement = get_measurement(), 9D
    # `estimate` may be a dict with keys like "payload_pos"/"payload_vel"/
    # "theta_x"/"theta_y"/"tension", or a flat array
    # [pos(3), vel(3), theta_x, theta_y, tension, ...]. See _parse_estimate().
    def _reset_state_estimator(self):
        """Reset the attached FCDEKF's internal filter state at episode start."""
        if self.state_estimator is None:
            self._current_estimate = None
            return
        try:
            self.state_estimator.reset(payload_pos=self.payload_pos, payload_vel=self.payload_vel)
        except TypeError:
            try:
                self.state_estimator.reset()
            except Exception as e:
                print(f">>> [FCDEKF] reset() failed, disabling estimator for this run. ({e})")
                self.use_state_estimator = False
        except AttributeError:
            pass  # estimator has no reset(); assume it re-initializes lazily
        except Exception as e:
            print(f">>> [FCDEKF] reset() failed, disabling estimator for this run. ({e})")
            self.use_state_estimator = False
        self._current_estimate = None

    def _refresh_state_estimate(self):
        """Run the FCDEKF forward one step on this timestep's 9D sensor
        measurement and cache the result. Called once per step()/reset() so
        get_obs()/get_local_obs() (which may be called multiple times per
        step, once per UAV) all see the same, single filter update.
        Falls back to ground truth — and permanently disables the estimator
        for the rest of this run — if anything about the call is wrong.
        """
        if not self.use_state_estimator or self.state_estimator is None:
            self._current_estimate = None
            return
        try:
            measurement = self.get_measurement()
            estimate = self.state_estimator.update(measurement)
            self._current_estimate = self._parse_estimate(estimate)
        except Exception as e:
            print(f">>> [FCDEKF] update() failed ({e}); falling back to ground-truth "
                  f"state and disabling the estimator for the rest of this run.")
            self.use_state_estimator = False
            self._current_estimate = None

    def _parse_estimate(self, estimate):
        """Best-effort parsing of the FCDEKF's return value into
        (pos[3], vel[3], theta_x, theta_y, tension). See interface note above."""
        if isinstance(estimate, dict):
            pos = np.asarray(estimate.get("payload_pos", estimate.get("pos", self.payload_pos)), dtype=np.float32)
            vel = np.asarray(estimate.get("payload_vel", estimate.get("vel", self.payload_vel)), dtype=np.float32)
            theta_x = float(estimate.get("theta_x", self.theta_x))
            theta_y = float(estimate.get("theta_y", self.theta_y))
            tension = float(estimate.get("tension", self.tension))
            return pos, vel, theta_x, theta_y, tension

        arr = np.asarray(estimate, dtype=np.float32).flatten()
        if arr.shape[0] >= 9:
            return arr[0:3], arr[3:6], float(arr[6]), float(arr[7]), float(arr[8])
        elif arr.shape[0] >= 3:
            return arr[0:3], self.payload_vel, self.theta_x, self.theta_y, self.tension
        raise ValueError(f"unrecognized FCDEKF estimate shape {arr.shape}")

    def _get_perceived_payload_state(self):
        """Return (pos, vel, theta_x, theta_y, tension) that observations
        should be built from: the FCDEKF estimate if active, else ground
        truth. Physics/reward code should keep using self.payload_pos etc.
        directly (that's the true state) — only get_obs()/get_local_obs()
        call this, since only the policy's *perception* should be estimated.
        """
        if self._current_estimate is not None:
            return self._current_estimate
        return self.payload_pos, self.payload_vel, self.theta_x, self.theta_y, self.tension

    # =====================================================================
    # VISUAL OBJECTS
    # =====================================================================
    def build_visual_entities(self):
        """Build visual entities for all UAVs and payload."""
        if self.viewer_initialized or self.scene is None:
            return

        colors = [
            (1.0, 0.0, 0.0),    # Red
            (0.0, 0.0, 1.0),    # Blue
            (1.0, 1.0, 0.0),    # Yellow
            (0.0, 1.0, 1.0),    # Cyan
        ]

        for i in range(self.num_uavs):
            setattr(self, f"uav{i}_vis", self.scene.add_entity(
                gs.morphs.Sphere(radius=0.12, pos=(0.0, 0.0, 2.0)),
                surface=gs.surfaces.Default(color=colors[i])
            ))

        self.payload_vis = self.scene.add_entity(
            gs.morphs.Box(size=(0.25, 0.25, 0.25), pos=(-1.0, 0.0, 1.5)),
            surface=gs.surfaces.Default(color=(0.0, 1.0, 0.0))
        )

        self.viewer_initialized = True

    # =====================================================================
    # RESET
    # =====================================================================
    def reset(self, target_radius=0.2, physics_level=None, env_level=None, curriculum_level=None):
        """Reset environment state."""
        self.time = 0.0
        self.control_time = 0.0  # FIX: Time since reset() returned, excluding settle phase
        self.current_step = 0
        self.elapsed_time = 0.0
        self.docking_bonus_claimed = False

        # Update curriculum levels
        if physics_level is not None:
            self.physics_level = physics_level
        if env_level is not None:
            self.env_level = env_level
        if curriculum_level is not None:
            self.curriculum_level = curriculum_level

        # Reset environment flags
        self.wind_enabled = False
        self.wind_strength = 0.0
        self.obstacles_enabled = False
        self.failure_enabled = False
        self.packet_loss = 0.0
        self._configure_stage()

        # Apply env curriculum
        self._apply_env_curriculum()

        # Physics curriculum
        self._apply_physics_curriculum()

        # Wind direction randomization
        if self.wind_enabled:
            random_dir = np.random.randn(3)
            random_dir /= (np.linalg.norm(random_dir) + 1e-8)
            self.wind_vector = random_dir * np.random.uniform(0.1, self.wind_strength)
        else:
            self.wind_vector = np.zeros(3, dtype=np.float32)

        # Generate obstacles
        self._generate_obstacles()

        # Randomize payload orientation & initial velocity
        rand_yaw = np.random.uniform(-np.pi, np.pi)
        cp_p, sp_p = np.cos(rand_yaw * 0.5), np.sin(rand_yaw * 0.5)
        self.payload_quat = np.array([cp_p, 0.0, 0.0, sp_p], dtype=np.float32)
        self.payload_vel = np.random.uniform(-0.1, 0.1, size=3).astype(np.float32)

        # Compute per-UAV vertical offsets so that each cable starts exactly taut.
        # The horizontal displacement is |formation_offsets[i]| * formation_scale,
        # which is 0.5m for N=2/N=4 and ~0.577m for N=3 (with formation_scale=1.0).
        self.vertical_offsets = []
        for i in range(self.num_uavs):
            horiz = np.linalg.norm(self.formation_offsets[i] * self.formation_scale)
            vert = np.sqrt(max(0.0, self.cable_length**2 - horiz**2))
            self.vertical_offsets.append(vert)

        # Positioning
        if self.start_at_target:
            self._reset_at_target()
        else:
            self._reset_dynamic(target_radius)

        # Reset state variables
        self.tension = 9.81
        self.theta_x = 0.0
        self.theta_y = 0.0
        self.prev_theta_x = 0.0
        self.prev_theta_y = 0.0
        self.swing_x = 0.0
        self.swing_y = 0.0
        self.prev_action = np.zeros(self.num_uavs * 3, dtype=np.float32)
        self.prev_tracking_error = np.linalg.norm(self.payload_pos - self.target)
        self.prev_target = np.copy(self.target)
        self.initial_training_target = np.copy(self.target)
        self.target_vel = np.zeros(3, dtype=np.float32)
        self.target_acc = np.zeros(3, dtype=np.float32)

        # Zero all velocities
        for i in range(self.num_uavs):
            setattr(self, f"uav{i}_vel", np.zeros(3, dtype=np.float32))
        self.payload_vel = np.zeros(3, dtype=np.float32)

        # ── FIX: Physics settle steps with Counter-Gravity Lift ──────────────
        nominal_payload_mass = 1.0
        gravity_acc = 9.81
        control_sensitivity = getattr(self, "control_sensitivity", 6.0)

        tension_share = nominal_payload_mass * gravity_acc / max(1, self.num_uavs)
        hover_z_action = np.clip(tension_share / control_sensitivity, -1.0, 1.0)

        settle_action = np.zeros(self.num_uavs * 3, dtype=np.float32)
        for i in range(self.num_uavs):
            settle_action[i * 3 + 2] = hover_z_action

        init_xy = [np.copy(getattr(self, f"uav{i}_pos")[:2]) for i in range(self.num_uavs)]

        for _ in range(200):
            self._simulate_uavs(settle_action, np.zeros(3, dtype=np.float32))

            for i in range(self.num_uavs):
                # Target the actual memory reference of the numpy array
                getattr(self, f"uav{i}_pos")[:2] = init_xy[i]

            self._update_swing()
            self._update_quaternions()


        # FIX: settle steps advanced self.time by 200*dt (~4s) before the real
        # episode starts. get_trajectory_state()/obs's time feature key off
        # self.time (not control_time), so without this reset every moving-
        # target stage started ~4s into its trajectory — a discontinuous jump
        # right at episode start, every episode. Zero it so self.time and
        # control_time begin in lockstep.
        self.time = 0.0
        self.control_time = 0.0
        self.current_step = 0

        # Re-initialize the FCDEKF (if attached) on the post-settle ground
        # truth, then take one estimate so get_obs() below sees a filtered
        # value from step 0 rather than a stale/None cache.
        self._reset_state_estimator()
        self._refresh_state_estimate()

        # Visual pose re-anchor
        if self.visualize and getattr(self, "scene", None) is not None:
            self.payload_vis.set_pos(self.payload_pos)
            for i in range(self.num_uavs):
                u_pos = getattr(self, f"uav{i}_pos")
                getattr(self, f"uav{i}_vis").set_pos(u_pos)
            self.scene.step()

        return self.get_obs()


    def _apply_physics_curriculum(self):
        """Apply mass and cable variations based on alpha or discrete physics_level."""
        alpha = getattr(self, "curriculum_alpha", 0.0)

        if alpha > 0.0:
            # Continuous parameter scaling
            mass_variation = 0.1 + 0.6 * alpha
            cable_variation = 0.05 + 0.5 * alpha

            raw_mass = np.random.uniform(1.0 - mass_variation, 1.0 + mass_variation)
            raw_cable = np.random.uniform(1.0 - cable_variation, 1.0 + cable_variation)
        else:
            # Fallback to discrete levels
            if self.physics_level == 0:
                raw_mass, raw_cable = np.random.uniform(0.9, 1.1), np.random.uniform(1.0, 1.1)
            elif self.physics_level == 1:
                raw_mass, raw_cable = np.random.uniform(0.8, 1.2), np.random.uniform(1.0, 1.2)
            elif self.physics_level == 2:
                raw_mass, raw_cable = np.random.uniform(0.7, 1.4), np.random.uniform(1.0, 1.4)
            elif self.physics_level == 3:
                raw_mass, raw_cable = np.random.uniform(0.6, 1.7), np.random.uniform(1.0, 1.7)
            else:
                raw_mass = 1.2 + 0.8 * np.random.randn()
                raw_cable = 1.3 + 0.6 * np.random.randn()

        self.payload_mass = float(np.clip(raw_mass, 0.15, 3.5))
        self.cable_length = float(np.clip(raw_cable, 0.70, 3.0))

    def _reset_at_target(self):
        """Reset with payload exactly at target."""
        self.payload_pos = np.copy(self.target)
        self.payload_vel = np.zeros(3, dtype=np.float32)

        for i in range(self.num_uavs):
            offset = self.formation_offsets[i] * self.formation_scale
            pos = (self.payload_pos + offset +
                   np.array([0, 0, self.vertical_offsets[i]], dtype=np.float32))
            setattr(self, f"uav{i}_pos", pos)
            setattr(self, f"uav{i}_vel", np.zeros(3, dtype=np.float32))

    def _reset_dynamic(self, target_radius):
        """Dynamic curriculum reset with safe target & initial position height."""
        # Ensure starting height is around 1.8m to prevent early ground collision during cable tensioning
        self.payload_pos = np.array([
            -1.0 + 0.1 * np.random.randn(),
            0.1 * np.random.randn(),
            1.8 + 0.05 * np.random.randn()
        ], dtype=np.float32)

        self.start_payload_pos = np.copy(self.payload_pos)
        self.payload_vel = np.zeros(3, dtype=np.float32)

        # Target placement
        angle = np.random.uniform(0, 2 * np.pi)
        target_x = self.payload_pos[0] + (target_radius * np.cos(angle))
        target_y = self.payload_pos[1] + (target_radius * np.sin(angle))

        target_z = self.payload_pos[2]
        self.target = np.array([target_x, target_y, target_z], dtype=np.float32)

        # UAV positions with formation geometry
        for i in range(self.num_uavs):
            offset = self.formation_offsets[i] * self.formation_scale
            pos = (self.payload_pos + offset +
                   np.array([0, 0, self.vertical_offsets[i]], dtype=np.float32))
            setattr(self, f"uav{i}_pos", pos)
            setattr(self, f"uav{i}_vel", np.zeros(3, dtype=np.float32))

    # =====================================================================
    # TRAJECTORY GENERATOR (from all 3 files, unified)
    # =====================================================================
    def get_trajectory_state(self, t_val, stage_name=None):
        """Analytical trajectory generator. Returns (pos, vel, acc)."""
        if stage_name is None:
            stage_name = getattr(self, "training_stage", self.stage)

        omega = 2.0 * np.pi / 24.0 * self.speed_multiplier
        scale = self.scale_multiplier

        pos = np.zeros(3, dtype=np.float32)
        vel = np.zeros(3, dtype=np.float32)
        acc = np.zeros(3, dtype=np.float32)

        # Static stages
        if stage_name in ("free", "hover", "wind", "physics_stress", "nmpc_active_shield"):
            pass

        elif stage_name == "line":
            max_disp = self.curriculum_level
            speed = 0.05 * self.speed_multiplier
            pos[0] = np.clip(t_val * speed, -max_disp, max_disp)
            vel[0] = speed

        elif stage_name == "slow_line":
            pos[0] = t_val * 0.1 * scale
            vel[0] = 0.1 * scale

        elif stage_name == "fast_line":
            pos[0] = t_val * 0.3 * scale
            vel[0] = 0.3 * scale

        elif stage_name == "circle":
            pos[0] = 3.0 * scale * np.cos(omega * t_val)
            pos[1] = 3.0 * scale * np.sin(omega * t_val) - 3.0 * scale
            vel[0] = -3.0 * scale * omega * np.sin(omega * t_val)
            vel[1] = 3.0 * scale * omega * np.cos(omega * t_val)
            acc[0] = -3.0 * scale * (omega ** 2) * np.cos(omega * t_val)
            acc[1] = -3.0 * scale * (omega ** 2) * np.sin(omega * t_val)

        elif stage_name == "small_circle":
            # Gentle first taste of curvature: small radius, unhurried rate.
            sc_omega = omega * 0.6
            pos[0] = 2.0 * scale * np.cos(sc_omega * t_val)
            pos[1] = 2.0 * scale * np.sin(sc_omega * t_val) - 2.0 * scale
            vel[0] = -2.0 * scale * sc_omega * np.sin(sc_omega * t_val)
            vel[1] = 2.0 * scale * sc_omega * np.cos(sc_omega * t_val)
            acc[0] = -2.0 * scale * (sc_omega ** 2) * np.cos(sc_omega * t_val)
            acc[1] = -2.0 * scale * (sc_omega ** 2) * np.sin(sc_omega * t_val)

        elif stage_name == "slow_circle":
            # NOTE: radius (5.0) is larger than "circle" (3.0). At the same
            # omega that made this stage's tangential speed *faster* than
            # "circle" despite the name — halve the rate so it's genuinely slow.
            sl_omega = omega * 0.5
            pos[0] = 5.0 * scale * np.cos(sl_omega * t_val)
            pos[1] = 5.0 * scale * np.sin(sl_omega * t_val) - 5.0 * scale
            vel[0] = -5.0 * scale * sl_omega * np.sin(sl_omega * t_val)
            vel[1] = 5.0 * scale * sl_omega * np.cos(sl_omega * t_val)
            acc[0] = -5.0 * scale * (sl_omega ** 2) * np.cos(sl_omega * t_val)
            acc[1] = -5.0 * scale * (sl_omega ** 2) * np.sin(sl_omega * t_val)

        elif stage_name == "half-circle":
            w_half = omega * 0.5
            duration = 10.0
            radius = 2.0 * scale
            if t_val < duration:
                phase = np.pi * (t_val / duration)
                pos[0] = radius * np.sin(phase)
                pos[1] = radius * (1.0 - np.cos(phase))
                vel[0] = radius * (np.pi / duration) * np.cos(phase)
                vel[1] = radius * (np.pi / duration) * np.sin(phase)
            else:
                pos[1] = radius * 2.0

        elif stage_name == "ellipse":
            pos[0] = 4.0 * scale * np.cos(omega * t_val)
            pos[1] = 2.0 * scale * np.sin(omega * t_val) - 2.0 * scale
            vel[0] = -4.0 * scale * omega * np.sin(omega * t_val)
            vel[1] = 2.0 * scale * omega * np.cos(omega * t_val)
            acc[0] = -4.0 * scale * (omega ** 2) * np.cos(omega * t_val)
            acc[1] = -2.0 * scale * (omega ** 2) * np.sin(omega * t_val)

        elif stage_name == "figure-8":
            warmup_time = 4.0
            s = min(1.0, t_val / warmup_time)
            amp_x, amp_y = 3.5 * scale * s, 1.8 * scale * s
            pos[0] = amp_x * np.sin(omega * t_val)
            pos[1] = amp_y * np.sin(2.0 * omega * t_val)
            vel[0] = amp_x * omega * np.cos(omega * t_val)
            vel[1] = amp_y * 2.0 * omega * np.cos(2.0 * omega * t_val)
            acc[0] = -amp_x * (omega ** 2) * np.sin(omega * t_val)
            acc[1] = -amp_y * ((2.0 * omega) ** 2) * np.sin(2.0 * omega * t_val)

        elif stage_name == "lemniscate":
            denom = 1.0 + np.sin(omega * t_val) ** 2
            pos[0] = (3.5 * scale * np.cos(omega * t_val)) / denom
            pos[1] = (3.5 * scale * np.sin(omega * t_val) * np.cos(omega * t_val)) / denom
            pos[2] = 0.2 * scale * np.sin(2.0 * omega * t_val)
            # Numerical derivatives
            dt_diff = 0.01
            p_prev = self._lemniscate_pos(t_val - dt_diff, omega, scale)
            p_next = self._lemniscate_pos(t_val + dt_diff, omega, scale)
            vel = (p_next - p_prev) / (2.0 * dt_diff)
            acc = (p_next - 2.0 * pos + p_prev) / (dt_diff ** 2)

        elif stage_name == "corridor":
            corridor_width = 2.0 * scale
            wall_freq = 0.8
            pos[0] = (t_val * 0.15 * self.speed_multiplier) * scale
            pos[1] = (corridor_width * np.sin(wall_freq * t_val * self.speed_multiplier) * 0.5) * scale
            pos[2] = 0.3 * scale * np.sin(1.5 * wall_freq * t_val * self.speed_multiplier)

        elif stage_name == "gates":
            gate_spacing = 4.0 * scale
            gate_idx = int((t_val * self.speed_multiplier) / 3.0)
            gate_phase = ((t_val * self.speed_multiplier) % 3.0) / 3.0
            base_x = gate_idx * gate_spacing + gate_phase * gate_spacing
            gate_y = 1.5 * scale * np.sin(gate_idx * 1.2)
            gate_z = 2.0 + 0.8 * scale * np.cos(gate_idx * 0.9)
            prev_gate_y = 1.5 * scale * np.sin((gate_idx - 1) * 1.2) if gate_idx > 0 else 0.0
            prev_gate_z = 2.0 + 0.8 * scale * np.cos((gate_idx - 1) * 0.9) if gate_idx > 0 else 2.0
            s = gate_phase
            smooth_s = s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)
            pos[0] = base_x
            pos[1] = (prev_gate_y + smooth_s * (gate_y - prev_gate_y)) * scale
            pos[2] = prev_gate_z + smooth_s * (gate_z - prev_gate_z)

        elif stage_name == "square":
            if t_val < 5.0:
                pos[0], pos[1] = (t_val / 5.0) * 2.0 * scale, 0.0
            elif t_val < 10.0:
                pos[0], pos[1] = 2.0 * scale, ((t_val - 5.0) / 5.0) * 2.0 * scale
            elif t_val < 15.0:
                pos[0], pos[1] = 2.0 * scale - ((t_val - 10.0) / 5.0) * 2.0 * scale, 2.0 * scale
            else:
                pos[0], pos[1] = 0.0, 2.0 * scale - ((t_val - 15.0) / 5.0) * 2.0 * scale

        else:  # spline
            pos[0] = 2.5 * scale * np.sin(omega * t_val) + 0.5 * scale * np.cos(3.0 * omega * t_val)
            pos[1] = 1.5 * scale * np.cos(omega * t_val) * np.sin(2.0 * omega * t_val)
            pos[2] = 0.1 * scale * np.sin(4.0 * omega * t_val)

        anchor = getattr(self, "start_payload_pos", np.array([-1.0, 0.0, 1.5], dtype=np.float32))
        return (anchor + pos).astype(np.float32), vel.astype(np.float32), acc.astype(np.float32)

    def _lemniscate_pos(self, t, omega, scale):
        """Helper for lemniscate numerical derivatives."""
        denom = 1.0 + np.sin(omega * t) ** 2
        return np.array([
            (3.5 * scale * np.cos(omega * t)) / denom,
            (3.5 * scale * np.sin(omega * t) * np.cos(omega * t)) / denom,
            0.2 * scale * np.sin(2.0 * omega * t)
        ], dtype=np.float32)

    def get_future_target(self, dt_lookahead):
        """Get future target position for co-planning."""
        target_pos, _, _ = self.get_trajectory_state(self.time + dt_lookahead)
        return target_pos

    def get_target_kinematics(self):
        """Get current target velocity and acceleration."""
        _, target_vel, target_acc = self.get_trajectory_state(self.time)
        return target_vel, target_acc

    def set_curriculum_alpha(self, alpha):
        """
        Sets continuous curriculum progress factor alpha in [0.0, 1.0].
        Smoothly scales speed, trajectory amplitude, and physics/env disturbances.
        """
        self.curriculum_alpha = float(np.clip(alpha, 0.0, 1.0))
        self.speed_multiplier = 0.3 + 0.7 * self.curriculum_alpha
        self.scale_multiplier = 0.5 + 0.5 * self.curriculum_alpha

        # Update environment and physics continuous parameters immediately
        self._apply_env_curriculum()
        self._apply_physics_curriculum()



    def _apply_env_curriculum(self):
        """Apply environmental disturbances based on alpha or discrete env_level."""
        alpha = getattr(self, "curriculum_alpha", 0.0)

        if alpha > 0.0:
            # Continuous scaling takes precedence
            if alpha > 0.2:
                self.wind_enabled = True
                self.wind_strength = 0.1 + 0.5 * ((alpha - 0.2) / 0.8)
            else:
                self.wind_enabled = False
                self.wind_strength = 0.0

            self.obstacles_enabled = (alpha > 0.6)

            if alpha > 0.85:
                self.packet_loss = 0.1 + 0.2 * ((alpha - 0.85) / 0.15)
            else:
                self.packet_loss = 0.0
        else:
            # Fallback to discrete levels
            if self.env_level == 1:
                self.wind_enabled = True
                self.wind_strength = 0.4
            elif self.env_level == 2:
                self.wind_enabled = True
                self.wind_strength = 0.3
                self.obstacles_enabled = True
            elif self.env_level == 3:
                self.wind_enabled = True
                self.wind_strength = 0.5
                self.obstacles_enabled = True
            elif self.env_level == 4:
                self.packet_loss = 0.3
            elif self.env_level == 5:
                self.failure_enabled = True

    # =====================================================================
    # OBSERVATION (supports both centralized and decentralized modes)
    # =====================================================================
    def get_obs(self):
        """Generate unified observation vector."""
        # Apply packet loss if active
        packet_loss_mask = self._apply_packet_loss()

        # Build global state vector
        obs_parts = []

        # [0:6N] All UAV positions and velocities
        for i in range(self.num_uavs):
            obs_parts.extend([
                getattr(self, f"uav{i}_pos") / 10.0,
                getattr(self, f"uav{i}_vel") / 5.0,
            ])

        # Shared metrics (payload, target, swing, etc.)
        t_curr = self.time
        target_now = self.target
        target_fut1 = self.get_future_target(0.3)
        target_vel, target_acc = self.get_target_kinematics()

        # Perceived payload state: FCDEKF estimate when active, else ground
        # truth (see _get_perceived_payload_state()). This is the only place
        # estimation noise/lag enters the observation — physics and reward
        # elsewhere in this file still use the true self.payload_pos etc.
        perc_pos, perc_vel, perc_theta_x, perc_theta_y, perc_tension = self._get_perceived_payload_state()

        obs_parts.extend([
            perc_pos / 10.0,
            perc_vel / 5.0,
            (target_now - perc_pos) / 5.0,
            target_vel / 5.0,
            np.array([perc_theta_x / np.pi, perc_theta_y / np.pi], dtype=np.float32),
            np.array([perc_tension / 20.0], dtype=np.float32),
            np.array([self.wind_strength], dtype=np.float32),
            np.array([self.time / 10.0], dtype=np.float32),
        ])

        obs = np.concatenate(obs_parts).astype(np.float32)

        # Fast-fail sanity check: obs must always be 6*N + 17 long (matches
        # GLOBAL_CRITIC_STATE_DIM in unified_config.py). Catches N-scaling
        # regressions immediately instead of letting them silently corrupt
        # training or crash deep inside packet-loss masking.
        expected_len = 6 * self.num_uavs + 17
        assert obs.shape[0] == expected_len, (
            f"get_obs() produced {obs.shape[0]} dims for N={self.num_uavs}, "
            f"expected {expected_len} (=6*N+17)."
        )

        if packet_loss_mask is not None:
            obs = obs * packet_loss_mask

        return obs

    def get_local_obs(self, uav_idx):
        """
        DEPRECATED — DO NOT USE FOR ROS2 DEPLOYMENT.
        This reference helper used to build a 21-dim base observation that
        does NOT match the runtime builder (unified_policy.build_local_obs_batch).
        The policy was trained on 15 base dims with hard-coded formation offsets.
        For deployment, call build_local_obs_batch() directly.
        """
        from unified_policy import build_local_obs_batch
        import torch
        global_obs = self.get_obs()
        obs_t = torch.from_numpy(global_obs).float().unsqueeze(0)
        local = build_local_obs_batch(obs_t, uav_idx, self.num_uavs, device='cpu')
        return local.squeeze(0).numpy()

    def get_all_local_obs(self):
        """Get local observations for all UAVs."""
        return [self.get_local_obs(i) for i in range(self.num_uavs)]

    def _apply_packet_loss(self):
        """Generate packet loss mask."""
        if self.packet_loss <= 0.0:
            return None

        # Global obs dimension is 6*N + 17
        obs_len = 6 * self.num_uavs + 17
        mask = np.random.random(obs_len) > self.packet_loss

        # Always protect critical channels
        mask[0: 6 * self.num_uavs] = True  # UAV state channels

        err_start = 6 * self.num_uavs + 6
        mask[err_start: err_start + 3] = True  # Target tracking error channels

        return mask.astype(np.float32)

    def _process_action(self, action):
        """Validates, pads if necessary, and clips RL actions to [-1.0, 1.0]."""
        action = np.array(action, dtype=np.float32).flatten()
        expected_dim = self.num_uavs * 3
        if len(action) < expected_dim:
            action = np.pad(action, (0, expected_dim - len(action)))
        return np.clip(action, -1.0, 1.0)
    # =====================================================================
    # STEP
    # =====================================================================
    def step(self, action):
        """Execute one simulation step."""
        if self.current_step == 0:
            self.time = 0.0
        self.time += self.dt
        self.control_time += self.dt  # FIX: Track time since control started
        self.current_step += 1

        # Update trajectory target
        self._update_trajectory_target()

        # Process action
        action = self._process_action(action)

        # Generate wind
        wind = self._generate_wind()

        # Apply failure if enabled
        if self.failure_enabled and self.time > 5.0 and len(action) >= 6:
            action[3:6] *= 0.2  # Degrade UAV1

        # Physics simulation (independent cable tensions from 4-UAV)
        self._simulate_uavs(action, wind)
        # self._simulate_payload()  # Now handled inside _simulate_uavs()
        self._update_swing()

        # Run the FCDEKF (if active) forward one step on the true post-physics
        # state, so get_obs()/get_local_obs() below perceive the estimate
        # rather than ground truth. Reward/termination below intentionally
        # keep using self.payload_pos etc. directly — the agent's *actions*
        # should be judged against reality even if its *observations* aren't.
        self._refresh_state_estimate()

        # Compute reward
        reward, info = self._compute_reward(action)

        # Check termination
        # FIX: _check_termination applies terminal bonuses/penalties (success
        # +150, out-of-bounds/collision -100) on top of the dense reward.
        # It must return the adjusted value, not just `done`, or the agent
        # never actually sees the sparse terminal signal.
        done, reward = self._check_termination(reward)

        # Update visuals
        self._update_visuals()
        self._update_quaternions()

        return self.get_obs(), reward, done, info

    # ==============================================================================
    # UNIFIED_WORLD.PY FIX: Smooth target vertical ramping / alignment
    # ==============================================================================

    def _update_trajectory_target(self):
        """Update moving target from trajectory generator."""
        active_stage = getattr(self, "training_stage", self.stage)

        if not self.is_training and active_stage == "hover":
            # Keep hover target height reasonable relative to initial payload height (1.5m)
            self.target = np.array([1.0, 1.0, 1.5], dtype=np.float32)
        elif not self.is_training:
            self.target, self.target_vel, self.target_acc = self.get_trajectory_state(self.time)
        else:
            if active_stage in ("free", "hover", "wind", "physics_stress", "nmpc_active_shield"):
                self.target = self.initial_training_target
            else:
                traj_pos, traj_vel, traj_acc = self.get_trajectory_state(self.time)
                traj_pos_zero, _, _ = self.get_trajectory_state(0.0)
                self.target = self.initial_training_target + (traj_pos - traj_pos_zero)

        # Update target velocity from finite differences
        if hasattr(self, "prev_target"):
            self.target_vel = (self.target - self.prev_target) / self.dt
        self.prev_target = np.copy(self.target)

    def _process_action(self, action):
        """Process and validate action."""
        action = np.array(action).flatten()
        expected = self.num_uavs * 3
        if len(action) < expected:
            action = np.pad(action, (0, expected - len(action)))
        action = np.clip(action, -1.0, 1.0)
        return action

    def _generate_wind(self):
        """Generate wind vector."""
        if self.wind_enabled:
            return self.wind_vector + self.wind_strength * 0.1 * np.random.randn(3)
        return np.zeros(3, dtype=np.float32)

    def _simulate_uavs(self, action, wind):
        """Simulate UAV dynamics with properly coupled cable tensions.

        FIX: Cable forces are now computed BEFORE the UAV state update and
        included in the force balance. Previously, tension was computed
        after UAV motion and applied as a post-hoc velocity kick, violating
        energy conservation and causing monotonic payload falling.
        """
        uav_base_mass = 0.5
        gravity_acc = 9.81
        gravity_vec = np.array([0.0, 0.0, -gravity_acc], dtype=np.float32)
        control_sensitivity = self.control_sensitivity

        # ── PHASE 1: Compute cable tensions from CURRENT state ─────────────────
        tether_forces_on_payload = []   # Force ON payload FROM each UAV
        tether_reactions_on_uavs = []   # Force ON each UAV FROM cable

        for i in range(self.num_uavs):
            uav_pos = getattr(self, f"uav{i}_pos")
            uav_vel = getattr(self, f"uav{i}_vel")

            disp = uav_pos - self.payload_pos
            dist = np.linalg.norm(disp)
            direction = disp / (dist + 1e-8)

            strain = max(0.0, dist - self.cable_length)
            rel_vel = uav_vel - self.payload_vel
            rel_axial = np.dot(rel_vel, direction)

            tension = (self.cable_k * strain) + (self.cable_d * rel_axial)
            tension = max(0.0, tension)

            force_on_payload = tension * direction      # pulls payload toward UAV
            tether_forces_on_payload.append(force_on_payload)
            tether_reactions_on_uavs.append(-force_on_payload)  # pulls UAV toward payload

        # ── PHASE 2: Update ALL UAVs with complete force balance ─────────────
        for i in range(self.num_uavs):
            control_force = action[i * 3:(i + 1) * 3] * control_sensitivity

            # Static hover thrust: cancels ONLY the UAV's own weight.
            # The payload is supported by cable tension, not by direct thrust
            # allocation. Zero control action means "hold the UAV's own
            # position" — cable tension carries the payload dynamically via
            # stretch, and by Newton's third law that same tension pulls this
            # UAV down by an equal amount whenever the cable is loaded.
            #
            # NOTE: we tried adding a static payload_mass/num_uavs share here
            # so zero action would be a true system-level hover point. That's
            # unstable: whenever the cable goes slack (tension=0 — which
            # happens often, e.g. whenever the UAV drifts within cable_length
            # of the payload), that share becomes *unbalanced free lift* with
            # nothing pulling it back down, so the pair rockets upward without
            # bound. The real fix for "policy never learns to fight gravity"
            # lives in the actor's initialization instead (see
            # unified_policy.py's hover-bias init) — it nudges exploration
            # toward the right thrust without ever changing the environment's
            # force balance, so it can't introduce this runaway.
            static_thrust = uav_base_mass * gravity_acc
            uav_lift_force = np.array([0.0, 0.0, static_thrust], dtype=np.float32)

            # Complete force balance: control + lift + gravity + cable + wind
            f_net = (control_force +
                     uav_lift_force +
                     (uav_base_mass * gravity_vec) +
                     tether_reactions_on_uavs[i] +
                     wind)

            uav_acc = f_net / uav_base_mass
            uav_vel = getattr(self, f"uav{i}_vel")
            uav_pos = getattr(self, f"uav{i}_pos")

            # Semi-implicit Euler: velocity first, then position
            uav_vel_new = uav_vel + uav_acc * self.dt
            uav_vel_new[0] *= 0.95  # X damping
            uav_vel_new[1] *= 0.95  # Y damping
            uav_vel_new[2] *= 0.98  # Z damping (relaxed from 0.65)
            uav_vel_new = np.clip(uav_vel_new, -2.0, 2.0)

            uav_pos_new = uav_pos + uav_vel_new * self.dt
            uav_pos_new = np.clip(uav_pos_new, -10.0, 10.0)
            uav_pos_new[2] = max(uav_pos_new[2], self.payload_pos[2] + 0.2)

            setattr(self, f"uav{i}_vel", uav_vel_new)
            setattr(self, f"uav{i}_pos", uav_pos_new)

        # ── PHASE 3: Update payload with total cable force ───────────────────
        f_total = sum(tether_forces_on_payload)
        self.tension = float(np.linalg.norm(f_total))

        payload_acc = f_total / self.payload_mass + gravity_vec
        # FIX: Loosened clip. cable_k=200 produces legitimate restoring accelerations
        # of 100-200 m/s². Old ±12 clip reduced them by 90%+, causing free-fall to floor.
        payload_acc = np.clip(payload_acc, -200.0, 200.0)

        self.payload_acc = payload_acc
        self.payload_vel += payload_acc * self.dt

        # FIX: Velocity damping prevents undamped oscillations when cable goes slack.
        self.payload_vel[0] *= 0.95
        self.payload_vel[1] *= 0.95
        self.payload_vel[2] *= 0.98

        self.payload_vel = np.clip(self.payload_vel, -15.0, 15.0)
        self.payload_pos += self.payload_vel * self.dt

        # SAFETY FLOOR: numerical guard against runaway freefall (e.g. transient
        # spikes during the very first settle steps), NOT a resting place.
        # FIX: this used to clamp at z=0.35, which sits ABOVE the crash
        # termination threshold in _check_termination (payload_pos[2] < 0.3).
        # That meant a payload that was genuinely falling got caught here
        # every single step, parked at 0.35 forever, and NEVER tripped the
        # out-of-bounds crash check — so a full freefall was invisible to
        # training: no terminal penalty, no episode reset, just ~20s of easy
        # "standing still" reward per episode. Clamping below the termination
        # threshold (0.15 < 0.3) means a real crash now reaches the crash
        # check and actually terminates with the -1.0 penalty, while this
        # still stops truly pathological negative-z blowups.
        if self.payload_pos[2] < 0.15:
            self.payload_pos[2] = 0.15
            self.payload_vel[2] = max(0.0, self.payload_vel[2])

    def _simulate_payload(self):
        """Simulate payload dynamics.

        NOTE: Payload is now updated inside _simulate_uavs() to ensure
        consistent cable forces. This method is kept for backward
        compatibility but does nothing — the payload state is already current.
        """
        pass

    def _update_swing(self):
        """Update swing angles and derivatives."""
        # Compute geometric center
        center = np.zeros(3, dtype=np.float32)
        for i in range(self.num_uavs):
            center += getattr(self, f"uav{i}_pos")
        center /= self.num_uavs

        dx = self.payload_pos[0] - center[0]
        dy = self.payload_pos[1] - center[1]
        dz = self.payload_pos[2] - center[2]

        self.theta_x = np.arctan2(dx, -dz)
        self.theta_y = np.arctan2(dy, -dz)
        self.swing_x = dx
        self.swing_y = dy

        # Derivative swing velocities
        theta_x_dot = (self.theta_x - self.prev_theta_x) / self.dt
        theta_y_dot = (self.theta_y - self.prev_theta_y) / self.dt
        self.prev_theta_x = self.theta_x
        self.prev_theta_y = self.theta_y

        # FIX: Compute swing metric here so it works even when visualize=False.
        # Horizontal swing magnitude = sqrt(swing_x² + swing_y²)
        self.last_swing_metric = float(np.sqrt(self.swing_x ** 2 + self.swing_y ** 2))

        return theta_x_dot, theta_y_dot

    def _compute_reward(self, action):
        """Unified, highly optimized pure-tensor reward function utilizing full GPU acceleration."""
        # 1. Dynamically identify and anchor to the active GPU/CPU device
        device = getattr(self, 'device', 'cpu')

        # 2. Convert all environmental states into unified device tensors at once
        payload_pos_t = torch.as_tensor(self.payload_pos, dtype=torch.float32, device=device)
        target_t = torch.as_tensor(self.target, dtype=torch.float32, device=device)
        payload_vel_t = torch.as_tensor(self.payload_vel, dtype=torch.float32, device=device)
        payload_acc_t = torch.as_tensor(self.payload_acc, dtype=torch.float32, device=device)
        action_t = torch.as_tensor(action, dtype=torch.float32, device=device)
        prev_action_t = torch.as_tensor(self.prev_action, dtype=torch.float32, device=device)

        current_stage = getattr(self, "training_stage", self.stage)

        # Tracking error
        tracking_error_vec = target_t - payload_pos_t
        tracking_error = torch.norm(tracking_error_vec, dim=-1)
        err_val = float(tracking_error.item()) if hasattr(tracking_error, 'item') else float(tracking_error)

        # Velocity tracking
        target_vel_np, target_acc_np = self.get_target_kinematics()
        target_vel_t = torch.as_tensor(target_vel_np, dtype=torch.float32, device=device)
        target_acc_t = torch.as_tensor(target_acc_np, dtype=torch.float32, device=device)

        velocity_error_vec = target_vel_t - payload_vel_t
        velocity_error = torch.norm(velocity_error_vec, dim=-1)
        actual_speed = torch.norm(payload_vel_t, dim=-1)

        # Future target for cross-track
        target_fut1_np = self.get_future_target(0.25)
        target_fut1_t = torch.as_tensor(target_fut1_np, dtype=torch.float32, device=device)

        # Cross-track error
        _STATIC_STAGES = ("free", "hover", "wind", "physics_stress", "nmpc_active_shield")
        if current_stage in _STATIC_STAGES:
            start_pos_t = torch.as_tensor(self.start_payload_pos, dtype=torch.float32, device=device)
            path_tangent = target_t - start_pos_t
        else:
            path_tangent = target_fut1_t - target_t

        path_tangent_norm = torch.norm(path_tangent, dim=-1) + 1e-6
        path_direction = path_tangent / path_tangent_norm

        if current_stage in _STATIC_STAGES:
            start_pos_t = torch.as_tensor(self.start_payload_pos, dtype=torch.float32, device=device)
            payload_relative = payload_pos_t - start_pos_t
        else:
            payload_relative = payload_pos_t - target_t

        along_track_proj = torch.sum(payload_relative * path_direction, dim=-1)
        cross_track_vector = payload_relative - (along_track_proj * path_direction)
        cross_track_error = torch.norm(cross_track_vector, dim=-1)

        # Heading alignment
        payload_speed = torch.norm(payload_vel_t, dim=-1) + 1e-6
        payload_vel_dir = payload_vel_t / payload_speed
        heading_alignment = torch.sum(payload_vel_dir * path_direction, dim=-1)

        # Acceleration tracking
        acceleration_tracking_error = torch.norm(target_acc_t - payload_acc_t, dim=-1)

        # Swing magnitude
        theta_x_t = torch.as_tensor(self.theta_x, dtype=torch.float32, device=device)
        theta_y_t = torch.as_tensor(self.theta_y, dtype=torch.float32, device=device)
        swing_magnitude = torch.sqrt(theta_x_t ** 2 + theta_y_t ** 2)

        # Control jerk
        control_jerk = torch.norm(action_t - prev_action_t, dim=-1)
        self.prev_action = np.copy(action)

        # Formation metrics (pairwise distances)
        # === FIXED  ===
        inter_uav_error = torch.tensor(0.0, device=device)
        formation_symmetry_reward = torch.tensor(0.0, device=device)
        hard_collision_penalty = torch.tensor(0.0, device=device)
        collision_min_dist = torch.tensor((2.0 * UAV_RADIUS) + 0.02, device=device)

        uav_positions = [getattr(self, f"uav{i}_pos") for i in range(self.num_uavs)]
        uav_tensors = [torch.as_tensor(p, dtype=torch.float32, device=device) for p in uav_positions]

        # Ensure execution mode check defaults safely to centralized
        current_mode = getattr(self, "execution_mode", "centralized")

        if current_mode == "decentralized":
            # Accumulate decentralized credit penalties for ALL agents equally per step
            for idx in range(self.num_uavs):
                for j in range(self.num_uavs):
                    if idx == j:
                        continue
                    pair_key = (min(idx, j), max(idx, j))
                    dist = torch.norm(uav_tensors[idx] - uav_tensors[j], dim=-1)
                    target_edge = torch.tensor(self.formation_pair_targets[pair_key], device=device)

                    lower_bound = 0.7 * target_edge
                    upper_bound = 1.4 * target_edge
                    inter_uav_error += torch.clamp(dist - upper_bound, min=0.0) + torch.clamp(lower_bound - dist,
                                                                                              min=0.0)

                    edge_error = torch.abs(dist - target_edge)
                    formation_symmetry_reward += torch.exp(-3.0 * edge_error)
                    hard_collision_penalty += torch.clamp(collision_min_dist - dist, min=0.0) ** 2 * 50.0

            # Normalize by total active interaction directions
            total_directions = max(1, self.num_uavs * (self.num_uavs - 1))
            formation_symmetry_reward = formation_symmetry_reward / total_directions
            normalized_inter_uav_error = inter_uav_error / total_directions

        else:
            # Centralized mode: Standard baseline pairwise combinations
            for i in range(self.num_uavs):
                for j in range(i + 1, self.num_uavs):
                    dist = torch.norm(uav_tensors[i] - uav_tensors[j], dim=-1)
                    target_edge = torch.tensor(self.formation_pair_targets[(i, j)], device=device)

                    lower_bound = 0.7 * target_edge
                    upper_bound = 1.4 * target_edge
                    inter_uav_error += torch.clamp(dist - upper_bound, min=0.0) + torch.clamp(lower_bound - dist,
                                                                                              min=0.0)

                    edge_error = torch.abs(dist - target_edge)
                    formation_symmetry_reward += torch.exp(-3.0 * edge_error)
                    hard_collision_penalty += torch.clamp(collision_min_dist - dist, min=0.0) ** 2 * 50.0

            num_pairs = max(1, self.num_uavs * (self.num_uavs - 1) // 2)
            formation_symmetry_reward = formation_symmetry_reward / num_pairs
            normalized_inter_uav_error = inter_uav_error / num_pairs
        # ===============================================



        # Altitude uniformity
        formation_altitude_reward = torch.tensor(0.0, device=device)
        mean_vertical_offset = torch.tensor(np.mean(self.vertical_offsets), device=device)
        for i in range(self.num_uavs):
            alt_diff = uav_tensors[i][2] - payload_pos_t[2]
            alt_error = torch.abs(alt_diff - mean_vertical_offset)
            formation_altitude_reward += torch.exp(-3.0 * alt_error)
        formation_altitude_reward = formation_altitude_reward / self.num_uavs

        # Center of mass alignment
        center_t = torch.stack(uav_tensors).mean(dim=0)
        horizontal_center_error = torch.norm(center_t[:2] - payload_pos_t[:2], dim=-1)
        formation_center_reward = torch.exp(-4.0 * horizontal_center_error)

        # Cohesion (distance to payload)
        min_payload_dist = torch.tensor(0.15, device=device)
        cohesion_error = torch.tensor(0.0, device=device)
        proximity_error = torch.tensor(0.0, device=device)
        for i in range(self.num_uavs):
            dist_to_payload = torch.norm(uav_tensors[i] - payload_pos_t, dim=-1)
            cohesion_error += torch.clamp(dist_to_payload - 1.5, min=0.0)
            proximity_error += torch.clamp(min_payload_dist - dist_to_payload, min=0.0) ** 2

        cohesion_penalty_payload = torch.clamp(cohesion_error * 2.0, max=10.0)
        payload_proximity_penalty = proximity_error * 15.0

        # Boundary proximity
        boundary_margin = 3.0
        boundary_limit = 8.0
        horiz_dist = torch.norm(payload_pos_t[:2], dim=-1)
        boundary_penalty = torch.clamp(2.0 * torch.clamp(horiz_dist - (boundary_limit - boundary_margin), min=0.0),
                                       max=20.0)

        # Obstacle collision
        obstacle_penalty = torch.tensor(self._check_obstacle_collision(self.payload_pos), device=device)

        # --- Stage-Specific Weights ---
        # === COMPLETE UNIFIED OVERHAUL ===
        w_tracking = 1.0
        w_formation = 1.0
        w_swing = 1.0
        w_jerk = 1.0
        w_overshoot = 1.0
        w_progress = 8.0

        # === REPLACEMENT FOR STAGE WEIGHTS BLOCKS ===
        # Initial defaults
        w_tracking = 1.0
        w_formation = 1.0
        w_swing = 1.0
        w_jerk = 1.0
        w_overshoot = 1.0
        w_progress = 8.0
        centripetal_scale = 1.0  # FIXED: Default scale factor multiplier

        _TRAJECTORY_STAGES = (
            "slow_line", "line", "fast_line", "circle", "figure_8",
            "spline", "physics_stress", "nmpc_active_shield"
        )

        if current_stage in ("free", "hover"):
            w_tracking = 10.0
            w_formation = 1.0
            w_swing = 0.005
            w_jerk = 0.0
            w_overshoot = 0.0
            w_progress = 15.0

        elif current_stage == "wind":
            w_tracking = 8.0
            w_formation = 0.5
            w_swing = 0.05
            w_jerk = 0.02
            w_overshoot = 0.0
            w_progress = 15.0

        elif current_stage in _TRAJECTORY_STAGES:
            w_tracking = 15.0
            w_formation = 0.5
            w_swing = 0.05
            w_jerk = 0.01
            w_overshoot = 0.1
            w_progress = 25.0
            centripetal_scale = 0.10  # FIXED: Store scaling configuration property safely

        # --- Core Rewards Assembly ---
        # --- Core Rewards Assembly ---
        if err_val > 0.5:
            # Strong linear push when far away to force navigation
            tracking_reward = torch.tensor(3.0 - (2.0 * err_val), device=device)
        else:
            # High-precision exponential lock inside the docking zone.
            # BUGFIX: exp(-1.5*err) is nearly flat once err drops below
            # ~0.15-0.2m (e.g. 4*exp(-1.5*0.2)=2.96 vs 4*exp(-1.5*0.05)=3.71 —
            # a weak gradient for a 15cm improvement), so PPO had little
            # incentive to keep tightening once "close enough". A second,
            # steeper term is added that only matters well inside the
            # ~10-15cm zone, giving a much stronger gradient exactly where
            # the 10cm precision target lives, without changing behavior far
            # from the target.
            tracking_reward = (
                torch.exp(-1.5 * tracking_error) * 4.0
                + torch.exp(-15.0 * tracking_error) * 2.0
            )
        # Ensure 'uav_velocities' matches the exact variable name used in your environment
        # for the tensor containing the velocities of all UAVs [batch_size, num_uavs, 3]
        velocity_sync_reward = torch.exp(-1.0 * velocity_error) * torch.clamp(actual_speed, max=1.0)
        alignment_reward = 1.5 * heading_alignment if heading_alignment > 0 else 3.0 * heading_alignment

        path_quality_penalty = -torch.clamp(2.5 * (cross_track_error ** 2) + 1.0 * cross_track_error, max=50.0)

        # --- FIX: Calculate first, then apply the variable scale factor ---
        centripetal_penalty = -0.5 * acceleration_tracking_error
        centripetal_penalty = centripetal_penalty * centripetal_scale

        stability_penalty = 2.0 * (swing_magnitude ** 2) + 1.0 * swing_magnitude
        jerk_penalty = 1.5 * control_jerk
        velocity_overshoot_penalty = 0.5 * torch.norm(payload_vel_t) * torch.clamp(0.5 - tracking_error, min=0.0)

        total_formation_bonus = 1.0 * (formation_symmetry_reward + formation_altitude_reward + formation_center_reward)
        inter_uav_penalty = -10.0 * normalized_inter_uav_error



        reward = (
                w_tracking * tracking_reward +
                velocity_sync_reward +
                alignment_reward +
                path_quality_penalty +
                centripetal_penalty -
                w_swing * stability_penalty -
                w_jerk * jerk_penalty -
                cohesion_penalty_payload -
                payload_proximity_penalty +
                inter_uav_penalty -
                hard_collision_penalty -
                w_overshoot * velocity_overshoot_penalty +
                w_formation * total_formation_bonus -
                boundary_penalty +
                obstacle_penalty
        )

        # Progress shaper
        progress = self.prev_tracking_error - err_val
        if progress < 0:
            reward += w_progress * (progress * 0.15)
        else:
            reward += w_progress * progress
        self.prev_tracking_error = err_val

        # Docking bonus
        if err_val < DOCKING_BONUS_THRESHOLD and not self.docking_bonus_claimed:
            reward += 50.0
            self.docking_bonus_claimed = True

        # Extract item cleanly for scalar output return
        reward_val = reward.item() if hasattr(reward, 'item') else float(reward)
        scaled_reward = reward_val / 100.0
        # Reward decomposition for RL debugging
        info = {
            "reward_total": scaled_reward,
            "reward_tracking": float((w_tracking * tracking_reward).item()) / 100.0,
            "reward_velocity_sync": float(velocity_sync_reward.item()) / 100.0,
            "reward_alignment": float(alignment_reward.item()) / 100.0,
            "reward_path_quality": float(path_quality_penalty.item()) / 100.0,
            "reward_centripetal": float(centripetal_penalty.item()) / 100.0,
            "reward_swing": float((-w_swing * stability_penalty).item()) / 100.0,
            "reward_jerk": float((-w_jerk * jerk_penalty).item()) / 100.0,
            "reward_cohesion": float((-cohesion_penalty_payload).item()) / 100.0,
            "reward_payload_proximity": float((-payload_proximity_penalty).item()) / 100.0,
            "reward_inter_uav": float(inter_uav_penalty.item()) / 100.0,
            "reward_hard_collision": float((-hard_collision_penalty).item()) / 100.0,
            "reward_formation": float((w_formation * total_formation_bonus).item()) / 100.0,
            "reward_boundary": float((-boundary_penalty).item()) / 100.0,
            "reward_obstacle": float(obstacle_penalty) / 100.0,
            "reward_progress": float(w_progress * progress) / 100.0,
            "reward_docking": 0.5 if (err_val < DOCKING_BONUS_THRESHOLD and not self.docking_bonus_claimed) else 0.0,
            "tracking_error": float(err_val),
            "swing_magnitude": float(swing_magnitude.item()),
            "payload_speed": float(torch.norm(payload_vel_t).item()),
        }

        return scaled_reward, info

    def _check_termination(self, reward_val):
        """Check episode termination conditions."""
        err_val = self.prev_tracking_error

        out_of_bounds = (
            abs(self.payload_pos[0]) > 8.0 or
            abs(self.payload_pos[1]) > 8.0 or
            self.payload_pos[2] > 10.0 or
            self.payload_pos[2] < 0.3
        )

        obstacle_hit = self.obstacles_enabled and self._check_obstacle_collision(self.payload_pos) <= -10.0
        tension_exceeded = getattr(self, "tension", 0.0) > getattr(self, "max_allowable_tension", 50.0)
        # BUGFIX: this used to be hardcoded here (max(0.03, 0.3 - 0.05*physics_level),
        # i.e. a 30cm base) and never read unified_config.BASE_SUCCESS_THRESHOLD /
        # PHYSICS_THRESHOLD_DECAY at all, so "success" was declared (and the
        # episode ended) once the payload was merely within 30cm of the
        # target — much looser than the ~10cm precision actually wanted, and
        # it gave the policy no reason to hold tighter than that. Now driven
        # by config so there's one place to tune it.
        success_threshold = max(
            SUCCESS_THRESHOLD_FLOOR,
            BASE_SUCCESS_THRESHOLD - PHYSICS_THRESHOLD_DECAY * self.physics_level
        )

        # Stages where the target is stationary vs. actively moving along a
        # trajectory (see _update_trajectory_target). The docking bonus below
        # originally required near-zero ABSOLUTE payload velocity, which a
        # payload chasing a moving target can essentially never satisfy —
        # so the bonus silently never fired for circle/figure-8/etc stages.
        _STATIC_TARGET_STAGES = ("free", "hover", "wind", "physics_stress", "nmpc_active_shield")
        active_stage = getattr(self, "training_stage", self.stage)
        target_is_moving = active_stage not in _STATIC_TARGET_STAGES

        if target_is_moving:
            target_vel_now = getattr(self, "target_vel", np.zeros(3, dtype=np.float32))
            vel_error = float(np.linalg.norm(self.payload_vel - target_vel_now))
            # Allow velocity to differ from zero, but require it to actually
            # be matching the target's motion (small tracking-velocity error).
            velocity_ok = vel_error < 0.3
        else:
            velocity_ok = float(np.linalg.norm(self.payload_vel)) < 0.15

        done = False
        if out_of_bounds:
            # BUGFIX: _compute_reward() already scales its output by /100
            # (see scaled_reward = reward_val / 100.0). This terminal penalty
            # must be scaled the same way, or it lands ~100x larger than a
            # normal step's reward and blows up the value-function target
            # right at episode boundaries. -100.0 (unscaled) -> -1.0 (scaled).
            reward_val -= 1.0
            done = True
        elif obstacle_hit:
            reward_val -= 1.0  # BUGFIX: scaled to match dense reward (-100.0 -> -1.0)
            done = True
        elif tension_exceeded:  # 2. ADD THIS ELIF BLOCK
            reward_val -= 1.0
            done = True
        elif (err_val < success_threshold and
              velocity_ok and
              self.control_time > 2.0):  # FIX: Use control_time, not time (which includes settle)
            # FIX: Added velocity check so success can't fire while payload
            # is still moving fast relative to the target (e.g., overshooting).
            reward_val += 1.5  # BUGFIX: scaled to match dense reward (+150.0 -> +1.5)
            done = True
        elif self.control_time > 20.0:  # FIX: Use control_time for episode length
            done = True

        # FIX: return the adjusted reward_val, not just done — otherwise the
        # terminal bonus/penalty computed above is discarded by the caller.
        return done, reward_val

    def _update_visuals(self):
        """Update visual entities."""
        if self.visualize and self.scene is not None:
            for i in range(self.num_uavs):
                getattr(self, f"uav{i}_vis").set_pos(getattr(self, f"uav{i}_pos"))
            self.payload_vis.set_pos(self.payload_pos)
            self.scene.step()

    def _update_quaternions(self):
        """Update quaternion representations."""
        # Payload quaternion from swing
        cp_l, sp_l = np.cos(self.theta_y * 0.5), np.sin(self.theta_y * 0.5)
        cr_l, sr_l = np.cos(self.theta_x * 0.5), np.sin(self.theta_x * 0.5)
        self.payload_quat = np.array([cr_l * cp_l, sr_l * cp_l, cr_l * sp_l, 0.0], dtype=np.float32)

        # UAV quaternions from velocity
        for i in range(self.num_uavs):
            v_arr = getattr(self, f"uav{i}_vel")
            p_val = np.clip(v_arr[0] * 0.15, -0.4, 0.4)
            r_val = np.clip(-v_arr[1] * 0.15, -0.4, 0.4)
            cp, sp = np.cos(p_val * 0.5), np.sin(p_val * 0.5)
            cr, sr = np.cos(r_val * 0.5), np.sin(r_val * 0.5)
            setattr(self, f"uav{i}_quat", np.array([cr * cp, sr * cp, cr * sp, 0.0], dtype=np.float32))

            # Tether direction
            u_pos = getattr(self, f"uav{i}_pos")
            disp = u_pos - self.payload_pos
            setattr(self, f"tether{i}_dir", disp / (np.linalg.norm(disp) + 1e-8))

        # Update visual quaternions
        if self.visualize and self.scene is not None:
            for i in range(self.num_uavs):
                if hasattr(self, f"uav{i}_vis"):
                    getattr(self, f"uav{i}_vis").set_quat(getattr(self, f"uav{i}_quat"))
            if hasattr(self, 'payload_vis'):
                self.payload_vis.set_quat(self.payload_quat)

    # =====================================================================
    # OBSTACLES
    # =====================================================================
    def _generate_obstacles(self):
        """Generate obstacle field."""
        self.obstacles = []
        if not self.obstacles_enabled:
            return

        if self.stage == "corridor":
            num_segments = 20
            for i in range(num_segments):
                t_seg = i * 1.0
                wall_y = 2.0 * np.sin(0.8 * t_seg)
                self.obstacles.append({
                    'pos': np.array([t_seg, wall_y - 1.5, 1.5]),
                    'size': (0.5, 0.5, 3.0), 'type': 'wall'
                })
                self.obstacles.append({
                    'pos': np.array([t_seg, wall_y + 1.5, 1.5]),
                    'size': (0.5, 0.5, 3.0), 'type': 'wall'
                })

        elif self.stage == "gates":
            num_gates = 8
            for i in range(num_gates):
                gate_x = i * 4.0
                gate_y = 1.5 * np.sin(i * 1.2)
                gate_z = 2.0 + 0.8 * np.cos(i * 0.9)
                self.obstacles.append({
                    'pos': np.array([gate_x, gate_y, gate_z]),
                    'size': (0.3, 2.5, 2.5), 'type': 'gate'
                })

    def _check_obstacle_collision(self, pos):
        """Check obstacle collision. Returns penalty."""
        if not self.obstacles_enabled or len(self.obstacles) == 0:
            return 0.0

        min_dist = float('inf')
        for obs in self.obstacles:
            obs_pos = obs['pos']
            obs_size = obs['size']
            dx = abs(pos[0] - obs_pos[0]) - obs_size[0] / 2
            dy = abs(pos[1] - obs_pos[1]) - obs_size[1] / 2
            dz = abs(pos[2] - obs_pos[2]) - obs_size[2] / 2

            if dx < 0 and dy < 0 and dz < 0:
                return -10.0

            dist = np.sqrt(max(0, dx) ** 2 + max(0, dy) ** 2 + max(0, dz) ** 2)
            min_dist = min(min_dist, dist)

        if min_dist < 1.0:
            return -2.0 * (1.0 - min_dist)
        return 0.0
