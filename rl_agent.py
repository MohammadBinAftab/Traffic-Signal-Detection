"""
rl_agent.py
Reinforcement Learning agent that adaptively controls
the YOLO confidence threshold based on frame conditions.

Algorithm: Q-Learning (tabular, no neural network needed)
"""

import numpy as np
import json
import os

class AdaptiveThresholdAgent:
    
    # The 5 threshold choices the agent can pick from
    THRESHOLD_OPTIONS = [0.20, 0.35, 0.50, 0.65, 0.80]
    
    def __init__(self, q_table_path="q_table.json"):
        self.q_table_path = q_table_path
        self.n_actions     = len(self.THRESHOLD_OPTIONS)
        
        # Q-Learning hyperparameters
        self.alpha   = 0.1    # learning rate  — how fast it updates
        self.gamma   = 0.9    # discount factor — how much future rewards matter
        self.epsilon = 0.2    # exploration rate — 20% random, 80% learned
        
        # Load existing Q-table if available, else start fresh
        self.q_table = self._load_q_table()
        
        # Track recent detections to calculate reward
        self.recent_det_counts = []   # number of detections per frame
        self.last_state  = None
        self.last_action = None

    # ── State Extraction ────────────────────────────────────────────
    def get_state(self, frame) -> tuple:
        """
        Analyse the frame and return a discrete state tuple.
        State = (brightness_level, blur_level, recent_det_level)
        Each dimension has 3 levels: low / medium / high → 27 total states
        """
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Brightness: mean pixel value
        brightness = np.mean(gray)
        if brightness < 60:
            b_level = 0   # dark / night / foggy
        elif brightness < 150:
            b_level = 1   # normal
        else:
            b_level = 2   # very bright / overexposed

        # 2. Blur: Laplacian variance (low = blurry, high = sharp)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 50:
            blur_level = 0    # very blurry (moving car, rain)
        elif blur_score < 300:
            blur_level = 1    # moderate
        else:
            blur_level = 2    # sharp and clear

        # 3. Recent detection density
        avg_dets = np.mean(self.recent_det_counts[-5:]) if self.recent_det_counts else 1
        if avg_dets == 0:
            det_level = 0     # agent has been missing signs
        elif avg_dets <= 3:
            det_level = 1     # reasonable number of detections
        else:
            det_level = 2     # too many detections (threshold too low)

        return (b_level, blur_level, det_level)

    # ── Action Selection ────────────────────────────────────────────
    def choose_action(self, state: tuple) -> int:
        """Epsilon-greedy: explore randomly OR exploit best known action."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)   # explore
        
        state_key = str(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.n_actions
        
        return int(np.argmax(self.q_table[state_key]))  # exploit

    def get_threshold(self, frame) -> float:
        """
        Main method called every frame.
        Returns the confidence threshold the agent recommends.
        """
        state  = self.get_state(frame)
        action = self.choose_action(state)
        
        self.last_state  = state
        self.last_action = action
        
        return self.THRESHOLD_OPTIONS[action]

    # ── Reward & Learning ───────────────────────────────────────────
    def update(self, frame, n_detections: int):
        """
        Call this AFTER YOLO runs with the agent's threshold.
        Calculates reward and updates the Q-table.
        
        n_detections: how many signs YOLO found this frame
        """
        if self.last_state is None:
            return

        self.recent_det_counts.append(n_detections)
        if len(self.recent_det_counts) > 20:
            self.recent_det_counts.pop(0)

        # ── Reward Design ──
        threshold = self.THRESHOLD_OPTIONS[self.last_action]
        reward = 0.0

        if n_detections == 0 and threshold > 0.65:
            reward = -0.5   # threshold too high, probably missing real signs

        elif n_detections == 0 and threshold <= 0.35:
            reward = +0.2   # low threshold, still nothing — genuinely empty frame

        elif 1 <= n_detections <= 4:
            reward = +1.0   # ideal: found a reasonable number of signs

        elif n_detections > 6:
            reward = -0.3   # threshold too low, too many false positives

        # Q-Learning update  →  Q(s,a) += α * [r + γ * max Q(s') - Q(s,a)]
        new_state = self.get_state(frame)
        new_state_key  = str(new_state)
        last_state_key = str(self.last_state)

        if new_state_key not in self.q_table:
            self.q_table[new_state_key] = [0.0] * self.n_actions
        if last_state_key not in self.q_table:
            self.q_table[last_state_key] = [0.0] * self.n_actions

        old_q      = self.q_table[last_state_key][self.last_action]
        future_q   = max(self.q_table[new_state_key])
        new_q      = old_q + self.alpha * (reward + self.gamma * future_q - old_q)
        
        self.q_table[last_state_key][self.last_action] = new_q

    # ── Persistence ─────────────────────────────────────────────────
    def save(self):
        """Save the Q-table to disk so learning persists between sessions."""
        with open(self.q_table_path, "w") as f:
            json.dump(self.q_table, f)

    def _load_q_table(self) -> dict:
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path) as f:
                return json.load(f)
        return {}