"""NumPy inference for the archived TensorFlow 1 Hide-and-Seek policy.

The original checkpoint format is retained.  This module evaluates the exact
policy network operations without requiring an unavailable TensorFlow 1 ARM64
runtime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class NumpyHideAndSeekPolicy:
    def __init__(self, checkpoint: str | Path):
        with np.load(checkpoint, allow_pickle=False) as archive:
            self.weights = {
                key: archive[key].astype(np.float32, copy=True)
                for key in archive.files
                if key != "policy_fn_and_args"
            }
        self.cell = None
        self.hidden = None

    def reset(self, n_agents: int = 4):
        self.cell = np.zeros((n_agents, 1, 256), dtype=np.float32)
        self.hidden = np.zeros((n_agents, 1, 256), dtype=np.float32)

    def _get(self, name: str):
        return self.weights[name]

    def _dense(self, value, name: str, *, relu: bool = False):
        value = value @ self._get(f"{name}/kernel:0")
        value = value + self._get(f"{name}/bias:0")
        return np.maximum(value, 0.0) if relu else value

    def _normalize(self, value, prefix: str):
        mean = self._get(f"policy/{prefix}/mean:0")
        square = self._get(f"policy/{prefix}/sq:0")
        debias = np.maximum(
            self._get(f"policy/{prefix}/debiasing_term:0"), 1e-6)
        mean = mean / debias
        std = np.sqrt(np.maximum(square / debias - np.square(mean), 1e-2))
        return np.clip((value.astype(np.float32) - mean) / std, -5.0, 5.0)

    def _convolve_lidar(self, lidar):
        kernel = self._get(
            "policy/policy_net//circ_conv1d0/conv1d/kernel:0")
        bias = self._get(
            "policy/policy_net//circ_conv1d0/conv1d/bias:0")
        padded = np.concatenate([lidar[:, :, -1:], lidar, lidar[:, :, :1]], axis=2)
        output = np.zeros(lidar.shape[:3] + (kernel.shape[-1],), dtype=np.float32)
        for offset in range(kernel.shape[0]):
            output += padded[:, :, offset:offset + lidar.shape[2]] @ kernel[offset]
        return np.maximum(output + bias, 0.0)

    def _attention(self, entities, mask):
        base = "policy/policy_net//self-attention8/residual_sa_block8"
        qk = self._dense(
            entities, f"{base}/self_attention/qkv_embed/qk_embed")
        value = self._dense(
            entities, f"{base}/self_attention/qkv_embed/v_embed")

        batch, timesteps, n_entities, _ = entities.shape
        qk = qk.reshape(batch, timesteps, n_entities, 4, 32, 2)
        query = qk[..., 0].transpose(0, 1, 3, 2, 4)
        key = qk[..., 1].transpose(0, 1, 3, 4, 2)
        value = value.reshape(batch, timesteps, n_entities, 4, 32)
        value = value.transpose(0, 1, 3, 2, 4)

        logits = query @ key / np.sqrt(32.0)
        attention_mask = mask[:, :, None, None, :].astype(np.float32)
        logits = logits - (1.0 - attention_mask) * 1e10
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        probabilities = np.exp(logits) * attention_mask
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True) + 1e-10
        probabilities *= attention_mask

        attended = probabilities @ value
        attended = attended.transpose(0, 1, 3, 2, 4)
        attended = attended.reshape(batch, timesteps, n_entities, 128)
        return entities + self._dense(attended, f"{base}/mlp1")

    def _lstm(self, value):
        if self.cell is None or self.cell.shape[0] != value.shape[0]:
            self.reset(value.shape[0])
        name = "policy/policy_net//lstm12/rnn/basic_lstm_cell"
        gates = np.concatenate([value, self.hidden], axis=-1)
        gates = self._dense(gates, name)
        input_gate, candidate, forget_gate, output_gate = np.split(gates, 4, axis=-1)
        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
        self.cell = (
            sigmoid(forget_gate + 1.0) * self.cell
            + sigmoid(input_gate) * np.tanh(candidate)
        )
        self.hidden = np.tanh(self.cell) * sigmoid(output_gate)
        return self.hidden

    def act(self, observation):
        inputs = {
            key: np.expand_dims(value, 1)
            for key, value in observation.items()
        }
        inputs["observation_self"] = self._normalize(
            inputs["observation_self"], "normalize_self_obs/obsfilter")
        for key in ("agent_qpos_qvel", "box_obs", "lidar", "ramp_obs"):
            inputs[key] = self._normalize(
                inputs[key], f"normalize_{key}/obsfilter/{key}")

        main = inputs["observation_self"]
        lidar = self._convolve_lidar(inputs["lidar"])
        main = np.concatenate([main, lidar.reshape(lidar.shape[:2] + (-1,))], axis=-1)

        nodes = {
            "agent_qpos_qvel": np.concatenate([
                np.repeat(main[:, :, None], inputs["agent_qpos_qvel"].shape[2], axis=2),
                inputs["agent_qpos_qvel"],
            ], axis=-1),
            "box_obs": np.concatenate([
                np.repeat(main[:, :, None], inputs["box_obs"].shape[2], axis=2),
                inputs["box_obs"],
            ], axis=-1),
            "ramp_obs": np.concatenate([
                np.repeat(main[:, :, None], inputs["ramp_obs"].shape[2], axis=2),
                inputs["ramp_obs"],
            ], axis=-1),
            "main": main,
        }
        ordered = ("agent_qpos_qvel", "box_obs", "ramp_obs", "main")
        for index, key in enumerate(ordered):
            nodes[key] = self._dense(
                nodes[key], f"policy/policy_net//dense6-{index}", relu=True)

        entities = np.concatenate([
            nodes["agent_qpos_qvel"],
            nodes["box_obs"],
            nodes["ramp_obs"],
            nodes["main"][:, :, None],
        ], axis=2)
        entity_mask = np.concatenate([
            inputs["mask_aa_obs"],
            inputs["mask_ab_obs"],
            inputs["mask_ar_obs"],
            np.ones(inputs["mask_ar_obs"].shape[:2] + (1,), dtype=np.float32),
        ], axis=-1)
        entities = self._attention(entities, entity_mask)
        masked = entities * entity_mask[..., None]
        pooled = (
            np.sum(masked, axis=-2)
            / (np.sum(entity_mask, axis=-1, keepdims=True) + 1e-5)
        )

        main = np.concatenate([nodes["main"], pooled], axis=-1)
        main = self._dense(
            main, "policy/policy_net//dense11-0", relu=True)
        main = self._lstm(main)

        beta = self._get("policy/policy_net//layernorm13/LayerNorm/beta:0")
        gamma = self._get("policy/policy_net//layernorm13/LayerNorm/gamma:0")
        mean = np.mean(main, axis=-1, keepdims=True)
        variance = np.mean(np.square(main - mean), axis=-1, keepdims=True)
        main = (main - mean) / np.sqrt(variance + 1e-12) * gamma + beta

        actions = {}
        for key in ("action_movement", "action_pull", "action_glueall"):
            logits = self._dense(main, f"policy/policy_out/{key}/dense")[:, 0]
            if key == "action_movement":
                actions[key] = np.argmax(logits.reshape(-1, 3, 11), axis=-1)
            else:
                actions[key] = np.argmax(logits, axis=-1)
        return actions
