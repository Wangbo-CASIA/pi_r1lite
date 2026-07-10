import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class R1LiteRepack(transforms.DataTransformFn):
    """Repack R1Lite LeRobot or runtime observations into policy input keys."""

    def __call__(self, data: dict) -> dict:
        flat = transforms.flatten_dict(data)
        images = data.get("images")
        if images is None:
            if "observation.images.cam_high" in flat.keys():
                images = {
                    "head": flat["observation.images.cam_high"],
                    "left_wrist": flat["observation.images.cam_left_wrist"],
                    "right_wrist": flat["observation.images.cam_right_wrist"],
                }
            else:
                images = {
                    "head": flat["observation.images.head"],
                    "left_wrist": flat["observation.images.left_wrist"],
                    "right_wrist": flat["observation.images.right_wrist"],
                }

        result = {
            "images": images,
            "state": (
                data["state"]
                if "state" in data
                else flat["observations.state.qpos"]
                if "observations.state.qpos" in flat
                else flat["observation.state"]
            ),
        }
        if "actions" in data:
            result["actions"] = data["actions"]
        elif "action.qpos" in flat:
            result["actions"] = flat["action.qpos"]
        elif "action" in flat:
            result["actions"] = flat["action"]
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        elif "prompt" in flat:
            result["prompt"] = flat["prompt"]
        return result


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _compact_joint_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    # if state.shape[-1] != 53 and :
    #     raise ValueError(f"Expected R1Lite state with 53 dims, got shape {state.shape}")
    if state.shape[-1] == 53:
        return np.concatenate(
        [
            state[..., 13:19],
            state[..., 25:26],
            state[..., 39:45],
            state[..., 51:52],
        ],
        axis=-1,
    ).astype(np.float32)
    elif state.shape[-1] == 14:
        return state
    else:
        raise ValueError(f"Expected R1Lite state with 53 dims, got shape {state.shape}")


def _compact_abs_eef_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] != 53:
        raise ValueError(f"Expected R1Lite state with 53 dims, got shape {state.shape}")
    return np.concatenate(
        [
            state[..., 0:7],
            state[..., 25:26],
            state[..., 26:33],
            state[..., 51:52],
        ],
        axis=-1,
    ).astype(np.float32)


def _validate_action_dim(actions: np.ndarray, dim: int, label: str) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape[-1] != dim:
        raise ValueError(f"Expected R1Lite {label} action with {dim} dims, got shape {actions.shape}")
    return actions


EVA_CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
LEGACY_CAMERA_KEYS = ("head", "left_wrist", "right_wrist")
R1LITE_GRIPPER_THRESHOLD = 75.0
EVA_GRIPPER_OPEN = 100.0
EVA_GRIPPER_CLOSE = 0.0


def _binary_gripper_from_eva(value: np.ndarray) -> np.ndarray:
    return np.where(value > R1LITE_GRIPPER_THRESHOLD, 0.0, 1.0).astype(np.float32)


def _eva_gripper_from_legacy(value: np.ndarray) -> np.ndarray:
    return (EVA_GRIPPER_OPEN + value * (EVA_GRIPPER_CLOSE - EVA_GRIPPER_OPEN)).astype(np.float32)


def _eva_eef_to_legacy(eef: np.ndarray) -> np.ndarray:
    eef = _validate_action_dim(eef, 16, "EVA absolute EEF")
    legacy = np.empty_like(eef, dtype=np.float32)
    legacy[..., 0:3] = eef[..., 0:3]
    legacy[..., 3:7] = eef[..., [4, 5, 6, 3]]
    legacy[..., 7] = _binary_gripper_from_eva(eef[..., 7])
    legacy[..., 8:11] = eef[..., 8:11]
    legacy[..., 11:15] = eef[..., [12, 13, 14, 11]]
    legacy[..., 15] = _binary_gripper_from_eva(eef[..., 15])
    return legacy


def _legacy_eef_to_eva(eef: np.ndarray) -> np.ndarray:
    eef = _validate_action_dim(eef, 16, "legacy absolute EEF")
    eva = np.empty_like(eef, dtype=np.float32)
    eva[..., 0:3] = eef[..., 0:3]
    eva[..., 3:7] = eef[..., [6, 3, 4, 5]]
    eva[..., 7] = _eva_gripper_from_legacy(eef[..., 7])
    eva[..., 8:11] = eef[..., 8:11]
    eva[..., 11:15] = eef[..., [14, 11, 12, 13]]
    eva[..., 15] = _eva_gripper_from_legacy(eef[..., 15])
    return eva


def _abs_eef_images(images: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    image_keys = set(images)
    if set(EVA_CAMERA_KEYS).issubset(image_keys):
        return (
            _parse_image(images["cam_high"]),
            _parse_image(images["cam_left_wrist"]),
            _parse_image(images["cam_right_wrist"]),
            True,
        )
    if set(LEGACY_CAMERA_KEYS).issubset(image_keys):
        return (
            _parse_image(images["head"]),
            _parse_image(images["left_wrist"]),
            _parse_image(images["right_wrist"]),
            False,
        )
    expected = " or ".join([str(EVA_CAMERA_KEYS), str(LEGACY_CAMERA_KEYS)])
    raise ValueError(f"Expected R1Lite image keys to contain {expected}, got {tuple(images)}")


@dataclasses.dataclass(frozen=True)
class R1LiteInputs(transforms.DataTransformFn):
    """Inputs for R1Lite dual-arm joint-delta policies."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("head", "left_wrist", "right_wrist")

    def __call__(self, data: dict) -> dict:
        images = data["images"]
        extra_cameras = set(images) - set(self.EXPECTED_CAMERAS)
        if extra_cameras:
            raise ValueError(f"Unexpected R1Lite image keys: {sorted(extra_cameras)}")

        base_image = _parse_image(images["head"])
        left_wrist = _parse_image(images["left_wrist"])
        right_wrist = _parse_image(images["right_wrist"])
        
        inputs = {
            "state": _compact_joint_state(data["state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = _validate_action_dim(data["actions"], 14, "joint-delta")

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class R1LiteOutputs(transforms.DataTransformFn):
    """Outputs for R1Lite dual-arm joint-delta policies."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}


@dataclasses.dataclass(frozen=True)
class R1LiteAbsEEFInputs(transforms.DataTransformFn):
    """Inputs for R1Lite dual-arm absolute EEF policies served to EVA-CLIENT."""

    def __call__(self, data: dict) -> dict:
        base_image, left_wrist_image, right_wrist_image, is_eva_payload = _abs_eef_images(data["images"])
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] == 53:
            state = _compact_abs_eef_state(state)
        elif state.shape[-1] == 16:
            state = _eva_eef_to_legacy(state)
        else:
            raise ValueError(f"Expected R1Lite absolute EEF state with 16 or 53 dims, got shape {state.shape}")

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            actions = _validate_action_dim(data["actions"], 16, "absolute EEF")
            inputs["actions"] = _eva_eef_to_legacy(actions) if is_eva_payload else actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class R1LiteAbsEEFOutputs(transforms.DataTransformFn):
    """Outputs for R1Lite dual-arm absolute EEF pose policies."""

    def __call__(self, data: dict) -> dict:
        return {
            "actions": _legacy_eef_to_eva(np.asarray(data["actions"], dtype=np.float32)[:, :16]),
            "action_mode": "eef",
        }


@dataclasses.dataclass(frozen=True)
class R1LiteDeltaEEFInputs(transforms.DataTransformFn):
    """Inputs for R1Lite dual-arm relative EEF delta policies."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("head", "left_wrist", "right_wrist")

    def __call__(self, data: dict) -> dict:
        images = data["images"]
        extra_cameras = set(images) - set(self.EXPECTED_CAMERAS)
        if extra_cameras:
            raise ValueError(f"Unexpected R1Lite image keys: {sorted(extra_cameras)}")

        inputs = {
            "state": _compact_abs_eef_state(data["state"]),
            "image": {
                "base_0_rgb": _parse_image(images["head"]),
                "left_wrist_0_rgb": _parse_image(images["left_wrist"]),
                "right_wrist_0_rgb": _parse_image(images["right_wrist"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = _validate_action_dim(data["actions"], 14, "relative EEF")

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class R1LiteDeltaEEFOutputs(transforms.DataTransformFn):
    """Outputs for R1Lite dual-arm relative EEF delta policies."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}
