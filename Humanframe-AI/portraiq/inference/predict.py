from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

try:
    from .visualize import draw_scoring_overlay
    from ..models.modeling import build_composition_model
    from ..utils.image_utils import build_model_transform, load_pil_rgb
except ImportError:  # pragma: no cover - script execution fallback
    from inference.visualize import draw_scoring_overlay
    from models.modeling import build_composition_model
    from utils.image_utils import build_model_transform, load_pil_rgb


class InferenceEngine:
    def __init__(
        self,
        config: Dict,
        checkpoint_path: str,
        cpu_optimized: bool = False,
    ) -> None:
        self.config = config
        self.device = self._resolve_device(config)

        self.model, _ = build_composition_model(config)
        self.model = self.model.to(self.device)
        self._load_checkpoint(checkpoint_path)

        if cpu_optimized and self.device.type == "cpu":
            self.model = self._optimize_for_cpu(self.model)

        backbone_name = config["model"]["backbone"]
        self.transform = build_model_transform(
            image_size=int(config["data"]["image_size"]),
            backbone_name=backbone_name,
            train=False,
        )

    def _resolve_device(self, config: Dict):
        wants_cuda = config.get("gpu", {}).get("device", "cuda") == "cuda"
        if wants_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_checkpoint(self, checkpoint_path: str):
        payload = torch.load(checkpoint_path, map_location=self.device)
        state_dict = payload.get("model", payload)
        self._maybe_adapt_scorer_head(state_dict)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _maybe_adapt_scorer_head(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Support legacy checkpoints that used a smaller 2-layer scorer head."""
        w0 = state_dict.get("scorer_head.layers.0.weight")
        w3 = state_dict.get("scorer_head.layers.3.weight")
        has_legacy_layout = (
            w0 is not None
            and w3 is not None
            and "scorer_head.layers.6.weight" not in state_dict
            and "scorer_head.layers.9.weight" not in state_dict
        )
        if not has_legacy_layout:
            return

        input_dim = int(w0.shape[1])
        hidden_dim = int(w0.shape[0])
        out_dim = int(w3.shape[0])
        if out_dim != 1:
            return

        self.model.scorer_head.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )
        self.model.scorer_head.to(self.device)

    def _optimize_for_cpu(self, model):
        torch.set_num_threads(max(1, int(self.config.get("inference", {}).get("cpu_threads", 2))))
        # Prefer ARM-friendly backend on Raspberry Pi; fallback safely if unavailable.
        if hasattr(torch.backends, "quantized"):
            try:
                supported = torch.backends.quantized.supported_engines
                if "qnnpack" in supported:
                    torch.backends.quantized.engine = "qnnpack"
                elif "fbgemm" in supported:
                    torch.backends.quantized.engine = "fbgemm"
            except Exception:
                pass

        try:
            return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        except Exception as exc:
            print(
                f"Warning: CPU dynamic quantization unavailable on this platform ({exc}). "
                "Falling back to non-quantized CPU inference."
            )
            return model

    def predict(self, image_path: str) -> Dict:
        image_pil = load_pil_rgb(Path(image_path))

        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw_score = self.model(image_tensor)
            predicted_score = float(torch.clamp(raw_score, 0.0, 10.0).item())

        overlay = draw_scoring_overlay(image_pil, predicted_score=predicted_score)

        return {
            "image_path": image_path,
            "predicted_score": round(predicted_score, 4),
            "final_score": round(predicted_score, 4),
            "overlay": overlay,
        }

    def predict_score(self, image_path: str) -> Dict:
        """Score-only inference path (no overlay rendering)."""
        image_pil = load_pil_rgb(Path(image_path))
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw_score = self.model(image_tensor)
            predicted_score = float(torch.clamp(raw_score, 0.0, 10.0).item())

        return {
            "image_path": image_path,
            "predicted_score": round(predicted_score, 4),
            "final_score": round(predicted_score, 4),
        }
