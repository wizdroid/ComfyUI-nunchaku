"""
This module provides the :class:`NunchakuFlux2DiTLoader` class for loading Nunchaku FLUX.2 models
(e.g. FLUX.2-klein-9B-Nunchaku).

It mirrors the UX of NunchakuFluxDiTLoader but accounts for Flux2-specific differences:
- NunchakuFlux2Transformer2DModel.from_pretrained does not support offload (raises NotImplementedError).
- Attention is set via set_attention_backend instead of set_attention_impl.
- No apply_cache_on_transformer adapter exists yet; we use the model's enable_cache API.
- The model uses 4-axis RoPE (axes_dim typically [32,32,32,32]), global_modulation, no vector_in (y is unused),
  and a different forward signature (no pooled_projections).
"""

import gc
import json
import logging
import os

import comfy.model_management
import comfy.model_patcher
import torch
from comfy.supported_models import Flux2

from nunchaku.models.transformers.transformer_flux2 import NunchakuFlux2Transformer2DModel
from nunchaku.utils import is_turing

from ...wrappers.flux2 import ComfyFlux2Wrapper
from ..utils import get_filename_list, get_full_path_or_raise

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuFlux2DiTLoader:
    """
    Loader for Nunchaku FLUX.2 models (e.g. FLUX.2-klein-9B-Nunchaku).

    This class manages model loading, device selection, attention backend selection,
    and first-block caching. CPU offload is not supported by the underlying nunchaku Flux2 loader
    and is therefore ignored (the model is always loaded directly to the target device).

    Attributes
    ----------
    transformer : :class:`~nunchaku.models.transformers.transformer_flux2.NunchakuFlux2Transformer2DModel` or None
        The loaded transformer model.
    metadata : dict or None
        Metadata associated with the loaded model (from safetensors).
    model_path : str or None
        Path to the loaded model.
    device : torch.device or None
        Device on which the model is loaded.
    data_type : str or None
        Data type used for inference.
    patcher : object or None
        ComfyUI model patcher instance.
    """

    def __init__(self):
        """
        Initialize the NunchakuFlux2DiTLoader.
        """
        self.transformer = None
        self.metadata = None
        self.model_path = None
        self.device = None
        self.data_type = None
        self.patcher = None
        self.device = comfy.model_management.get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        """
        Define the input types and tooltips for the node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and their descriptions for the node interface.
        """
        safetensor_files = get_filename_list("diffusion_models")

        ngpus = torch.cuda.device_count()

        all_turing = True
        for i in range(torch.cuda.device_count()):
            if not is_turing(f"cuda:{i}"):
                all_turing = False

        if all_turing:
            attention_options = ["nunchaku-fp16"]  # turing GPUs do not support flashattn2
            dtype_options = ["float16"]
        else:
            attention_options = ["nunchaku-fp16", "flashattn2"]
            dtype_options = ["bfloat16", "float16"]

        return {
            "required": {
                "model_path": (
                    safetensor_files,
                    {
                        "tooltip": "The Nunchaku FLUX.2 safetensors file (e.g. flux2-klein-9B-nun-kv.safetensors). "
                        "Only Nunchaku-quantized Flux2 models are supported."
                    },
                ),
                "cache_threshold": (
                    "FLOAT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1,
                        "step": 0.001,
                        "tooltip": "First-block cache residual difference threshold. 0 disables caching. "
                        "Higher values increase speed at the cost of potential quality loss.",
                    },
                ),
                "attention": (
                    attention_options,
                    {
                        "tooltip": "Attention implementation. 'nunchaku-fp16' can be faster on 30/40/50-series GPUs. "
                        "'flashattn2' is the standard backend."
                    },
                ),
                "data_type": (
                    dtype_options,
                    {
                        "tooltip": "Data type for the transformer. bfloat16 is recommended on Ampere+ GPUs. "
                        "float16 may be required on older GPUs."
                    },
                ),
                "device_id": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": max(0, ngpus - 1),
                        "tooltip": "CUDA device ID to use for this model.",
                    },
                ),
            },
            "optional": {},
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku FLUX.2 DiT Loader"
    OUTPUT_NODE = False

    def load_model(self, model_path: str, cache_threshold: float, attention: str, data_type: str, device_id: int):
        """
        Load a Nunchaku FLUX.2 model and wrap it for ComfyUI.

        Parameters
        ----------
        model_path : str
            Filename (relative to ComfyUI's diffusion_models directory) of the Nunchaku Flux2 safetensors.
        cache_threshold : float
            Residual diff threshold for first-block cache (0 disables).
        attention : str
            Attention backend name ("nunchaku-fp16" or "flashattn2").
        data_type : str
            "bfloat16" or "float16".
        device_id : int
            CUDA device index.

        Returns
        -------
        tuple
            A tuple containing the loaded and patched model (Comfy ModelPatcher wrapping a Flux2 model_base).
        """
        device = torch.device(f"cuda:{device_id}")

        model_path = get_full_path_or_raise("diffusion_models", model_path)

        if device_id >= torch.cuda.device_count():
            raise ValueError(f"Invalid device_id: {device_id}. Only {torch.cuda.device_count()} GPUs available.")

        gpu_properties = torch.cuda.get_device_properties(device_id)
        gpu_memory = gpu_properties.total_memory / (1024**2)
        gpu_name = gpu_properties.name
        logger.debug(f"GPU {device_id} ({gpu_name}) Memory: {gpu_memory} MiB")

        # NOTE: NunchakuFlux2Transformer2DModel.from_pretrained does not support offload.
        # We always load directly to the target device. The UI option is omitted for this node.
        cpu_offload_enabled = False

        if (
            self.model_path != model_path
            or self.device != device
            or self.data_type != data_type
        ):
            if self.transformer is not None:
                model_size = comfy.model_management.module_size(self.transformer)
                transformer = self.transformer
                self.transformer = None
                transformer.to("cpu")
                del transformer
                gc.collect()
                comfy.model_management.cleanup_models_gc()
                comfy.model_management.soft_empty_cache()
                comfy.model_management.free_memory(model_size, device)

            # from_pretrained for Flux2 nunchaku: no 'offload' argument (raises NotImplementedError if passed).
            self.transformer, self.metadata = NunchakuFlux2Transformer2DModel.from_pretrained(
                model_path,
                device=device,
                torch_dtype=torch.float16 if data_type == "float16" else torch.bfloat16,
                return_metadata=True,
            )
            self.model_path = model_path
            self.device = device
            self.data_type = data_type

        transformer = self.transformer

        # Apply first-block cache if requested. Flux2 nunchaku exposes enable_cache directly.
        if cache_threshold > 0:
            try:
                transformer.enable_cache(residual_diff_threshold=cache_threshold)
                logger.debug(f"Enabled Flux2 cache with residual_diff_threshold={cache_threshold}")
            except Exception as e:
                logger.warning(f"Failed to enable cache on NunchakuFlux2Transformer2DModel: {e}")
        else:
            try:
                if getattr(transformer, "is_cache_enabled", None) and transformer.is_cache_enabled():
                    transformer.disable_cache()
            except Exception:
                pass

        # Set attention backend (Flux2 uses set_attention_backend, not set_attention_impl).
        try:
            if attention == "nunchaku-fp16":
                transformer.set_attention_backend("nunchaku-fp16")
            else:
                transformer.set_attention_backend("flashattn2")
        except Exception as e:
            logger.warning(f"Failed to set attention backend '{attention}': {e}")

        # Resolve comfy_config (metadata or fallback file).
        if self.metadata is None:
            comfy_config = None
        else:
            comfy_config_str = self.metadata.get("comfy_config", None)
            if comfy_config_str:
                try:
                    comfy_config = json.loads(comfy_config_str)
                except Exception:
                    comfy_config = None
            else:
                comfy_config = None

        if not comfy_config or "model_config" not in comfy_config:
            # Fallback to a config file next to this module.
            default_config_root = os.path.join(os.path.dirname(__file__), "configs")
            base = os.path.basename(model_path)
            # Strip common nunchaku / quantization prefixes and suffixes so "flux2-klein-9B-nun-kv.safetensors"
            # can resolve to "flux2-klein-9B.json".
            for token in ["svdq-int4-", "svdq-fp4-", "svdq-", "-nun-kv", "-nun", ".safetensors"]:
                base = base.replace(token, "")
            config_path = os.path.join(default_config_root, f"{base}.json")
            if not os.path.exists(config_path):
                config_path = os.path.join(default_config_root, "flux2-klein-9B.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"ComfyUI model config not found in metadata and no fallback config at {config_path}. "
                    "Please provide a 'comfy_config' entry in the safetensors metadata or add a config JSON."
                )
            logger.info(f"Loading ComfyUI model config from {config_path}")
            comfy_config = json.load(open(config_path, "r"))

        model_config_dict = comfy_config.get("model_config", {}).copy()
        if "disable_unet_model_creation" not in model_config_dict:
            model_config_dict["disable_unet_model_creation"] = True

        model_class_name = comfy_config.get("model_class", "Flux2")
        if model_class_name != "Flux2":
            logger.warning(f"Unexpected model_class '{model_class_name}' for Flux2 loader; proceeding with Flux2.")

        model_config = Flux2(model_config_dict)
        model_config.set_inference_dtype(torch.bfloat16, None)
        model_config.custom_operations = None

        model = model_config.get_model({})
        model.diffusion_model = ComfyFlux2Wrapper(
            transformer,
            config=model_config_dict,
            ctx_for_copy={
                "comfy_config": comfy_config,
                "model_config": model_config,
                "device": device,
                "device_id": device_id,
            },
        )

        model = comfy.model_patcher.ModelPatcher(model, device, device_id)
        return (model,)
