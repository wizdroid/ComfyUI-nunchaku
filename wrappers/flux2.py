"""
This module provides a wrapper for the :class:`~nunchaku.models.transformers.transformer_flux2.NunchakuFlux2Transformer2DModel`,
enabling integration with ComfyUI forward, reference latents, and first-block caching.
"""

from typing import Callable

import torch
from comfy.ldm.common_dit import pad_to_patch_size
from einops import rearrange, repeat
from torch import nn

from nunchaku.models.transformers.transformer_flux2 import NunchakuFlux2Transformer2DModel

try:
    from nunchaku.caching.fbcache import cache_context, create_cache_context
except Exception:
    cache_context = None
    create_cache_context = None


class ComfyFlux2Wrapper(nn.Module):
    """
    Wrapper for :class:`~nunchaku.models.transformers.transformer_flux2.NunchakuFlux2Transformer2DModel`
    to support ComfyUI workflows, reference latents, and caching.

    Parameters
    ----------
    model : :class:`~nunchaku.models.transformers.transformer_flux2.NunchakuFlux2Transformer2DModel`
        The underlying Nunchaku Flux2 model to wrap.
    config : dict
        Model configuration dictionary (the inner "model_config" from comfy_config).
    customized_forward : Callable, optional
        Optional custom forward function.
    forward_kwargs : dict, optional
        Additional keyword arguments for the forward pass.
    ctx_for_copy : dict
        A dict that holds initialization context for later duplication of this wrapper.

    Attributes
    ----------
    model : :class:`~nunchaku.models.transformers.transformer_flux2.NunchakuFlux2Transformer2DModel`
        The wrapped model.
    dtype : torch.dtype
        Data type of the model parameters.
    config : dict
        Model configuration.
    loras : list
        List of LoRA metadata (currently unused for Flux2; kept for API compatibility).
    customized_forward : Callable or None
        Custom forward function if provided.
    forward_kwargs : dict
        Additional arguments for the forward pass.
    ctx_for_copy : dict
        Context for duplication.
    """

    def __init__(
        self,
        model: NunchakuFlux2Transformer2DModel,
        config: dict,
        customized_forward: Callable = None,
        forward_kwargs: dict | None = None,
        ctx_for_copy: dict | None = None,
    ):
        super(ComfyFlux2Wrapper, self).__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.config = config if config is not None else {}
        self.loras = []

        self.customized_forward = customized_forward
        self.forward_kwargs = {} if forward_kwargs is None else forward_kwargs
        self.ctx_for_copy = (ctx_for_copy or {}).copy()

        self._prev_timestep = None
        self._cache_context = None

    def process_img(self, x, index=0, h_offset=0, w_offset=0, transformer_options=None):
        """
        Preprocess an input latent tensor for the Flux2 model.

        Pads (if needed) and rearranges the latent into tokens and generates corresponding image IDs
        using the configured number of axes (typically 4 for Flux2 Klein).

        Parameters
        ----------
        x : torch.Tensor
            Input latent tensor of shape (batch, channels, height, width). For Flux2 this is usually 128 channels.
        index : int, optional
            Index for image ID encoding (used for reference latents).
        h_offset : int, optional
            Height offset for patch IDs.
        w_offset : int, optional
            Width offset for patch IDs.
        transformer_options : dict, optional
            Comfy transformer options (may contain rope_options for advanced positioning).

        Returns
        -------
        img : torch.Tensor
            Rearranged latent tensor of shape (batch, num_tokens, channels).
        img_ids : torch.Tensor
            Image ID tensor of shape (batch, num_tokens, num_axes).
        """
        if transformer_options is None:
            transformer_options = {}

        bs, c, h, w = x.shape
        patch_size = self.config.get("patch_size", 1)
        x = pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        h_len = (h + (patch_size // 2)) // patch_size
        w_len = (w + (patch_size // 2)) // patch_size

        h_offset = (h_offset + (patch_size // 2)) // patch_size
        w_offset = (w_offset + (patch_size // 2)) // patch_size

        steps_h = h_len
        steps_w = w_len

        rope_options = transformer_options.get("rope_options", None)
        if rope_options is not None:
            h_len = (h_len - 1.0) * rope_options.get("scale_y", 1.0) + 1.0
            w_len = (w_len - 1.0) * rope_options.get("scale_x", 1.0) + 1.0

            index += rope_options.get("shift_t", 0.0)
            h_offset += rope_options.get("shift_y", 0.0)
            w_offset += rope_options.get("shift_x", 0.0)

        axes_dim = self.config.get("axes_dim", [32, 32, 32, 32])
        num_axes = len(axes_dim)

        img_ids = torch.zeros((steps_h, steps_w, num_axes), device=x.device, dtype=torch.float32)
        # Comfy's Flux (incl. Flux2) only populates the first 3 axes even when len(axes_dim) == 4.
        # The 4th axis (if present) stays at 0.
        img_ids[:, :, 0] = img_ids[:, :, 1] + index
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(
            h_offset, h_len - 1 + h_offset, steps=steps_h, device=x.device, dtype=torch.float32
        ).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(
            w_offset, w_len - 1 + w_offset, steps=steps_w, device=x.device, dtype=torch.float32
        ).unsqueeze(0)
        # axes 3+ (if any) remain zero

        return img, repeat(img_ids, "h w c -> b (h w) c", b=bs)

    def forward(
        self,
        x,
        timestep,
        context,
        y=None,
        guidance=None,
        ref_latents=None,
        control=None,
        transformer_options=None,
        **kwargs,
    ):
        """
        Forward pass for the wrapped Flux2 model.

        Parameters
        ----------
        x : torch.Tensor
            Input latent tensor (B, C, H, W). C is typically 128 for Flux2.
        timestep : float or torch.Tensor
            Diffusion timestep.
        context : torch.Tensor
            Text encoder hidden states (B, seq_len, context_in_dim), e.g. 12288 for Klein.
        y : torch.Tensor or None
            Pooled projections / ADM. Not used by Flux2 Klein (vec_in_dim=None); accepted for API compatibility.
        guidance : torch.Tensor or float or None
            Guidance value. Klein has guidance_embeds=False, but we still forward it if provided.
        ref_latents : list[torch.Tensor] or None
            Optional reference latents for img2img / reference workflows.
        control : dict or None
            ControlNet samples (input/output). Currently ignored for Flux2 nunchaku (no controlnet injection in forward).
        transformer_options : dict, optional
            ComfyUI transformer options (patches, rope_options, etc.).
        **kwargs
            Additional keyword arguments (e.g. ref_latents_method).

        Returns
        -------
        torch.Tensor
            Output latent tensor of shape (B, out_channels, H, W) matching the input spatial size.
        """
        if transformer_options is None:
            transformer_options = {}

        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 1:
                timestep_float = timestep.item()
            else:
                timestep_float = timestep.flatten()[0].item()
        else:
            assert isinstance(timestep, float)
            timestep_float = timestep

        model = self.model
        assert isinstance(model, NunchakuFlux2Transformer2DModel)

        bs, c, h_orig, w_orig = x.shape
        patch_size = self.config.get("patch_size", 1)
        h_len = (h_orig + (patch_size // 2)) // patch_size
        w_len = (w_orig + (patch_size // 2)) // patch_size

        img, img_ids = self.process_img(x, transformer_options=transformer_options)
        img_tokens = img.shape[1]

        # Reference latents (simplified handling, mirroring the Flux1 nunchaku wrapper approach)
        if ref_latents is not None:
            h = 0
            w = 0
            for ref in ref_latents:
                h_offset = 0
                w_offset = 0
                if ref.shape[-2] + h > ref.shape[-1] + w:
                    w_offset = w
                else:
                    h_offset = h

                kontext, kontext_ids = self.process_img(
                    ref, index=1, h_offset=h_offset, w_offset=w_offset, transformer_options=transformer_options
                )
                img = torch.cat([img, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)
                h = max(h, ref.shape[-2] + h_offset)
                w = max(w, ref.shape[-1] + w_offset)

        axes_dim = self.config.get("axes_dim", [32, 32, 32, 32])
        num_axes = len(axes_dim)
        txt_ids = torch.zeros((bs, context.shape[1], num_axes), device=x.device, dtype=torch.float32)

        # LoRA composition: currently disabled in nunchaku for Flux2 (SVDQLoRAMixin not present).
        # We keep the slot for future compatibility but do nothing if loras are supplied.
        if self.loras:
            # Placeholder: if future nunchaku Flux2 builds support LoRA, implement compose + update_lora_params here.
            pass

        # ControlNet: Flux2 nunchaku forward does not currently expose controlnet_block_samples.
        # We accept the argument for compatibility but do not inject.
        if control is not None:
            # Future: if supported, map control["input"] / control["output"] into the call.
            pass

        # Caching (first-block cache via nunchaku's cache_context when enabled on the model)
        use_cache = False
        try:
            if getattr(model, "is_cache_enabled", None) is not None:
                use_cache = bool(model.is_cache_enabled())
            elif getattr(model, "_is_cached", False):
                use_cache = True
        except Exception:
            use_cache = False

        if cache_context is not None and (use_cache or getattr(model, "residual_diff_threshold_multi", 0)):
            cache_invalid = False
            if self._prev_timestep is None:
                cache_invalid = True
            elif self._prev_timestep < timestep_float + 1e-5:
                cache_invalid = True

            if cache_invalid or self._cache_context is None:
                self._cache_context = create_cache_context()

            self._prev_timestep = timestep_float
            with cache_context(self._cache_context):
                if self.customized_forward is None:
                    out = model(
                        hidden_states=img,
                        encoder_hidden_states=context,
                        timestep=timestep,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        guidance=guidance if self.config.get("guidance_embed") else None,
                        **self.forward_kwargs,
                    ).sample
                else:
                    out = self.customized_forward(
                        model,
                        hidden_states=img,
                        encoder_hidden_states=context,
                        timestep=timestep,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        guidance=guidance if self.config.get("guidance_embed") else None,
                        **self.forward_kwargs,
                    ).sample
        else:
            if self.customized_forward is None:
                out = model(
                    hidden_states=img,
                    encoder_hidden_states=context,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance if self.config.get("guidance_embed") else None,
                    **self.forward_kwargs,
                ).sample
            else:
                out = self.customized_forward(
                    model,
                    hidden_states=img,
                    encoder_hidden_states=context,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance if self.config.get("guidance_embed") else None,
                    **self.forward_kwargs,
                ).sample

        # The nunchaku Flux2 model returns a diffusers-like object with .sample of shape (B, T, C)
        # We need to reshape back to (B, out_channels, H, W) using the original spatial size.
        # For patch_size=1 and out_channels=128 this is straightforward.
        out_channels = self.config.get("out_channels", 128)
        # out shape from model: (B, num_tokens_total, patch_size*patch_size*out_channels)
        # For ref_latents we concatenated extra tokens; we only want the first img_tokens for the main image.
        out = out[:, :img_tokens, :]
        ps = patch_size
        h_len = (h_orig + (ps // 2)) // ps
        w_len = (w_orig + (ps // 2)) // ps
        out = rearrange(
            out,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h_len,
            w=w_len,
            ph=ps,
            pw=ps,
            c=out_channels,
        )
        # Crop to original size if padding was added
        out = out[:, :, :h_orig, :w_orig]
        return out
