import math
import random

import torch

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import nodes


FIXED_ASPECT_RATIOS = {
    "1:1": (1.0, 1.0),
    "4:5": (4.0, 5.0),
    "5:4": (5.0, 4.0),
    "2:3": (2.0, 3.0),
    "3:2": (3.0, 2.0),
    "3:4": (3.0, 4.0),
    "4:3": (4.0, 3.0),
    "9:16": (9.0, 16.0),
    "16:9": (16.0, 9.0),
    "2.35:1": (2.35, 1.0),
    "21:9": (21.0, 9.0),
}


ASPECT_RATIOS = {"Random": None, **FIXED_ASPECT_RATIOS}
UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]


def _round_to_multiple(value, multiple):
    if multiple <= 1:
        return int(round(value))
    return int(round(value / multiple) * multiple)


def _resolve_aspect_ratio(aspect_ratio, random_seed):
    if aspect_ratio != "Random":
        return aspect_ratio

    aspect_names = list(FIXED_ASPECT_RATIOS.keys())
    return aspect_names[random.Random(random_seed).randrange(len(aspect_names))]


def _dimensions_for(aspect_ratio, megapixels, multiple, random_seed=0):
    selected_aspect_ratio = _resolve_aspect_ratio(aspect_ratio, random_seed)
    ratio_w, ratio_h = FIXED_ASPECT_RATIOS[selected_aspect_ratio]
    total_pixels = megapixels * 1024 * 1024
    scale = math.sqrt(total_pixels / (ratio_w * ratio_h))
    width = max(multiple, _round_to_multiple(ratio_w * scale, multiple))
    height = max(multiple, _round_to_multiple(ratio_h * scale, multiple))
    return width, height


def _sigma_schedule(model, steps, sampler_name, scheduler):
    device = getattr(model, "load_device", comfy.model_management.get_torch_device())
    sampler = comfy.samplers.KSampler(
        model,
        steps=steps,
        device=device,
        sampler=sampler_name,
        scheduler=scheduler,
        denoise=1.0,
        model_options=model.model_options,
    )
    return sampler.sigmas.detach().cpu()


def _clamp(value, low, high):
    return max(low, min(high, value))


def _stage2_start_index(stage2_steps, stage2_sigmas, handoff_percent):
    max_start = len(stage2_sigmas) - 2
    if max_start < 1:
        raise ValueError("stage2_steps must produce at least one non-terminal handoff sigma.")
    requested = round(stage2_steps * handoff_percent / 100.0)
    return _clamp(requested, 1, max_start)


def _nearest_stage1_end_index(stage1_sigmas, boundary_sigma):
    max_end = len(stage1_sigmas) - 2
    if max_end < 1:
        raise ValueError("stage1_steps must produce at least one non-terminal handoff sigma.")

    boundary = float(boundary_sigma)
    if boundary <= 0.0:
        raise ValueError("The handoff sigma must be greater than zero.")
    if boundary >= float(stage1_sigmas[0]):
        raise ValueError(
            "The stage-2 handoff sigma is outside the stage-1 schedule. "
            "Use a later handoff percent or compatible model sampling settings."
        )

    candidates = [idx for idx in range(1, max_end + 1) if float(stage1_sigmas[idx - 1]) > boundary]
    if not candidates:
        raise ValueError("Could not find a valid stage-1 handoff step for the boundary sigma.")

    return min(candidates, key=lambda idx: abs(float(stage1_sigmas[idx]) - boundary))


def _build_sigma_pair(stage1_model, stage2_model, stage1_steps, stage2_steps, stage1_sampler, stage1_scheduler, stage2_sampler, stage2_scheduler, handoff_percent):
    stage1_sigmas = _sigma_schedule(stage1_model, stage1_steps, stage1_sampler, stage1_scheduler)
    stage2_sigmas = _sigma_schedule(stage2_model, stage2_steps, stage2_sampler, stage2_scheduler)

    stage2_start = _stage2_start_index(stage2_steps, stage2_sigmas, handoff_percent)
    boundary_sigma = stage2_sigmas[stage2_start].clone()
    stage1_end = _nearest_stage1_end_index(stage1_sigmas, boundary_sigma)

    stage1_custom = torch.cat((stage1_sigmas[:stage1_end], boundary_sigma.reshape(1)))
    stage2_custom = stage2_sigmas[stage2_start:].clone()
    stage2_custom[0] = boundary_sigma

    return stage1_custom, stage2_custom, stage1_end, stage2_start, float(boundary_sigma)


def _sample_with_sigmas(model, seed, cfg, sampler_name, scheduler, positive, negative, latent, sigmas, disable_noise):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = latent.get("noise_mask", None)
    steps = max(1, len(sigmas) - 1)
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    device = getattr(model, "load_device", comfy.model_management.get_torch_device())

    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
        disable_noise=disable_noise,
        force_full_denoise=False,
        noise_mask=noise_mask,
        sigmas=sigmas.to(device),
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out


def _zero_out_conditioning(conditioning):
    out = []
    for item in conditioning:
        metadata = item[1].copy()
        pooled_output = metadata.get("pooled_output", None)
        if pooled_output is not None:
            metadata["pooled_output"] = torch.zeros_like(pooled_output)
        conditioning_lyrics = metadata.get("conditioning_lyrics", None)
        if conditioning_lyrics is not None:
            metadata["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
        out.append([torch.zeros_like(item[0]), metadata])
    return out


def _latent_spatial_downscale(model):
    latent_format = model.get_model_object("latent_format")
    return getattr(latent_format, "spacial_downscale_ratio", 8)


def _target_latent_size(latent, model, final_width, final_height):
    samples = latent["samples"]
    downscale = _latent_spatial_downscale(model)
    current_width = samples.shape[-1] * downscale
    current_height = samples.shape[-2] * downscale

    if final_width == 0:
        final_width = max(64, round(current_width * final_height / current_height))
    elif final_height == 0:
        final_height = max(64, round(current_height * final_width / current_width))

    final_width = max(64, int(final_width))
    final_height = max(64, int(final_height))

    target_latent_width = max(1, round(final_width / downscale))
    target_latent_height = max(1, round(final_height / downscale))
    return target_latent_width, target_latent_height


def _will_upscale_latent(latent, model, final_width, final_height):
    if final_width == 0 and final_height == 0:
        return False

    target_latent_width, target_latent_height = _target_latent_size(latent, model, final_width, final_height)
    samples = latent["samples"]
    return target_latent_width != samples.shape[-1] or target_latent_height != samples.shape[-2]


def _upscale_latent_if_needed(latent, model, final_width, final_height, upscale_method):
    if final_width == 0 and final_height == 0:
        return latent, False

    samples = latent["samples"]
    target_latent_width, target_latent_height = _target_latent_size(latent, model, final_width, final_height)

    if target_latent_width == samples.shape[-1] and target_latent_height == samples.shape[-2]:
        return latent, False

    out = latent.copy()
    out["samples"] = comfy.utils.common_upscale(
        samples,
        target_latent_width,
        target_latent_height,
        upscale_method,
        "disabled",
    )
    return out, True


class KreaDualResolutionSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (list(ASPECT_RATIOS.keys()), {"default": "1:1"}),
                "base_megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "final_megapixels": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "multiple": ("INT", {"default": 16, "min": 8, "max": 128, "step": 8}),
                "random_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xffffffffffffffff,
                        "control_after_generate": True,
                        "advanced": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("base_width", "base_height", "final_width", "final_height")
    FUNCTION = "execute"
    CATEGORY = "Ashen3"

    def execute(self, aspect_ratio, base_megapixels, final_megapixels, multiple, random_seed=0):
        base_width, base_height = _dimensions_for(aspect_ratio, base_megapixels, multiple, random_seed)
        final_width, final_height = _dimensions_for(aspect_ratio, final_megapixels, multiple, random_seed)
        return (base_width, base_height, final_width, final_height)


class KreaTwoStageSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage1_model": ("MODEL",),
                "stage2_model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "handoff_percent": ("FLOAT", {"default": 16.67, "min": 0.01, "max": 99.99, "step": 0.01, "round": 0.01}),
                "stage1_steps": ("INT", {"default": 52, "min": 2, "max": 10000}),
                "stage1_cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "stage1_sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "stage1_scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "stage2_steps": ("INT", {"default": 12, "min": 2, "max": 10000}),
                "stage2_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "stage2_sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "stage2_scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "final_width": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "final_height": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "upscale_method": (UPSCALE_METHODS, {"default": "bislerp", "advanced": True}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "Ashen3"

    def sample(
        self,
        stage1_model,
        stage2_model,
        positive,
        negative,
        latent_image,
        seed,
        handoff_percent,
        stage1_steps,
        stage1_cfg,
        stage1_sampler_name,
        stage1_scheduler,
        stage2_steps,
        stage2_cfg,
        stage2_sampler_name,
        stage2_scheduler,
        final_width,
        final_height,
        upscale_method="bislerp",
    ):
        stage1_sigmas, stage2_sigmas, stage1_end, stage2_start, boundary_sigma = _build_sigma_pair(
            stage1_model,
            stage2_model,
            stage1_steps,
            stage2_steps,
            stage1_sampler_name,
            stage1_scheduler,
            stage2_sampler_name,
            stage2_scheduler,
            handoff_percent,
        )
        upscale_requested = _will_upscale_latent(latent_image, stage2_model, final_width, final_height)
        stage1_run_sigmas = stage1_sigmas.clone()
        if upscale_requested:
            stage1_run_sigmas[-1] = 0.0

        print(
            "Krea Two-Stage Sampler: "
            f"stage1_end={stage1_end}, stage2_start={stage2_start}, boundary_sigma={boundary_sigma:.8f}, "
            f"noise_mode={'fresh_high_res' if upscale_requested else 'carry_leftover'}"
        )

        stage1 = _sample_with_sigmas(
            stage1_model,
            seed,
            stage1_cfg,
            stage1_sampler_name,
            stage1_scheduler,
            positive,
            negative,
            latent_image,
            stage1_run_sigmas,
            disable_noise=False,
        )

        stage1, did_upscale = _upscale_latent_if_needed(stage1, stage2_model, final_width, final_height, upscale_method)
        stage2_negative = _zero_out_conditioning(negative) if math.isclose(stage2_cfg, 1.0, rel_tol=0.0, abs_tol=1e-6) else negative

        stage2 = _sample_with_sigmas(
            stage2_model,
            seed,
            stage2_cfg,
            stage2_sampler_name,
            stage2_scheduler,
            positive,
            stage2_negative,
            stage1,
            stage2_sigmas,
            disable_noise=not did_upscale,
        )

        return (stage2,)


NODE_CLASS_MAPPINGS = {
    "KreaDualResolutionSelector": KreaDualResolutionSelector,
    "KreaTwoStageSampler": KreaTwoStageSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KreaDualResolutionSelector": "Krea Dual Resolution Selector",
    "KreaTwoStageSampler": "Two-Stage Sampler",
}
