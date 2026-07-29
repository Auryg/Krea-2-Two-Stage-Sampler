import math
import random

import torch

import comfy.model_management
import comfy.model_sampling
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


RANDOM_ASPECT_RATIO_MODES = {
    "Random": tuple(FIXED_ASPECT_RATIOS),
    "Random Vertical": tuple(
        name for name, (width, height) in FIXED_ASPECT_RATIOS.items() if height > width
    ),
    "Random Horizontal": tuple(
        name for name, (width, height) in FIXED_ASPECT_RATIOS.items() if width > height
    ),
    "Random Constrained": ("1:1", "4:5", "5:4", "2:3", "3:2", "3:4", "4:3"),
}


ASPECT_RATIOS = {**dict.fromkeys(RANDOM_ASPECT_RATIO_MODES), **FIXED_ASPECT_RATIOS}
UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]


def _round_to_multiple(value, multiple):
    if multiple <= 1:
        return int(round(value))
    return int(round(value / multiple) * multiple)


def _resolve_aspect_ratio(aspect_ratio, random_seed):
    if aspect_ratio not in RANDOM_ASPECT_RATIO_MODES:
        return aspect_ratio

    aspect_names = RANDOM_ASPECT_RATIO_MODES[aspect_ratio]
    return aspect_names[random.Random(random_seed).randrange(len(aspect_names))]


def _dimensions_for(aspect_ratio, megapixels, multiple, random_seed=0):
    selected_aspect_ratio = _resolve_aspect_ratio(aspect_ratio, random_seed)
    ratio_w, ratio_h = FIXED_ASPECT_RATIOS[selected_aspect_ratio]
    total_pixels = megapixels * 1024 * 1024
    scale = math.sqrt(total_pixels / (ratio_w * ratio_h))
    width = max(multiple, _round_to_multiple(ratio_w * scale, multiple))
    height = max(multiple, _round_to_multiple(ratio_h * scale, multiple))
    return width, height


def _krea2_raw_shift(width, height):
    min_tokens = 256
    max_tokens = 6400
    min_shift = 0.5
    max_shift = 1.15
    sampling_width = math.ceil(width / 16) * 16
    sampling_height = math.ceil(height / 16) * 16
    image_tokens = sampling_width * sampling_height / (8 * 8 * 2 * 2)
    slope = (max_shift - min_shift) / (max_tokens - min_tokens)
    return slope * image_tokens + (min_shift - slope * min_tokens)


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


def _percent_schedule_index(steps, sigmas, handoff_percent, stage_name):
    max_index = len(sigmas) - 2
    if max_index < 1:
        raise ValueError(
            f"{stage_name} must produce at least one non-terminal handoff sigma."
        )
    requested = round(steps * handoff_percent / 100.0)
    return _clamp(requested, 1, max_index)


def _nearest_schedule_index(
    sigmas,
    boundary_sigma,
    minimum_index,
    stage_name,
):
    max_index = len(sigmas) - 2
    if max_index < minimum_index:
        raise ValueError(
            f"{stage_name} does not have enough non-terminal sigmas for this handoff."
        )

    boundary = float(boundary_sigma)
    if boundary <= 0.0:
        raise ValueError("The handoff sigma must be greater than zero.")
    if boundary >= float(sigmas[0]):
        raise ValueError(
            f"The handoff sigma is outside the {stage_name} schedule. "
            "Use compatible model sampling settings or move the handoff later."
        )

    candidates = [
        idx
        for idx in range(minimum_index, max_index + 1)
        if float(sigmas[idx - 1]) > boundary
    ]
    if not candidates:
        raise ValueError(
            f"Could not find a valid {stage_name} handoff step. "
            "Use compatible model sampling settings or move the handoffs farther apart."
        )

    return min(candidates, key=lambda idx: abs(float(sigmas[idx]) - boundary))


def _build_sigma_pair(stage1_model, stage2_model, stage1_steps, stage2_steps, stage1_sampler, stage1_scheduler, stage2_sampler, stage2_scheduler, handoff_percent):
    stage1_sigmas = _sigma_schedule(stage1_model, stage1_steps, stage1_sampler, stage1_scheduler)
    stage2_sigmas = _sigma_schedule(stage2_model, stage2_steps, stage2_sampler, stage2_scheduler)

    stage1_end = _percent_schedule_index(
        stage1_steps,
        stage1_sigmas,
        handoff_percent,
        "stage 1",
    )
    boundary_sigma = stage1_sigmas[stage1_end].clone()
    stage2_start = _nearest_schedule_index(
        stage2_sigmas,
        boundary_sigma,
        1,
        "stage 2",
    )

    stage1_custom = stage1_sigmas[:stage1_end + 1].clone()
    stage2_custom = stage2_sigmas[stage2_start:].clone()
    stage2_custom[0] = boundary_sigma

    return stage1_custom, stage2_custom, stage1_end, stage2_start, float(boundary_sigma)


def _build_sigma_triplet(
    stage1_model,
    stage2_model,
    stage1_steps,
    stage2_steps,
    stage1_sampler,
    stage1_scheduler,
    stage2_sampler,
    stage2_scheduler,
    handoff_percent,
    stage3_handoff_percent,
):
    stage1_full = _sigma_schedule(
        stage1_model,
        stage1_steps,
        stage1_sampler,
        stage1_scheduler,
    )
    stage2_full = _sigma_schedule(
        stage2_model,
        stage2_steps,
        stage2_sampler,
        stage2_scheduler,
    )

    stage3_start = _percent_schedule_index(
        stage1_steps,
        stage1_full,
        stage3_handoff_percent,
        "stage 1",
    )
    stage2_end_sigma = stage1_full[stage3_start].clone()

    stage1_custom = None
    stage1_end = None
    stage1_end_sigma = None
    if handoff_percent <= 0.0:
        stage2_start = 0
    else:
        stage1_end = _percent_schedule_index(
            stage1_steps,
            stage1_full,
            handoff_percent,
            "stage 1",
        )
        stage1_end_sigma = stage1_full[stage1_end].clone()
        stage1_custom = stage1_full[:stage1_end + 1].clone()
        stage2_start = _nearest_schedule_index(
            stage2_full,
            stage1_end_sigma,
            1,
            "stage 2",
        )

        if float(stage2_end_sigma) >= float(stage1_end_sigma):
            raise ValueError(
                "stage3_handoff_percent must produce a later denoising sigma than "
                "handoff_percent. Move the handoffs farther apart or use compatible "
                "model sampling settings."
            )

    stage2_end = _nearest_schedule_index(
        stage2_full,
        stage2_end_sigma,
        stage2_start + 1,
        "stage 2",
    )
    stage2_custom = torch.cat(
        (stage2_full[stage2_start:stage2_end], stage2_end_sigma.reshape(1))
    )
    if stage1_end_sigma is not None:
        stage2_custom[0] = stage1_end_sigma

    stage3_custom = stage1_full[stage3_start:].clone()
    stage3_custom[0] = stage2_end_sigma

    return (
        stage1_custom,
        stage2_custom,
        stage3_custom,
        stage1_end,
        stage2_start,
        stage2_end,
        stage3_start,
        None if stage1_end_sigma is None else float(stage1_end_sigma),
        float(stage2_end_sigma),
    )


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

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("base_width", "base_height", "final_width", "final_height", "seed")
    FUNCTION = "execute"
    CATEGORY = "Ashen3"

    def execute(self, aspect_ratio, base_megapixels, final_megapixels, multiple, random_seed=0):
        base_width, base_height = _dimensions_for(aspect_ratio, base_megapixels, multiple, random_seed)
        final_width, final_height = _dimensions_for(aspect_ratio, final_megapixels, multiple, random_seed)
        return (base_width, base_height, final_width, final_height, random_seed)


class Krea2ModelSampling:
    MODES = ["raw_dynamic", "turbo_fixed", "manual"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sampling_mode": (
                    cls.MODES,
                    {
                        "default": "raw_dynamic",
                        "tooltip": (
                            "Raw uses Krea 2's official resolution-dependent shift. "
                            "Turbo pins the shift to 1.15. Manual uses manual_shift."
                        ),
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": nodes.MAX_RESOLUTION,
                        "step": 8,
                        "tooltip": "Sampling width; used only by raw_dynamic.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": nodes.MAX_RESOLUTION,
                        "step": 8,
                        "tooltip": "Sampling height; used only by raw_dynamic.",
                    },
                ),
                "manual_shift": (
                    "FLOAT",
                    {
                        "default": 1.15,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": False,
                        "advanced": True,
                        "tooltip": "Constant shift used only by manual mode.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Ashen3"

    def patch(self, model, sampling_mode, width, height, manual_shift=1.15):
        if sampling_mode == "raw_dynamic":
            shift = _krea2_raw_shift(width, height)
        elif sampling_mode == "turbo_fixed":
            shift = 1.15
        elif sampling_mode == "manual":
            shift = manual_shift
        else:
            raise ValueError(f"Unknown Krea 2 sampling mode: {sampling_mode}")

        class ModelSamplingKrea2(
            comfy.model_sampling.ModelSamplingFlux,
            comfy.model_sampling.CONST,
        ):
            pass

        model_sampling = ModelSamplingKrea2(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        patched_model = model.clone()
        patched_model.add_object_patch("model_sampling", model_sampling)

        print(
            "Krea 2 Model Sampling: "
            f"mode={sampling_mode}, shift={shift:.8f}, width={width}, height={height}"
        )
        return (patched_model,)


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
                "handoff_percent": (
                    "FLOAT",
                    {
                        "default": 16.67,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": 0.01,
                        "tooltip": (
                            "Point in the denoising process where stage 1 hands off to stage 2. "
                            "0% uses only stage 2; 100% uses only stage 1."
                        ),
                    },
                ),
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
        if handoff_percent <= 0.0:
            stage2_sigmas = _sigma_schedule(
                stage2_model,
                stage2_steps,
                stage2_sampler_name,
                stage2_scheduler,
            )
            stage2_input, did_upscale = _upscale_latent_if_needed(
                latent_image,
                stage2_model,
                final_width,
                final_height,
                upscale_method,
            )
            stage2_negative = (
                _zero_out_conditioning(negative)
                if math.isclose(stage2_cfg, 1.0, rel_tol=0.0, abs_tol=1e-6)
                else negative
            )
            print(
                "Krea Two-Stage Sampler: "
                f"stage2_only=true, stage2_steps={len(stage2_sigmas) - 1}, "
                f"resized_input={str(did_upscale).lower()}"
            )
            stage2 = _sample_with_sigmas(
                stage2_model,
                seed,
                stage2_cfg,
                stage2_sampler_name,
                stage2_scheduler,
                positive,
                stage2_negative,
                stage2_input,
                stage2_sigmas,
                disable_noise=False,
            )
            return (stage2,)

        if handoff_percent >= 100.0:
            stage1_sigmas = _sigma_schedule(
                stage1_model,
                stage1_steps,
                stage1_sampler_name,
                stage1_scheduler,
            )
            print(
                "Krea Two-Stage Sampler: "
                f"stage1_only=true, stage1_steps={len(stage1_sigmas) - 1}"
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
                stage1_sigmas,
                disable_noise=False,
            )
            return (stage1,)

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


class KreaThreeStageSampler(KreaTwoStageSampler):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        required = {}
        for name, definition in inputs["required"].items():
            required[name] = definition
            if name == "handoff_percent":
                required["stage3_handoff_percent"] = (
                    "FLOAT",
                    {
                        "default": 83.33,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": 0.01,
                        "tooltip": (
                            "Point in the denoising process where stage 2 hands off "
                            "to stage 3. Stage 3 reuses all stage 1 settings. This "
                            "must be greater than or equal to handoff_percent."
                        ),
                    },
                )
        inputs["required"] = required
        return inputs

    def sample(
        self,
        stage1_model,
        stage2_model,
        positive,
        negative,
        latent_image,
        seed,
        handoff_percent,
        stage3_handoff_percent,
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
        if stage3_handoff_percent < handoff_percent:
            raise ValueError(
                "stage3_handoff_percent must be greater than or equal to "
                "handoff_percent."
            )

        if stage3_handoff_percent >= 100.0:
            return super().sample(
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
                upscale_method,
            )

        stage1_negative = negative

        if math.isclose(
            stage3_handoff_percent,
            handoff_percent,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            if stage3_handoff_percent <= 0.0:
                stage3_sigmas = _sigma_schedule(
                    stage1_model,
                    stage1_steps,
                    stage1_sampler_name,
                    stage1_scheduler,
                )
                stage3_input, did_upscale = _upscale_latent_if_needed(
                    latent_image,
                    stage1_model,
                    final_width,
                    final_height,
                    upscale_method,
                )
                print(
                    "Krea Three-Stage Sampler: "
                    f"stage3_only=true, stage3_steps={len(stage3_sigmas) - 1}, "
                    f"resized_input={str(did_upscale).lower()}"
                )
                stage3 = _sample_with_sigmas(
                    stage1_model,
                    seed,
                    stage1_cfg,
                    stage1_sampler_name,
                    stage1_scheduler,
                    positive,
                    stage1_negative,
                    stage3_input,
                    stage3_sigmas,
                    disable_noise=False,
                )
                return (stage3,)

            stage1_full = _sigma_schedule(
                stage1_model,
                stage1_steps,
                stage1_sampler_name,
                stage1_scheduler,
            )
            stage3_start = _percent_schedule_index(
                stage1_steps,
                stage1_full,
                stage3_handoff_percent,
                "stage 1",
            )
            boundary_sigma = stage1_full[stage3_start].clone()
            stage1_sigmas = stage1_full[:stage3_start + 1].clone()
            stage3_sigmas = stage1_full[stage3_start:].clone()
            upscale_requested = _will_upscale_latent(
                latent_image,
                stage1_model,
                final_width,
                final_height,
            )
            stage1_run_sigmas = stage1_sigmas.clone()
            if upscale_requested:
                stage1_run_sigmas[-1] = 0.0

            print(
                "Krea Three-Stage Sampler: "
                f"stage2_skipped=true, stage1_end={stage3_start}, "
                f"stage3_start={stage3_start}, boundary_sigma={float(boundary_sigma):.8f}, "
                f"noise_mode={'fresh_high_res' if upscale_requested else 'carry_leftover'}"
            )
            stage1 = _sample_with_sigmas(
                stage1_model,
                seed,
                stage1_cfg,
                stage1_sampler_name,
                stage1_scheduler,
                positive,
                stage1_negative,
                latent_image,
                stage1_run_sigmas,
                disable_noise=False,
            )
            stage3_input, did_upscale = _upscale_latent_if_needed(
                stage1,
                stage1_model,
                final_width,
                final_height,
                upscale_method,
            )
            stage3 = _sample_with_sigmas(
                stage1_model,
                seed,
                stage1_cfg,
                stage1_sampler_name,
                stage1_scheduler,
                positive,
                stage1_negative,
                stage3_input,
                stage3_sigmas,
                disable_noise=not did_upscale,
            )
            return (stage3,)

        (
            stage1_sigmas,
            stage2_sigmas,
            stage3_sigmas,
            stage1_end,
            stage2_start,
            stage2_end,
            stage3_start,
            stage1_boundary_sigma,
            stage2_boundary_sigma,
        ) = _build_sigma_triplet(
            stage1_model,
            stage2_model,
            stage1_steps,
            stage2_steps,
            stage1_sampler_name,
            stage1_scheduler,
            stage2_sampler_name,
            stage2_scheduler,
            handoff_percent,
            stage3_handoff_percent,
        )

        if stage1_sigmas is None:
            stage2_input, did_upscale = _upscale_latent_if_needed(
                latent_image,
                stage2_model,
                final_width,
                final_height,
                upscale_method,
            )
            stage2_disable_noise = False
            stage1_log = "stage1_skipped=true"
        else:
            upscale_requested = _will_upscale_latent(
                latent_image,
                stage2_model,
                final_width,
                final_height,
            )
            stage1_run_sigmas = stage1_sigmas.clone()
            if upscale_requested:
                stage1_run_sigmas[-1] = 0.0
            stage1 = _sample_with_sigmas(
                stage1_model,
                seed,
                stage1_cfg,
                stage1_sampler_name,
                stage1_scheduler,
                positive,
                stage1_negative,
                latent_image,
                stage1_run_sigmas,
                disable_noise=False,
            )
            stage2_input, did_upscale = _upscale_latent_if_needed(
                stage1,
                stage2_model,
                final_width,
                final_height,
                upscale_method,
            )
            stage2_disable_noise = not did_upscale
            stage1_log = (
                f"stage1_end={stage1_end}, "
                f"stage1_boundary_sigma={stage1_boundary_sigma:.8f}"
            )

        stage2_negative = (
            _zero_out_conditioning(negative)
            if math.isclose(stage2_cfg, 1.0, rel_tol=0.0, abs_tol=1e-6)
            else negative
        )
        print(
            "Krea Three-Stage Sampler: "
            f"{stage1_log}, stage2_start={stage2_start}, stage2_end={stage2_end}, "
            f"stage3_start={stage3_start}, "
            f"stage2_boundary_sigma={stage2_boundary_sigma:.8f}, "
            f"noise_mode={'fresh_high_res' if not stage2_disable_noise else 'carry_leftover'}"
        )
        stage2 = _sample_with_sigmas(
            stage2_model,
            seed,
            stage2_cfg,
            stage2_sampler_name,
            stage2_scheduler,
            positive,
            stage2_negative,
            stage2_input,
            stage2_sigmas,
            disable_noise=stage2_disable_noise,
        )
        stage3 = _sample_with_sigmas(
            stage1_model,
            seed,
            stage1_cfg,
            stage1_sampler_name,
            stage1_scheduler,
            positive,
            stage1_negative,
            stage2,
            stage3_sigmas,
            disable_noise=True,
        )
        return (stage3,)


NODE_CLASS_MAPPINGS = {
    "KreaDualResolutionSelector": KreaDualResolutionSelector,
    "Krea2ModelSampling": Krea2ModelSampling,
    "KreaTwoStageSampler": KreaTwoStageSampler,
    "KreaThreeStageSampler": KreaThreeStageSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KreaDualResolutionSelector": "Krea Dual Resolution Selector",
    "Krea2ModelSampling": "Krea 2 Model Sampling",
    "KreaTwoStageSampler": "Two-Stage Sampler",
    "KreaThreeStageSampler": "Three-Stage Sampler",
}
