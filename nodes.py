# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import torch
import logging
import inspect
import collections
import os

import nodes
import comfy.sd
import comfy.lora
import comfy.float
import comfy.utils
import comfy.model_patcher
import comfy.model_management
import folder_paths

from .ops import GGMLOps, move_patch_to_device
from .loader import gguf_sd_loader, gguf_clip_loader
from .dequant import is_quantized, is_torch_compatible

def update_folder_names_and_paths(key, targets=[]):
    # check for existing key
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    # find base key & add w/ fallback, sanity check + warning
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        logging.warning(f"Unknown file list already present on key {key}: {base}")

# Add a custom keys for files ending in .gguf
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])
update_folder_names_and_paths("clip_gguf", ["text_encoders", "clip"])

def _bake_quantized_weights(patcher, reserve_mib=None):
    if reserve_mib is None:
        reserve_mib = int(os.environ.get("GGUF_BAKE_RESERVE_MIB", "2048"))
    """Dequantise the resident quantised weights once instead of once per forward.

    `GGMLLayer.forward_ggml_cast_weights` calls `cast_bias_weight` -> `dequantize_tensor` on every
    call, so a 4.75 G-parameter DiT re-dequantises its entire weight set on each step. On
    MOSS-Image 512x512 that was measured at 27% of the end-to-end time. Baking swaps the
    quantised weights for fp16 residents and turns `comfy_cast_weights` off: the per-step cost
    goes away and the resident set grows by the fp16-minus-quantised difference (4.4 -> 9.1 GiB).

    Layers are converted one at a time and the loop stops when the free memory reaches
    `reserve_mib`, so a model that cannot hold the whole fp16 set gets a partial bake instead of
    starving the device. The reserve is not a proof of safety: intermediates are allocated after
    this point, which is why the caller also gates on token count.

    Only weights already on the load device are baked, so offloaded layers keep the old behaviour.
    """
    model = patcher.model
    device = patcher.load_device
    targets = []
    for m in model.modules():
        weight = getattr(m, "weight", None)
        if weight is None or not hasattr(m, "is_ggml_quantized") or not m.is_ggml_quantized():
            continue
        if weight.device != device:
            continue
        targets.append(m)
    if not targets:
        return True

    # partially_load() hands weights over through async copies; replacing a tensor underneath a
    # copy still in flight is how a page fault gets raised. Drain first.
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass

    # The text encoder is still resident at the first DiT forward and eats the budget the bake
    # needs (measured: free 5690 MB, both patchers loaded). Its conditioning is already computed,
    # so evict it and keep only the DiT.
    try:
        keep = [lm for lm in comfy.model_management.current_loaded_models if lm.model is patcher]
        want = comfy.model_management.get_free_memory(device) + 2560 * 2**20
        comfy.model_management.free_memory(want, device, keep_loaded=keep)
    except Exception as e:
        logging.warning("GGUF bake could not evict the text encoder: %s", e)
    try:
        loaded = [type(lm.model).__name__ for lm in comfy.model_management.current_loaded_models]
    except Exception:
        loaded = []
    logging.info("GGUF bake start: free=%.0f MB of %d MB reserve, loaded=%s",
                 comfy.model_management.get_free_memory(device) / 2**20, reserve_mib, loaded)
    # Measured after the eviction: the encoder's release must not be counted as bake growth.
    before = torch.cuda.memory_allocated(device)
    baked = 0
    for m in targets:
        if comfy.model_management.get_free_memory(device) < reserve_mib * 2**20:
            logging.warning("GGUF bake stopped at %d/%d layers: under %d MB free",
                            baked, len(targets), reserve_mib)
            break
        m.weight = torch.nn.Parameter(m.get_weight(m.weight.to(device), torch.float16), requires_grad=False)
        m.comfy_cast_weights = False
        if getattr(m, "bias", None) is not None:
            m.bias = torch.nn.Parameter(m.bias.to(device=device, dtype=torch.float16), requires_grad=False)
        baked += 1
    grown = torch.cuda.memory_allocated(device) - before
    model.model_loaded_weight_memory = getattr(model, "model_loaded_weight_memory", 0) + grown
    # Replacing 252 quantised tensors leaves the freed segments scattered through the pool; the
    # display driver then cannot find a contiguous block for a scanout pin even with ~2 GB free
    # (measured 2026-10-18). Return the empty segments to the driver before sampling starts.
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    logging.info("GGUF bake: %d/%d layers resident as fp16, +%.0f MB", baked, len(targets), grown / 2**20)
    return baked > 0


class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False
    # Opt-in, set by UnetLoaderGGUF when dequant_dtype is "bake". See _arm_bake.
    bake_quantized_weights = False
    # Cap on the latent's H*W. The gate is about the sampling intermediates, not the sequence
    # length: 1920x1440 (120*90 = 10800) lost the desktop's framebuffer to a full bake.
    bake_max_latent_pixels = 4096

    def load(self, *args, **kwargs):
        out = super().load(*args, **kwargs)
        self._arm_bake()
        return out

    def partially_load(self, *args, **kwargs):
        # model_load() reaches the weights through partially_load, not load(); hooking only load()
        # silently did nothing. Baking itself waits for the first apply_model, where the text
        # encoder is already evicted and the sequence length is known.
        out = super().partially_load(*args, **kwargs)
        self._arm_bake()
        return out

    def _arm_bake(self):
        if not self.bake_quantized_weights or getattr(self, "_bake_armed", False):
            return
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "apply_model"):
            return
        self._bake_armed = True
        orig = model.apply_model

        def once(x, *a, **kw):
            model.apply_model = orig
            latent_pixels = 0
            try:
                if hasattr(x, "shape") and x.dim() >= 2:
                    latent_pixels = int(x.shape[-2]) * int(x.shape[-1])
            except Exception:
                pass
            if latent_pixels <= self.bake_max_latent_pixels:
                # A bake failure must leave the unbaked path intact: the first step is already
                # running and there is nothing to fall back to once the exception escapes.
                try:
                    _bake_quantized_weights(self)
                except Exception as e:
                    logging.warning("GGUF bake failed, keeping quantised weights: %s: %s",
                                    type(e).__name__, e)
            else:
                logging.info("GGUF bake skipped: latent %d px over the %d limit",
                             latent_pixels, self.bake_max_latent_pixels)
            return orig(x, *a, **kw)

        model.apply_model = once

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)

        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = move_patch_to_device(patches, self.load_device if self.patch_on_device else self.offload_device)
            # TODO: do we ever have legitimate duplicate patches? (i.e. patch on top of patched weight)
            out_weight.patches = [(patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                    weight.to(device=self.offload_device, copy=inplace_update), inplace_update
                )

            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)

            out_weight = comfy.lora.calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, "patches", [])
                if len(patches) > 0:
                    p.patches = []
        # TODO: Find another way to not unload after patches
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)


    def pin_weight_to_device(self, key):
        op_key = key.rsplit('.', 1)[0]
        if not self.mmap_released and op_key in self.named_modules_to_munmap:
            # TODO: possible to OOM, find better way to detach
            self.named_modules_to_munmap[op_key].to(self.load_device).to(self.offload_device)
            del self.named_modules_to_munmap[op_key]
        super().pin_weight_to_device(key)

    mmap_released = False
    named_modules_to_munmap = {}

    def load(self, *args, force_patch_weights=False, **kwargs):
        if not self.mmap_released:
            self.named_modules_to_munmap = dict(self.model.named_modules())

        # always call `patch_weight_to_device` even for lowvram
        super().load(*args, force_patch_weights=True, **kwargs)

        # make sure nothing stays linked to mmap after first load
        if not self.mmap_released:
            linked = []
            if kwargs.get("lowvram_model_memory", 0) > 0:
                for n, m in self.named_modules_to_munmap.items():
                    if hasattr(m, "weight"):
                        device = getattr(m.weight, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, "bias"):
                        device = getattr(m.bias, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked and self.load_device != self.offload_device:
                logging.info(f"Attempting to release mmap ({len(linked)})")
                for n, m in linked:
                    # TODO: possible to OOM, find better way to detach
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True
            self.named_modules_to_munmap = {}

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        # GGUF specific clone values below
        n.patch_on_device = getattr(self, "patch_on_device", False)
        n.mmap_released = getattr(self, "mmap_released", False)
        if src_cls != GGUFModelPatcher:
            n.size = 0 # force recalc
        return n

class UnetLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
        return {
            "required": {
                "unet_name": (unet_names,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "bootleg"
    TITLE = "Unet Loader (GGUF)"

    def load_unet(self, unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None):
        ops = GGMLOps()

        # "bake" is not a dtype: it asks for the weights to be dequantised once and kept resident,
        # which the patcher does on the first apply_model (see _arm_bake).
        bake = dequant_dtype == "bake"
        if bake:
            ops.Linear.dequant_dtype = None
        elif dequant_dtype in ("default", None):
            ops.Linear.dequant_dtype = None
        elif dequant_dtype in ["target"]:
            ops.Linear.dequant_dtype = dequant_dtype
        else:
            ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

        if patch_dtype in ("default", None):
            ops.Linear.patch_dtype = None
        elif patch_dtype in ["target"]:
            ops.Linear.patch_dtype = patch_dtype
        else:
            ops.Linear.patch_dtype = getattr(torch, patch_dtype)

        # init model
        unet_path = folder_paths.get_full_path("unet", unet_name)
        sd, extra = gguf_sd_loader(unet_path)

        kwargs = {}
        valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
        if "metadata" in valid_params:
            kwargs["metadata"] = extra.get("metadata", {})

        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}, **kwargs,
        )
        if model is None:
            logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
            raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))
        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device
        if bake:
            model.bake_quantized_weights = True
        return (model,)

class UnetLoaderGGUFAdvanced(UnetLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
        return {
            "required": {
                "unet_name": (unet_names,),
                "dequant_dtype": (["default", "target", "float32", "float16", "bfloat16", "bake"], {"default": "default"}),
                "patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_on_device": ("BOOLEAN", {"default": False}),
            }
        }
    TITLE = "Unet Loader (GGUF/Advanced)"

class CLIPLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        base = nodes.CLIPLoader.INPUT_TYPES()
        return {
            "required": {
                "clip_name": (s.get_filename_list(),),
                "type": base["required"]["type"],
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "bootleg"
    TITLE = "CLIPLoader (GGUF)"

    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list("clip")
        files += folder_paths.get_filename_list("clip_gguf")
        return sorted(files)

    def load_data(self, ckpt_paths):
        clip_data = []
        for p in ckpt_paths:
            if p.endswith(".gguf"):
                sd = gguf_clip_loader(p)
            else:
                sd = comfy.utils.load_torch_file(p, safe_load=True)
                if "scaled_fp8" in sd: # NOTE: Scaled FP8 would require different custom ops, but only one can be active
                    raise NotImplementedError(f"Mixing scaled FP8 with GGUF is not supported! Use regular CLIP loader or switch model(s)\n({p})")
            clip_data.append(sd)
        return clip_data

    def load_patcher(self, clip_paths, clip_type, clip_data):
        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type = clip_type,
            state_dicts = clip_data,
            model_options = {
                "custom_operations": GGMLOps,
                "initial_device": comfy.model_management.text_encoder_offload_device()
            },
            embedding_directory = folder_paths.get_folder_paths("embeddings"),
        )
        clip.patcher = GGUFModelPatcher.clone(clip.patcher)
        return clip

    def load_clip(self, clip_name, type="stable_diffusion"):
        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher([clip_path], clip_type, self.load_data([clip_path])),)

class DualCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        base = nodes.DualCLIPLoader.INPUT_TYPES()
        file_options = (s.get_filename_list(), )
        return {
            "required": {
                "clip_name1": file_options,
                "clip_name2": file_options,
                "type": base["required"]["type"],
            }
        }

    TITLE = "DualCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, type):
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        clip_paths = (clip_path1, clip_path2)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher(clip_paths, clip_type, self.load_data(clip_paths)),)

class TripleCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = (s.get_filename_list(), )
        return {
            "required": {
                "clip_name1": file_options,
                "clip_name2": file_options,
                "clip_name3": file_options,
            }
        }

    TITLE = "TripleCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, type="sd3"):
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        clip_path3 = folder_paths.get_full_path("clip", clip_name3)
        clip_paths = (clip_path1, clip_path2, clip_path3)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher(clip_paths, clip_type, self.load_data(clip_paths)),)

class QuadrupleCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = (s.get_filename_list(), )
        return {
            "required": {
            "clip_name1": file_options,
            "clip_name2": file_options,
            "clip_name3": file_options,
            "clip_name4": file_options,
        }
    }

    TITLE = "QuadrupleCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, clip_name4, type="stable_diffusion"):
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        clip_path3 = folder_paths.get_full_path("clip", clip_name3)
        clip_path4 = folder_paths.get_full_path("clip", clip_name4)
        clip_paths = (clip_path1, clip_path2, clip_path3, clip_path4)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher(clip_paths, clip_type, self.load_data(clip_paths)),)

NODE_CLASS_MAPPINGS = {
    "UnetLoaderGGUF": UnetLoaderGGUF,
    "CLIPLoaderGGUF": CLIPLoaderGGUF,
    "DualCLIPLoaderGGUF": DualCLIPLoaderGGUF,
    "TripleCLIPLoaderGGUF": TripleCLIPLoaderGGUF,
    "QuadrupleCLIPLoaderGGUF": QuadrupleCLIPLoaderGGUF,
    "UnetLoaderGGUFAdvanced": UnetLoaderGGUFAdvanced,
}

