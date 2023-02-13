import re
import sys
from modules import scripts, script_callbacks, ui_extra_networks, extra_networks, shared, sd_models, sd_vae, sd_samplers, processing


operations = {
    "txt2img": processing.StableDiffusionProcessingTxt2Img,
    "img2img": processing.StableDiffusionProcessingImg2Img,
}


def is_debug():
    return shared.opts.data.get("randomizer_keywords_debug", False)


class RandomizerKeywordConfigOption(extra_networks.ExtraNetwork):
    def __init__(self, keyword_name, param_type, value_min=0, value_max=None, option_name=None, validate_cb=None, adjust_cb=None):
        super().__init__(keyword_name)
        self.param_type = param_type
        self.value_min = value_min
        self.value_max = value_max
        self.validate_cb = validate_cb
        self.adjust_cb = adjust_cb

        self.option_name = option_name
        if self.option_name is None:
            self.option_name = keyword_name

        self.has_original = False
        self.original_value = None

    def activate(self, p, params_list):
        if not params_list:
            return

        if not self.has_original:
            self.original_value = shared.opts.data[self.option_name]
            self.has_original = True

        value = params_list[0].items[0]
        value = self.param_type(value)

        if self.adjust_cb:
            value = self.adjust_cb(value, p)

        if isinstance(value, int) or isinstance(value, float):
            if self.value_min:
                value = max(value, self.value_min)
            if self.value_max:
                value = min(value, self.value_max)

        if self.validate_cb:
            error = self.validate_cb(value, p)
            if error:
                raise RuntimeError(f"Validation for '{self.name}' keyword failed: {error}")

        if is_debug():
            print(f"[RandomizerKeywords] Set CONFIG option: {self.option_name} -> {value}")

        shared.opts.data[self.option_name] = value

    def deactivate(self, p):
        if self.has_original:
            if is_debug():
                print(f"[RandomizerKeywords] Reset CONFIG option: {self.option_name} -> {self.original_value}")

            shared.opts.data[self.option_name] = self.original_value
            self.has_original = False
            self.original_value = None


class RandomizerKeywordSamplerParam(extra_networks.ExtraNetwork):
    def __init__(self, param_name, param_type, value_min=0, value_max=None, op_type=None, validate_cb=None, adjust_cb=None):
        super().__init__(param_name)
        self.param_type = param_type
        self.value_min = value_min
        self.value_max = value_max
        self.op_type = op_type
        self.validate_cb = validate_cb
        self.adjust_cb = adjust_cb

    def activate(self, p, params_list):
        if not params_list:
            return

        if self.op_type:
            ty = operations[self.op_type]
            if not isinstance(p, ty):
                return

        value = params_list[0].items[0]
        value = self.param_type(value)

        if self.adjust_cb:
            value = self.adjust_cb(value, p)

        if isinstance(value, int) or isinstance(value, float):
            if self.value_min:
                value = max(value, self.value_min)
            if self.value_max:
                value = min(value, self.value_max)

        if self.validate_cb:
            error = self.validate_cb(value, p)
            if error:
                raise RuntimeError(f"Validation for '{self.name}' keyword failed: {error}")

        if is_debug():
            print(f"[RandomizerKeywords] Set SAMPLER option: {self.name} -> {value}")

        setattr(p, self.name, value)

    def deactivate(self, p):
        pass


def validate_sampler_name(x, p):
    if isinstance(p, processing.StableDiffusionProcessingImg2Img):
        choices = sd_samplers.samplers_for_img2img
    else:
        choices = sd_samplers.samplers

    names = set(x.name for x in choices)

    if x not in names:
        return f"Invalid sampler '{x}'"
    return None


class RandomizerKeywordCheckpoint(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__("checkpoint")
        self.original_checkpoint_info = None

    def activate(self, p, params_list):
        if not params_list:
            return

        if self.original_checkpoint_info is None:
            self.original_checkpoint_info = shared.sd_model.sd_checkpoint_info

        params = params_list[0]
        assert len(params.items) > 0, "Must provide checkpoint name"

        name = params.items[0]
        info = sd_models.get_closet_checkpoint_match(name)
        if info is None:
            raise RuntimeError(f"Unknown checkpoint: {name}")

        if is_debug():
            print(f"[RandomizerKeywords] Set CHECKPOINT: {info.name}")

        sd_models.reload_model_weights(shared.sd_model, info)

    def deactivate(self, p):
        if self.original_checkpoint_info is not None:
            if is_debug():
                print(f"[RandomizerKeywords] Reset CHECKPOINT: {self.original_checkpoint_info.name}")

            sd_models.reload_model_weights(shared.sd_model, self.original_checkpoint_info)
            self.original_checkpoint_info = None


class RandomizerKeywordVAE(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__("vae")
        self.has_original = False
        self.original_vae_info = None

    def find_vae(self, name: str):
        if name.lower() in ['auto', 'automatic']:
            return sd_vae.unspecified
        if name.lower() == 'none':
            return None
        else:
            choices = [x for x in sorted(sd_vae.vae_dict, key=lambda x: len(x)) if name.lower().strip() in x.lower()]
            if len(choices) == 0:
                return None
            else:
                return sd_vae.vae_dict[choices[0]]

    def activate(self, p, params_list):
        if not params_list:
            return

        if not self.has_original:
            self.original_vae_info = shared.opts.sd_vae
            self.has_original = True

        params = params_list[0]
        assert len(params.items) > 0, "Must provide VAE name or 'auto' for automatic"

        name = params.items[0]
        info = self.find_vae(name)
        if info is None:
            raise RuntimeError(f"Unknown VAE: {name}")

        if is_debug():
            print(f"[RandomizerKeywords] Set VAE: {info.name}")

        sd_vae.reload_vae_weights(shared.sd_model, vae_file=info)

    def deactivate(self, p):
        if self.has_original:
            if is_debug():
                print(f"[RandomizerKeywords] Reset VAE: {self.original_vae_info.name}")

            shared.opts.data["sd_vae"] = self.original_vae_info
            sd_vae.reload_vae_weights()

            self.original_checkpoint_info = None
            self.has_original = False


def update_extension_args(ext_name, p, value, arg_idx):
    if isinstance(p, processing.StableDiffusionProcessingImg2Img):
        all_scripts = scripts.scripts_img2img.alwayson_scripts
    else:
        all_scripts = scripts.scripts_txt2img.alwayson_scripts

    script_class = extension_classes[ext_name]
    script = next(iter([s for s in all_scripts if isinstance(s, script_class)]), None)
    assert script, f"Could not find script for {script_class}!"

    args = list(p.script_args)

    if is_debug():
        print(f"[RandomizerKeywords] Args in {ext_name}: {args[script.args_from:script.args_to]}")
        print(f"[RandomizerKeywords] For {ext_name}: Changed arg {arg_idx} from {args[script.args_from + arg_idx]} to {value}")

    args[script.args_from + arg_idx] = value
    p.script_args = tuple(args)


class RandomizerKeywordExtAddNetModel(extra_networks.ExtraNetwork):
    def __init__(self, index):
        super().__init__(f"addnet_model_{index+1}")
        self.index = i

    def activate(self, p, params_list):
        if not params_list:
            return

        model_util = sys.modules.get("scripts.model_util")
        if not model_util:
            raise RuntimeError("Could not load additional_networks model_util")

        value = params_list[0].items[0]
        name = model_util.find_closest_lora_model_name(value)
        if not name:
            raise RuntimeError(f"Could not find LoRA with name {value}")

        update_extension_args("additional_networks", p, True, 0)
        update_extension_args("additional_networks", p, name, 3 + 4 * self.index)  # enabled, separate_weights, (module, {model}, weight_unet, weight_tenc), ...

    def deactivate(self, p):
        pass


class RandomizerKeywordExtAddNetWeight(extra_networks.ExtraNetwork):
    def __init__(self, index, kind=None):
        if kind is None:
            name = f"addnet_weight_{index+1}"
        else:
            name = f"addnet_{kind}_weight_{index+1}"

        super().__init__(name)
        self.index = i
        self.kind = kind

    def activate(self, p, params_list):
        if not params_list:
            return

        model_util = sys.modules.get("scripts.model_util")
        if not model_util:
            raise RuntimeError("Could not load additional_networks model_util")

        value = float(params_list[0].items[0])

        # enabled, separate_weights, (module, model, {weight_unet, weight_tenc}), ...
        update_extension_args("additional_networks", p, True, 0)
        if self.kind is None or self.kind == "unet":
            update_extension_args("additional_networks", p, value, 4 + 4 * self.index)
        if self.kind is None or self.kind == "tenc":
            update_extension_args("additional_networks", p, value, 5 + 4 * self.idnex)

    def deactivate(self, p):
        pass



config_params = [
    RandomizerKeywordConfigOption("clip_skip", int, 1, 12, option_name="CLIP_stop_at_last_layers")
]


# Sampler parameters that can be controlled. They are parameters in the
# Processing class.
sampler_params = [
    RandomizerKeywordSamplerParam("cfg_scale", float, 1),
    RandomizerKeywordSamplerParam("seed", int, -1),
    RandomizerKeywordSamplerParam("subseed", int, -1),
    RandomizerKeywordSamplerParam("subseed_strength", float, 0),
    RandomizerKeywordSamplerParam("sampler_name", str, validate_cb=validate_sampler_name),
    RandomizerKeywordSamplerParam("steps", int, 1),
    RandomizerKeywordSamplerParam("width", int, 64, adjust_cb=lambda x, p: x % 8),
    RandomizerKeywordSamplerParam("height", int, 64, adjust_cb=lambda x, p: x % 8),
    RandomizerKeywordSamplerParam("tiling", bool),
    RandomizerKeywordSamplerParam("restore_faces", bool),
    RandomizerKeywordSamplerParam("s_churn", float),
    RandomizerKeywordSamplerParam("s_tmin", float),
    RandomizerKeywordSamplerParam("s_tmax", float),
    RandomizerKeywordSamplerParam("s_noise", float),
    RandomizerKeywordSamplerParam("eta", float, 0),
    RandomizerKeywordSamplerParam("ddim_discretize", str),
    RandomizerKeywordSamplerParam("denoising_strength", float),

    # txt2img
    RandomizerKeywordSamplerParam("hr_upscaler", str, op_type="txt2img"),
    RandomizerKeywordSamplerParam("hr_scale", float, 0, op_type="txt2img"),
    RandomizerKeywordSamplerParam("hr_second_pass_steps", int, 1, op_type="txt2img"),
    RandomizerKeywordSamplerParam("hr_upscale_to_x", int, 64, adjust_cb=lambda x, p: x % 8, op_type="txt2img"),
    RandomizerKeywordSamplerParam("hr_upscale_to_y", int, 64, adjust_cb=lambda x, p: x % 8, op_type="txt2img"),

    # img2img
    RandomizerKeywordSamplerParam("mask_blur", float, op_type="img2img"),
    RandomizerKeywordSamplerParam("inpainting_mask_weight", float, op_type="img2img"),
]


other_params = [
    RandomizerKeywordCheckpoint(),
    RandomizerKeywordVAE()
]


extension_params = []
extension_modules = {}
extension_classes = {}
supported_modules = {
    "additional_networks": []
}

for i in range(5):
    supported_modules["additional_networks"].extend([
        RandomizerKeywordExtAddNetModel(i),
        RandomizerKeywordExtAddNetWeight(i),
        RandomizerKeywordExtAddNetWeight(i, "unet"),
        RandomizerKeywordExtAddNetWeight(i, "tenc"),
    ])


all_params = []


def on_app_started(demo, app):
    global loaded, all_params
    if loaded:
        return

    for s in scripts.scripts_data:
        for m, params in supported_modules.items():
            if s.module.__name__ == m + ".py":
                assert m not in extension_modules
                print(f"[RandomizerKeywords] Adding support for extension: {m}")
                extension_modules[m] = s.module
                extension_classes[m] = s.script_class
                extension_params.extend(params)

    all_params = config_params + sampler_params + other_params + extension_params
    print(f"[RandomizerKeywords] Supported keywords: {', '.join([p.name for p in all_params])}")

    for param in all_params:
        extra_networks.register_extra_network(param)


def on_ui_settings():
    section = ('randomizer_keywords', "Randomizer Keywords")
    shared.opts.add_option("randomizer_keywords_debug", shared.OptionInfo(False, "Print debug messages", section=section))
    shared.opts.add_option("randomizer_keywords_strip_keywords", shared.OptionInfo(True, "Strip randomizer keywords out of prompts", section=section))


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_settings(on_ui_settings)
