from modules import script_callbacks, ui_extra_networks, extra_networks, shared, sd_models


class RandomizerKeywordCheckpoint(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__("checkpoint")
        self.original_checkpoint_info = None

    def activate(self, p, params_list):
        if self.original_checkpoint_info is None:
            self.original_checkpoint_info = shared.sd_model.sd_checkpoint_info

        params = params_list[0]
        assert len(params.items) > 0, "Must provide checkpoint name"

        name = params.items[0]
        info = sd_models.get_closet_checkpoint_match(name)
        if info is None:
            raise RuntimeError(f"Unknown checkpoint: {name}")

        sd_models.reload_model_weights(shared.sd_model, info)

    def deactivate(self, p):
        if self.original_checkpoint_info is not None:
            sd_models.reload_model_weights(shared.sd_model, self.original_checkpoint_info)
            self.original_checkpoint_info = None


def before_ui():
    extra_networks.register_extra_network(RandomizerKeywordCheckpoint())


script_callbacks.on_before_ui(before_ui)
