class BaseRegistry:
    def __init__(self, registry_dict: dict, registry_type: str, required=True, wildcard=False) -> None:
        self.registry_dict = registry_dict
        self.registry_type = registry_type
        self.required = required
        self.wildcard = wildcard

    def __call__(self, **config):
        config.pop("id", None)  # only for hydra
        name = config.pop("name")
        if self.wildcard and ("None" in name or name not in self.registry_dict.keys()):
            return self.registry_dict["None"](**config)
        elif self.required and name == "None":
            raise ValueError(f"{name} not found in {self.registry_type}.")
        elif name == "None":
            return None
        return self.registry_dict[name](**config)
