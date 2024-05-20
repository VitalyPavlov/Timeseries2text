import yaml
from fire import Fire
from addict import Dict
import os


def update_config(config, params):
    for k, v in params.items():
        # *path, key = k.split(".")
        config.update({k: v})
        # print(f"Overwriting {k} = {v} (was {config.get(key)})")
    return config


def model_cfg(**kwargs):
    with open("./config/default.yaml") as cfg:
        base_config = yaml.load(cfg, Loader=yaml.FullLoader)

    if "config" in kwargs.keys():
        cfg_path = kwargs["config"]
        with open(f"./config/experiments/{cfg_path}") as cfg:
            cfg_yaml = yaml.load(cfg, Loader=yaml.FullLoader)

        merged_cfg = update_config(base_config, cfg_yaml)
    else:
        merged_cfg = base_config

    update_cfg = update_config(merged_cfg, kwargs)
    return update_cfg


if __name__ == "__main__":
    cfg = Dict(Fire(model_cfg))
