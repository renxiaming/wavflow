from wavtts.model.backbones.dit import DiT
from wavtts.model.backbones.mmdit import MMDiT
from wavtts.model.cfm import CFM


__all__ = ["CFM", "DiT", "MMDiT", "Trainer"]


def __getattr__(name):
    if name == "Trainer":
        from wavtts.model.trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
