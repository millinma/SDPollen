from helpers import BaseRegistry
from .bootstrapping import Bootstrapping

CURRICULUM_SCORE_REGISTRY = BaseRegistry({
    "Bootstrapping": Bootstrapping
}, "CURRICULUM_SCORE_REGISTRY", required=False)
