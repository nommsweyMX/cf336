"""Path setup: this piece reuses the DSP toolkit and mastering chain written for "What You Repeat"
(songs/why-llarta/production) and the drum and effect sounds written for "La Loca Tlacuaque"
(songs/la-loca-tlacuaque/production). Import this module first."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.normpath(os.path.join(HERE, "..", "..", "why-llarta", "production"))
TLAC = os.path.normpath(os.path.join(HERE, "..", "..", "la-loca-tlacuaque", "production"))
if SHARED not in sys.path:
    sys.path.append(SHARED)


def load(name, path):
    """Import a module from an explicit file path under a private name (no sys.path changes)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def tlac_sounds():
    """The drum kit, 808, piano, accordion and effects from the possum song."""
    return load("tlac_sounds", os.path.join(TLAC, "sounds.py"))


def master_chain():
    """The mix and mastering functions from "What You Repeat" (needs this piece's score imported first)."""
    return load("wl_mix", os.path.join(SHARED, "mix.py"))
