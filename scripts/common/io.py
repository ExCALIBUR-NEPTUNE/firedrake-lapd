import os.path
import pprint
import yaml
from firedrake import PETSc


def read_yaml_config(fname, process_derived=None, verbose=True):
    root_dir = os.path.dirname(os.path.dirname(__file__))
    fpath = os.path.join(root_dir, fname)
    with open(fpath) as fh:
        cfg = yaml.load(fh, Loader=yaml.FullLoader)
    cfg["root_dir"] = root_dir

    if process_derived is not None:
        process_derived(cfg)

    if verbose:
        # Pretty-print options
        pp_str = pprint.pformat(cfg, depth=2)
        PETSc.Sys.Print(f"Options read from {fpath}: ")
        PETSc.Sys.Print(pp_str)

    return cfg


def set_default_param(cfg, key, default):
    if not key in cfg:
        cfg[key] = default
