"""
ENVIRONMENT LOADER - Resolve Import Conflicts

Purpose:
  Load the installed 'mobile-env' Gymnasium package while avoiding conflicts
  with the local 'mobile_env_local' directory shadow.
  
Problem:
  Python prefers local directories over site-packages, so importing 'mobile_env'
  would try to load the local directory first, causing missing module errors.
  
Solution:
  Explicitly find and load from site-packages using importlib.
"""

import importlib.util
import sys
import sysconfig
from pathlib import Path


def load_installed_mobile_env_package(package_name="mobile_env"):
    """Load the installed mobile_env package from the current Python environment.

    This is needed because the local workspace also contains a package named
    `mobile_env` that would otherwise shadow the installed package.
    """
    site_packages = Path(sysconfig.get_paths()["purelib"])
    package_path = site_packages / package_name
    init_file = package_path / "__init__.py"
    if not init_file.exists():
        raise ImportError(f"Installed package '{package_name}' not found in {site_packages}")

    spec = importlib.util.spec_from_file_location("installed_mobile_env", str(init_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load installed package from {init_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["installed_mobile_env"] = module
    spec.loader.exec_module(module)
    return module
