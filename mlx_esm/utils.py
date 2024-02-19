import sys
from typing import Optional


# Copy-pasted from tqdm.auto
def is_notebook() -> bool:
  try:
    get_ipython = sys.modules["IPython"].get_ipython
    if "IPKernelApp" not in get_ipython().config:
      raise KeyError("IPKernelApp")
    return True
  except KeyError:
    return False


# This hackery is needed because without ncols the CLI version becomes shit, but with it
# the notebook version becomes shit. Such is life.
def tqdm_ncols() -> Optional[int]:
  return None if is_notebook() else 120
