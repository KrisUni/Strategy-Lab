"""
Import all indicator specs in explicit, deterministic order.

Order matters for:
1. Registry iteration (optimizer param ordering, UI sidebar ordering)
2. Topological sort (dependencies must be importable before dependents)

Do NOT replace with dynamic discovery (e.g. os.listdir) — that would make
the order OS/filesystem-dependent and break Optuna trial reproducibility.
"""
# Entry group (order mirrors IndicatorSpec.order)
from . import pamrp_entry      # order=1
from . import bbwp_entry       # order=2
from . import adx_entry        # order=3
from . import ma_trend         # order=4
from . import rsi_entry        # order=5
from . import volume_entry     # order=6
from . import supertrend_entry # order=7
from . import vwap             # order=8
from . import macd_entry       # order=9
from . import stoch_rsi_entry  # order=10

# Exit group
from . import pamrp_exit       # order=1
from . import stoch_rsi_exit   # order=2
from . import ma_exit          # order=3
from . import bbwp_exit        # order=4
from . import adx_exit         # order=5
from . import macd_exit        # order=6
from . import rsi_exit         # order=7
from . import supertrend_exit  # order=8
from . import volume_exit      # order=9

# Risk group
from . import stop_loss        # order=1
from . import take_profit      # order=2
from . import trailing_stop    # order=3
from . import atr_trail        # order=4
from . import time_exit        # order=5

# Validate registry integrity at import time
from ..registry import validate_registry
validate_registry()
