"""CFA: Counterfactual Data Augmentation for class imbalance.

Implements the algorithm of Temraz, M., & Keane, M. T. (2022), "Solving the
Class Imbalance Problem Using a Counterfactual Method for Data Augmentation",
Machine Learning with Applications 9, 100375.

Typical use:

    >>> from cfa import CFA
    >>> cfa = CFA(max_diff_features=2, tolerance=0.1)   # paper defaults
    >>> X_resampled, y_resampled = cfa.fit_resample(X, y)

For full metadata (synthetic mask, counts, etc.) use ``run_cfa(X, y)`` which
returns a :class:`CFAResult`.
"""

from cfa.core import CFA
from cfa.types import CFAResult

__version__ = "0.2.0"
__all__ = ["CFA", "CFAResult", "__version__"]
