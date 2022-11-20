"""
swtloc
Stroke Width Transform Localization Library
"""

# Author : Achintya Gupta
# Purpose : Imports
try:
    from .swtlocalizer import SWTLocalizer
except ImportError:
    from swtlocalizer import SWTLocalizer

__version__ = "2.1.1"

