"""Test-suite environment configuration."""

import os

os.environ["ENSIM_DISABLE_3D"] = "1"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
