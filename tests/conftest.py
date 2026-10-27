"""Test environment setup."""

import os
import time

# AstrBot's background scheduler rejects POSIX TZ values such as ``UTC-8``.
# Keep the test process deterministic and importable in those environments.
os.environ["TZ"] = "UTC"
time.tzset()
