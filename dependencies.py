#!/usr/bin/env python3
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import platform
if not platform.python_version_tuple()[:2] == ('3', '10'):
    raise RuntimeError('Call this script with Python 3.10 (newest version tested in CI)')

# update requirements files using pip-compile
import os
os.system('pip-compile --resolver backtracking -o requirements.txt')
os.system('pip-compile --resolver backtracking --extra ci -o requirements-ci.txt')
os.system('pip-compile --resolver backtracking --extra all --extra ci -o requirements-optional.txt')
