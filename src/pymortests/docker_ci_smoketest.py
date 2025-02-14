# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)
import os
import warnings

import pytest

from pymor.core.config import _PACKAGES, config


@pytest.mark.parametrize('pkg', _PACKAGES.keys())
@pytest.mark.skipif(condition=not os.environ.get('DOCKER_PYMOR', False),
                    reason='Guarantee only valid in the docker container')
def test_config(pkg):
    assert getattr(config, f'HAVE_{pkg}') or pkg in config.disabled


@pytest.mark.skipif(condition=not os.environ.get('DOCKER_PYMOR', False),
                    reason='Guarantee only valid in the docker container')
def test_no_dune_warnings():
    if not config.HAVE_DUNEGDT:
        pytest.skip('dune-gdt not enalbed')
    _test_dune_import_warn()


@pytest.mark.skipif(condition=not os.environ.get('DOCKER_PYMOR', False),
                    reason='Guarantee only valid in the docker container')
def test_dune_warnings(monkeypatch):
    if not config.HAVE_DUNEGDT:
        pytest.skip('dune-gdt not enalbed')
    from dune import gdt, xt
    monkeypatch.setattr(gdt, '__version__', '2020.0.0')
    monkeypatch.setattr(xt, '__version__', '2020.0.0')
    with pytest.xfail(''):
        _test_dune_import_warn()


def _test_dune_import_warn():
    with warnings.catch_warnings():
        from pymor.core.config import _get_dunegdt_version

        # this will result in an error if a warning is caught
        warnings.simplefilter('error')
        _get_dunegdt_version()
