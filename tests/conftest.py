#!/usr/bin/python
import pytest


def pytest_addoption(parser):
    parser.addoption("--cmdopt", action="store", default="test", help="option: 'test' or \
    'generate'. Only use 'generate' if you've changed the tests and need to update the expected \
    output!")


@pytest.fixture
def cmdopt(request):
    return request.config.getoption("--cmdopt")
