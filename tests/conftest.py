import subprocess
from pathlib import Path

import pytest

@pytest.fixture(scope='session')
def docker_image_name() -> str:
    return Path(__file__).parents[1].stem.lower()
