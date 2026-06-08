# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import mmengine
from packaging.version import parse

from .version import __version__, version_info

MMCV_MIN = '2.0.0rc4'
MMCV_MAX = '2.5.0'
MMENGINE_MIN = '0.5.0'
MMENGINE_MAX = '1.0.0'


def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Default: 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])
    else:
        release.extend([0, 0])
    return tuple(release)


# Use packaging.version comparison directly (more robust with packaging>=26.x
# where the custom tuple comparison in digit_version may have edge cases).
from packaging.version import Version  # noqa: E402

mmcv_version = Version(mmcv.__version__)
mmcv_min_version = Version(MMCV_MIN)
mmcv_max_version = Version(MMCV_MAX)

if not (mmcv_min_version <= mmcv_version < mmcv_max_version):
    warnings.warn(
        f'MMCV=={mmcv.__version__} is used but may be incompatible. '
        f'Expected mmcv>={MMCV_MIN}, <{MMCV_MAX}.')

mmengine_version = Version(mmengine.__version__)
mmengine_min_version = Version(MMENGINE_MIN)
mmengine_max_version = Version(MMENGINE_MAX)

if not (mmengine_min_version <= mmengine_version < mmengine_max_version):
    warnings.warn(
        f'MMEngine=={mmengine.__version__} is used but may be incompatible. '
        f'Expected mmengine>={MMENGINE_MIN}, <{MMENGINE_MAX}.')

__all__ = ['__version__', 'version_info', 'digit_version']
