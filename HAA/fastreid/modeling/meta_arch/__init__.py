# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import META_ARCH_REGISTRY, build_model


# import all the meta_arch, so they will be registered

from .HAA_MGN import HAA_MGN
from .HAA_BASELINE import HAA_BASELINE