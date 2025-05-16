#!/usr/bin/env python3
# -*- coding:utf-8 -*-

try:
    from .utils import configure_module

    configure_module()
except:

    print('Warning :Failed to configure YoloX module. But I wish it will not affect the Tracker.')


__version__ = "0.1.0"
