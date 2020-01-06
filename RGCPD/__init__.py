# -*- coding: utf-8 -*-
"""Documentation about RGCPD"""
import os, inspect, sys
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
if curr_dir not in sys.path:
    sys.path.append(curr_dir)
    
from class_RGCPD import RGCPD
from func_fc import fcev


__version__ = '0.1'

__author__ = 'Sem Vijverberg '
__email__ = 'sem.vijverberg@vu.nl'

