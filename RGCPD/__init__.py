# -*- coding: utf-8 -*-
"""Documentation about RGCPD"""
import sys, os
main_dir = './RGCPD'
# curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
# main_dir = '/'.join(curr_dir.split('/')[:-1])
sys.path.append(main_dir)
# sys.path.append('./forecasting')
# subdates_dir = os.path.join(main_dir, 'RGCPD/')
# fc_dir = os.path.join(main_dir, 'forecasting/')

# if main_dir not in sys.path:
#     sys.path.append(main_dir)
    # sys.path.append(subdates_dir)
    # sys.path.append(fc_dir)

from class_RGCPD import RGCPD
# from func_fc import fcev
from class_EOF import EOF
from class_BivariateMI import BivariateMI
from class_RV import RV_class




__version__ = '0.1'

__author__ = 'Sem Vijverberg '
__email__ = 'sem.vijverberg@vu.nl'

