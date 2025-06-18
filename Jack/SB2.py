import sys
import os
sys.path.append(os.path.abspath('..'))

from distance_position import find_dist_from_y

bingo = find_dist_from_y(500, debug = True)
print(bingo)