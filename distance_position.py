def find_dist_from_y(y_pix_height, debug = False):
    distance_from_cam = 7916.1069 / (y_pix_height + 86.1396) - 1.0263
    if debug:
        print(f"{distance_from_cam:.3f}m")
    return distance_from_cam