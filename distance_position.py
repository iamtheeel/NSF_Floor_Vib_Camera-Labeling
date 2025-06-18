def find_dist_from_y(y_pix_height, resolution = False, debug = False):
    distance_from_cam = 7916.1069 / (y_pix_height + 86.1396) - 1.0263
    if resolution:
        print(f"Resolution innacuracy: +/-{(7916.1069/(distance_from_cam+86.1396)**2)/2:.3f}px")
    if debug:
        print(f"{distance_from_cam:.3f}m")
    return distance_from_cam

def find_resolution_px_dist(y_distance_processed):
    resolution = (f"+/-{(7916.1069/(y_distance_processed+86.1396)**2)*100:.3f}cm/px")
    return resolution