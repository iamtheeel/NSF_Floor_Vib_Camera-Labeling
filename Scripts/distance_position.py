def find_dist_from_y(y_pix_height, resolution = False, debug = False):
    distance_from_cam = 7916.1069 / (y_pix_height + 86.1396) - 1.0263
    if resolution:
        print(f"Resolution innacuracy: {(7916.1069/(y_pix_height+86.1396)**2)*100:.3f}cm/px")
    if debug:
        print(f"{distance_from_cam:.3f}m")
    return distance_from_cam

#use y pixel value as the value
def find_resolution_px_dist(y_pix):
    resolution = (f"{(7916.1069/(y_pix+86.1396)**2)*100:.3f}cm/px")
    return resolution