def measure_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox

    foot_x = int((x1 + x2) / 2)
    foot_y = y2

    return (foot_x, foot_y)

def get_closest_keypoint(player_center, court_keypoints, keypoint_indices):
    min_distance = float('inf')
    closest_keypoint = keypoint_indices[0]

    for i in keypoint_indices:
        court_keypoint = (court_keypoints[i * 2], court_keypoints[i * 2 + 1])

        distance = abs(player_center[1] - court_keypoint[1])

        if distance < min_distance:
            min_distance = distance
            closest_keypoint = i

    return closest_keypoint

def get_height_of_bbox(bbox):
    return bbox[3] - bbox[1]

def measure_xy_distance(p1, p2):
    return (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))

def get_center_bbox(bbox):
    x1, y1, x2, y2 = bbox

    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    return (center_x, center_y)

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox

    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    return (center_x, center_y)