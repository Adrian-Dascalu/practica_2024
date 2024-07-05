def convert_pixel_distance_to_meters(pixel_distance, reference_height_meters, reference_height_pixels):
    return (pixel_distance * reference_height_meters) / reference_height_pixels

def convert_meters_distance_to_pixels(meters_distance, reference_height_meters, reference_height_pixels):
    return (meters_distance * reference_height_pixels) / reference_height_meters