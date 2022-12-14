
def calculate_iou(a, b, epsilon=1e-6):
    # Intersection
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # Overlap
    width = (x2 - x1)
    height = (y2 - y1)

    if (width<0) or (height <0):
        return 0.0

    area_overlap = width * height

    # union
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # return iou
    return area_overlap / (area_combined+epsilon)
