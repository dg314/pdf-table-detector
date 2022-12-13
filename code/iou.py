
# from shapely.geometry import Polygon

# def calculate_iou(pred, labels):
#     xmin1,ymin1,xmax1,ymax1 = pred
#     xmin2,ymin2,xmax2,ymax2 = labels

#     poly_1 = Polygon([[xmin1, ymax1], [xmax1, ymax1], [xmax1, ymin1], [xmin1, ymin1]])
#     poly_2 = Polygon([[xmin2, ymax2], [xmax2, ymax2], [xmax2, ymin2], [xmin2, ymin2]])

#     iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
#     return iou


def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w*h


def calculate_iou(a, b):
    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)