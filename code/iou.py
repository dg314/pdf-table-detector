
from shapely.geometry import Polygon

def calculate_iou(pred, labels):
    xmin1,ymin1,xmax1,ymax1 = pred
    xmin2,ymin2,xmax2,ymax2 = labels

    poly_1 = Polygon([[xmin1, ymax1], [xmax1, ymax1], [xmax1, ymin1], [xmin1, ymin1]])
    poly_2 = Polygon([[xmin2, ymax2], [xmax2, ymax2], [xmax2, ymin2], [xmin2, ymin2]])

    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou