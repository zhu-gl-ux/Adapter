PASCAL_DIR = "/you/datapath"
ADE_DIR = "/you/datapath"
DATASETS_IMG_DIRS = {"voc": PASCAL_DIR, "ade": ADE_DIR}

VOC = ['background',
       'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
       'bus', 'car', 'cat', 'chair', 'cow',
       'diningtable', 'dog', 'horse', 'motorbike', 'person',
       'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
PASCAL_NUM_CLASSES = len(VOC)

ADE = ["void",
       "wall", "building", "sky", "floor", "tree",
       "ceiling", "road", "bed ", "windowpane", "grass",
       "cabinet", "sidewalk", "person", "earth", "door",
       "table", "mountain", "plant", "curtain", "chair",
       "car", "water", "painting", "sofa", "shelf",
       "house", "sea", "mirror", "rug", "field",
       "armchair", "seat", "fence", "desk", "rock",
       "wardrobe", "lamp", "bathtub", "railing", "cushion",
       "base", "box", "column", "signboard", "chest of drawers",
       "counter", "sand", "sink", "skyscraper", "fireplace",
       "refrigerator", "grandstand", "path", "stairs", "runway",
       "case", "pool table", "pillow", "screen door", "stairway",
       "river", "bridge", "bookcase", "blind", "coffee table",
       "toilet", "flower", "book", "hill", "bench",
       "countertop", "stove", "palm", "kitchen island", "computer",
       "swivel chair", "boat", "bar", "arcade machine", "hovel",
       "bus", "towel", "light", "truck", "tower",
       "chandelier", "awning", "streetlight", "booth", "television receiver",
       "airplane", "dirt track", "apparel", "pole", "land",
       "bannister", "escalator", "ottoman", "bottle", "buffet",
       "poster", "stage", "van", "ship", "fountain",
       "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
       "stool", "barrel", "basket", "waterfall", "tent",
       "bag", "minibike", "cradle", "oven", "ball",
       "food", "step", "tank", "trade name", "microwave",
       "pot", "animal", "bicycle", "lake", "dishwasher",
       "screen", "blanket", "sculpture", "hood", "sconce",
       "vase", "traffic light", "tray", "ashcan", "fan",
       "pier", "crt screen", "plate", "monitor", "bulletin board",
       "shower", "radiator", "glass", "clock", "flag"]
ADE_NUM_CLASSES = len(ADE)
