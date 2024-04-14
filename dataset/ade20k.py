# adapted from https://github.com/paperswithcode/torchbench/blob/master/torchbench/datasets/ade20k.py
import os
from training.utils import download_extract
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from training.utils import download_extract

ARCHIVE_DICT = {
    "trainval": {
        "url": (
            "http://data.csail.mit.edu/places/ADEchallenge/" "ADEChallengeData2016.zip"
        ),
        "md5": "7328b3957e407ddae1d3cbf487f149ef",
        "base_dir": "ADEChallengeData2016",
    }
}


class ADE20K(VisionDataset):
    """`ADE20K Dataset.

    ADE20K <https://groups.csail.mit.edu/vision/datasets/ADE20K/>`_

    Args:
        root (string): Root directory of the ADE20K dataset
        split (string, optional): The image split to use, ``train`` or ``val``
        download (bool, optional): If true, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in a
            PIL image and returns a transformed version. E.g,
            ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes
            in the PIL image target and transforms it.
        transforms (callable, optional): A function/transform that takes input
            sample and its target as entry and returns a transformed version.

    Examples:
        Get dataset for training and download from internet

        .. code-block:: python

            dataset = ADE20K('./data/ade20k', split='train', download=True)

            img, target = dataset[0]

        Get dataset for validation and download from internet

        .. code-block:: python

            dataset = ADE20K('./data/ade20k', split='val', download=True)

            img, target = dataset[0]
    """

    MAP_LABELS = {
        0: "background",
        1: "wall",
        2: "building;edifice",
        3: "sky",
        4: "floor;flooring",
        5: "tree",
        6: "ceiling",
        7: "road;route",
        8: "bed",
        9: "windowpane;window",
        10: "grass",
        11: "cabinet",
        12: "sidewalk;pavement",
        13: "person",
        14: "earth;ground",
        15: "door;double;door",
        16: "table",
        17: "mountain;mount",
        18: "plant;flora;plant;life",
        19: "curtain;drape;drapery;mantle;pall",
        20: "chair",
        21: "car;auto;automobile;machine;motorcar",
        22: "water",
        23: "painting;picture",
        24: "sofa;couch;lounge",
        25: "shelf",
        26: "house",
        27: "sea",
        28: "mirror",
        29: "rug;carpet;carpeting",
        30: "field",
        31: "armchair",
        32: "seat",
        33: "fence;fencing",
        34: "desk",
        35: "rock;stone",
        36: "wardrobe;closet;press",
        37: "lamp",
        38: "bathtub;bathing;tub;bath;tub",
        39: "railing;rail",
        40: "cushion",
        41: "base;pedestal;stand",
        42: "box",
        43: "column;pillar",
        44: "signboard;sign",
        45: "chest;of;drawers;chest;bureau;dresser",
        46: "counter",
        47: "sand",
        48: "sink",
        49: "skyscraper",
        50: "fireplace;hearth;open;fireplace",
        51: "refrigerator;icebox",
        52: "grandstand;covered;stand",
        53: "path",
        54: "stairs;steps",
        55: "runway",
        56: "case;display;case;showcase;vitrine",
        57: "pool;table;billiard;table;snooker;table",
        58: "pillow",
        59: "screen;door;screen",
        60: "stairway;staircase",
        61: "river",
        62: "bridge;span",
        63: "bookcase",
        64: "blind;screen",
        65: "coffee;table;cocktail;table",
        66: "toilet;can;commode;crapper;pot;potty;stool",
        67: "flower",
        68: "book",
        69: "hill",
        70: "bench",
        71: "countertop",
        72: "stove;kitchen;stove;range;kitchen;cooking;stove",
        73: "palm;palm;tree",
        74: "kitchen;island",
        75: "computer",
        76: "swivel;chair",
        77: "boat",
        78: "bar",
        79: "arcade;machine",
        80: "hovel;hut;hutch;shack;shanty",
        81: "bus;coach;double-decker;passenger;vehicle",
        82: "towel",
        83: "light;light;source",
        84: "truck;motortruck",
        85: "tower",
        86: "chandelier;pendant;pendent",
        87: "awning;sunshade;sunblind",
        88: "streetlight;street;lamp",
        89: "booth;cubicle;stall;kiosk",
        90: "television",
        91: "airplane;aeroplane;plane",
        92: "dirt;track",
        93: "apparel;wearing;apparel;dress;clothes",
        94: "pole",
        95: "land;ground;soil",
        96: "bannister;banister;balustrade;balusters;handrail",
        97: "escalator;moving;staircase;moving;stairway",
        98: "ottoman;pouf;pouffe;puff;hassock",
        99: "bottle",
        100: "buffet;counter;sideboard",
        101: "poster;posting;placard;notice;bill;card",
        102: "stage",
        103: "van",
        104: "ship",
        105: "fountain",
        106: "conveyer;belt;conveyor;belt;conveyor;transporter",
        107: "canopy",
        108: "washer;automatic;washer;washing;machine",
        109: "plaything;toy",
        110: "swimming;pool;swimming;bath;natatorium",
        111: "stool",
        112: "barrel;cask",
        113: "basket;handbasket",
        114: "waterfall;falls",
        115: "tent;collapsible;shelter",
        116: "bag",
        117: "minibike;motorbike",
        118: "cradle",
        119: "oven",
        120: "ball",
        121: "food;solid;food",
        122: "step;stair",
        123: "tank;storage;tank",
        124: "trade;name;brand;name;brand;marque",
        125: "microwave;microwave;oven",
        126: "pot;flowerpot",
        127: "animal;animate;being;beast;brute;creature;fauna",
        128: "bicycle;bike;wheel;cycle",
        129: "lake",
        130: "dishwasher;dish;washer;dishwashing;machine",
        131: "screen;silver;screen;projection;screen",
        132: "blanket;cover",
        133: "sculpture",
        134: "hood;exhaust;hood",
        135: "sconce",
        136: "vase",
        137: "traffic;light;traffic;signal;stoplight",
        138: "tray",
        139: "trash;can;garbage;wastebin;bin;ashbin;dustbin;barrel;bin",
        140: "fan",
        141: "pier;wharf;wharfage;dock",
        142: "crt;screen",
        143: "plate",
        144: "monitor;monitoring;device",
        145: "bulletin;board;notice;board",
        146: "shower",
        147: "radiator",
        148: "glass;drinking;glass",
        149: "clock",
        150: "flag",
    }

    def __init__(
        self,
        root,
        split="train",
        download=False,
        transform=None,
        target_transform=None,
        transforms=None,
    ):
        super(ADE20K, self).__init__(root, transforms, transform, target_transform)

        base_dir = ARCHIVE_DICT["trainval"]["base_dir"]

        if split not in ["train", "val"]:
            raise ValueError('Invalid split! Please use split="train" or split="val"')

        if split == "train":
            self.images_dir = os.path.join(self.root, base_dir, "images", "training")
            self.targets_dir = os.path.join(
                self.root, base_dir, "annotations", "training"
            )
        elif split == "val":
            self.images_dir = os.path.join(self.root, base_dir, "images", "validation")
            self.targets_dir = os.path.join(
                self.root, base_dir, "annotations", "validation"
            )

        self.split = split

        if download:
            self.download()

        self.images = []
        self.targets = []

        for file_name in os.listdir(self.images_dir):
            self.images.append(os.path.join(self.images_dir, file_name))
            self.targets.append(
                os.path.join(self.targets_dir, file_name.replace("jpg", "png"))
            )

    def download(self):
        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):

            archive_dict = ARCHIVE_DICT["trainval"]
            download_extract(
                archive_dict["url"],
                self.root,
                "ADEChallengeData2016.zip",
                archive_dict["md5"],
            )

        else:
            msg = (
                "You set download=True, but a folder ADE already exist "
                "in the root directory. If you want to re-download or "
                "re-extract the archive, delete the folder."
            )
            print(msg)

    def __getitem__(self, index):
        """Getitem special method.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target)
        """

        image = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.targets[index])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images)

    def extra_repr(self):
        lines = ["Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)
