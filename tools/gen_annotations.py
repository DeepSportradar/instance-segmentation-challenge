import json
import glob
import re
import imageio
import numpy as np
from pycocotools import mask as mask_tools
from tqdm import tqdm


_split_to_arenas = {
    "A": ['KS-FR-CAEN', 'KS-FR-LIMOGES', 'KS-FR-ROANNE'],
    "B": ['KS-FR-NANTES', 'KS-FR-BLOIS', 'KS-FR-FOS'],
    "C": ['KS-FR-LEMANS', 'KS-FR-MONACO', 'KS-FR-STRASBOURG'],
    "D": ['KS-FR-GRAVELINES', 'KS-FR-STCHAMOND', 'KS-FR-POITIERS'],
    "E": ['KS-FR-NANCY', 'KS-FR-BOURGEB', 'KS-FR-VICHY'],
}


def scan():
    """
    Compute the full COCO-format annotation JSON from the files on the disk.
    """
    files = glob.glob(f'deepsport_dataset/deepsport_dataset_*/*/*/*')
    images = set([re.sub(r"(_(0|40|humans)\.png|\.json)", "", file) for file in files])
    images = sorted(images)

    root = dict(images=[], annotations=[], categories=[
        dict(id=0, name="background", supercategory="background"),
        dict(id=1, name="human", supercategory="human"),
    ])

    an_id = -1
    for i, image in enumerate(tqdm(images)):
        img = imageio.imread(image+'_0.png')
        panoptic = imageio.imread(image+'_humans.png')
        img_id = len(root['images'])
        root['images'].append(dict(
            file_name='/'.join(image.split('/')[1:])+'_0.png',
            width=img.shape[1],
            height=img.shape[0],
            id=img_id,
        ))
        for pan_id in np.unique(panoptic):
            if pan_id < 1000 or pan_id >= 2000:
                continue
            an_id += 1
            mask = panoptic == pan_id
            rle = mask_tools.encode(np.asfortranarray(mask))

            root['annotations'].append(dict(
                id=len(root['annotations']),
                image_id=img_id,
                category_id=1,
                area=mask_tools.area(rle).item(),
                bbox=mask_tools.toBbox(rle).tolist(),
                segmentation=dict(size=rle['size'], counts=rle['counts'].decode('utf-8')),
                iscrowd=0,
            ))
    return root


def dump_split(name, splits, *, root):
    """
    Filter the full COCO annotation JSON in the split of interest.
    Dump as JSON.
    """
    arenas = set(sum([_split_to_arenas[split] for split in splits], []))
    ret = {}
    ret['categories'] = root['categories']
    ret['images'] = [image for image in root['images']
                     if image['file_name'].split('/')[1] in arenas]
                     # FIXME Adapt when Kaggle deepsport_dataset_X are merged
    if name == 'train':
        del ret['images'][4::7]
    elif name == 'val':
        ret['images'] = ret['images'][4::7]

    kept_im_ids = set(image['id'] for image in ret['images'])

    ret['annotations'] = [annot for annot in root['annotations']
                          if annot['image_id'] in kept_im_ids]
    print(name, len(ret['images']), 'images', len(ret['annotations']), 'annotations')
    json.dump(ret, open(f'deepsport_dataset/{name}.json', 'w'))


root = scan()
dump_split('train', 'BCDE', root=root)
dump_split('val', 'BCDE', root=root)
dump_split('trainval', 'BCDE', root=root)
dump_split('test', 'A', root=root)
dump_split('trainvaltest', 'ABCDE', root=root)