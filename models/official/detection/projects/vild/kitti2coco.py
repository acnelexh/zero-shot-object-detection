import glob
import os
import json
from PIL import Image
from pathlib import Path
import pdb

def generate_annotations(image_type = "training"):
    #directory where the json file will be stored
    kitti_annotation_directory = Path('./data/kitti/annotations')
    if not kitti_annotation_directory.exists():
        kitti_annotation_directory.mkdir(parents=True)


    image_directory = Path("./data/kitti/training/image_2")
    image_list = [f for f in image_directory.glob("*.png")]
    image_list.sort()

    category_names = [ 
        "Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"
    ]

    coco_categories = []
    for i, c in enumerate(category_names):
        coco_categories.append({
            "id" : i,
            "name" : c,
            "keypoints" : []
        })


    coco_images = []
    for i, image in enumerate(image_list):
        im = Image.open(image)
        w, h = im.size
        coco_images.append(
            {
                "id" : i,
                "file_name" : os.path.basename(image),
                "height" : h,
                "width" : w
            }
        )

    coco_annotations = []

    #pdb.set_trace()
    annotation_folder = Path("./data/kitti/training/label_2")
    id = 0
    for i, image in enumerate(image_list):

        #get the file name with extension
        file_name = os.path.splitext(os.path.basename(image))[0]

        #get the annotation file
        annotation_file = annotation_folder/f"{file_name}.txt"

        # small modification
        
        if not annotation_file.exists():
            continue
        #pdb.set_trace()
        f = open(annotation_file, 'r')
        lines = f.readlines()
        for line in lines:
            line = line.split()
            cat = line[0]
            if cat != 'DontCare':
                box  = [float(x) for x in line[4: 8]]
                bbox = [box[0], box[1], round(box[2] - box[0], 2), round(box[3] - box[1], 2)]
                area = round(bbox[2] * bbox[3], 2)
                cat_id = category_names.index(cat)
                image_id = i
                
                annotation = {
                    "area" : area,
                    "image_id" : image_id,
                    "bbox" : bbox,
                    "category_id" : cat_id,
                    "id" : id,
                    "iscrowd" : 0,
                    "segmentation" : {}
                }

                coco_annotations.append(annotation)

                id += 1

    #write to annotation file
    out_file = os.path.join(kitti_annotation_directory, "instances_" + image_type + ".json")

    with open(out_file, 'w') as f:
        json.dump({
            "images" : coco_images,
            "categories" : coco_categories,
            "annotations" : coco_annotations
        },f)

    
generate_annotations()