import json
import pdb
from collections import Counter

def get_instance(data, cat_id):
    return [a for a in data['annotations'] if a['category_id'] == cat_id]

def count_images(data, cat_id):
    instances = get_instance(data, cat_id)
    return len(instances), len(Counter([instance['image_id'] for instance in instances]))


def main():
    f = open('Amodal-Instance-Segmentation-through-KINS-Dataset/instances_val.json')
    data = json.load(f)

    #lvis_keys = ['info', 'categories', 'annotations', 'images', 'licences']
    elvis_data = {}


    #convert the categories
    elvis_data['categories'] = []
    for cat in data['categories']:
        elvis_data['categories'].append({
            "id" : cat['id'],
            "name" : cat['name']
        })


    #convert the ima
    elvis_data['images'] = []
    #pdb.set_trace()
    for image in data['images']:
        elvis_data['images'].append({
            "id" : image['id'],
            "width" : image['width'],
            "height" : image['height'],
            "file_name": image['file_name']
        })

    print([e['id'] for e in elvis_data['images']])

    #convert the annotations
    elvis_data['annotations'] = []
    for annotation in data['annotations']:
        elvis_data['annotations'].append({
            "id" : annotation['id'],
            "area" : annotation['area'],
            "image_id" : annotation['image_id'],
            "segmentation" : annotation['segmentation'],
            "bbox" : annotation['bbox'],
            "category_id" : annotation['category_id']
        })

    for categories in elvis_data['categories']:
        instance_count, image_count = count_images(elvis_data, categories['id'])
        categories['image_count'] = image_count
        categories['instance_count'] = instance_count
    
    with open("sample.json", "w") as outfile:
        json.dump(elvis_data, outfile)


if __name__ == "__main__":
    main()