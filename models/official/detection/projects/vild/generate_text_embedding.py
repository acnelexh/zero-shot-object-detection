import json
from pathlib import Path
import torch
from tqdm import tqdm
import numpy
import pdb
from easydict import EasyDict
import clip
import pandas as pd

USE_GPU = False
SAVE_PATH = Path("./tmp")

#@title Define hyperparameters
FLAGS = {
    'prompt_engineering': True,
    'this_is': True,
    
    'temperature': 100.0,
    'use_softmax': False,
}
FLAGS = EasyDict(FLAGS)

single_template = [
    'a photo of {article} {}.'
]

multiple_templates = [
    'There is {article} {} in the scene.',
    'There is the {} in the scene.',
    'a photo of {article} {} in the scene.',
    'a photo of the {} in the scene.',
    'a photo of one {} in the scene.',


    'itap of {article} {}.',
    'itap of my {}.',  # itap: I took a picture of
    'itap of the {}.',
    'a photo of {article} {}.',
    'a photo of my {}.',
    'a photo of the {}.',
    'a photo of one {}.',
    'a photo of many {}.',

    'a good photo of {article} {}.',
    'a good photo of the {}.',
    'a bad photo of {article} {}.',
    'a bad photo of the {}.',
    'a photo of a nice {}.',
    'a photo of the nice {}.',
    'a photo of a cool {}.',
    'a photo of the cool {}.',
    'a photo of a weird {}.',
    'a photo of the weird {}.',

    'a photo of a small {}.',
    'a photo of the small {}.',
    'a photo of a large {}.',
    'a photo of the large {}.',

    'a photo of a clean {}.',
    'a photo of the clean {}.',
    'a photo of a dirty {}.',
    'a photo of the dirty {}.',

    'a bright photo of {article} {}.',
    'a bright photo of the {}.',
    'a dark photo of {article} {}.',
    'a dark photo of the {}.',

    'a photo of a hard to see {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of {article} {}.',
    'a low resolution photo of the {}.',
    'a cropped photo of {article} {}.',
    'a cropped photo of the {}.',
    'a close-up photo of {article} {}.',
    'a close-up photo of the {}.',
    'a jpeg corrupted photo of {article} {}.',
    'a jpeg corrupted photo of the {}.',
    'a blurry photo of {article} {}.',
    'a blurry photo of the {}.',
    'a pixelated photo of {article} {}.',
    'a pixelated photo of the {}.',

    'a black and white photo of the {}.',
    'a black and white photo of {article} {}.',

    'a plastic {}.',
    'the plastic {}.',

    'a toy {}.',
    'the toy {}.',
    'a plushie {}.',
    'the plushie {}.',
    'a cartoon {}.',
    'the cartoon {}.',

    'an embroidered {}.',
    'the embroidered {}.',

    'a painting of the {}.',
    'a painting of a {}.',
]

def article(name):
      return 'an' if name[0] in 'aeiou' else 'a'

def processed_name(name, rm_dot=False):
  # _ for lvis
  # / for obj365
  res = name.replace('_', ' ').replace('/', ' or ').lower()
  if rm_dot:
    res = res.rstrip('.')
  return res

def build_text_embedding(categories, model):
    if FLAGS.prompt_engineering:
        templates = multiple_templates
    else:
        templates = single_template

    run_on_gpu = USE_GPU

    with torch.no_grad():
        all_text_embeddings = []
        print('Building text embeddings...')
        for category in tqdm(categories):
            texts = [
                template.format(processed_name(category['name'], rm_dot=True),
                                article=article(category['name']))
            for template in templates]
            if FLAGS.this_is:
                texts = [
                        'This is ' + text if text.startswith('a') or text.startswith('the') else text 
                        for text in texts
                        ]
            #pdb.set_trace()
            texts = clip.tokenize(texts) #tokenize
            if run_on_gpu:
                texts = texts.cuda()
            text_embeddings = model.encode_text(texts) #embed with text encoder
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            all_text_embeddings.append(text_embedding)
        all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
        if run_on_gpu:
            all_text_embeddings = all_text_embeddings.cuda()
    return all_text_embeddings.cpu().numpy()


def save_embedding(text_features, categories, dest_dir=Path("./tmp")):
    category_indices = {cat['id']: cat for cat in categories}
    numpy.save(dest_dir/"label_embed.npy", text_features, allow_pickle=True)
    ids = []
    names = []
    m_list = []
    for item in categories:
        m_list.append(f"{item['id']}:{item['name']}")
    df = pd.DataFrame(m_list)
    pdb.set_trace()
    df.to_csv(dest_dir/"label_map.csv", index=False)



def main():
    #pdb.set_trace()
    category_name_string = "superman;batman;spiderman"
    category_names = [x.strip() for x in category_name_string.split(';')]
    #category_names = ['background'] + category_names
    categories = [{'name': item, 'id': idx+1,} for idx, item in enumerate(category_names)]
    clip.available_models()
    clip_model, _ = clip.load("ViT-B/32")
    if USE_GPU:
        clip_model.cuda()
    else:
        clip_model.to('cpu')
    text_features = build_text_embedding(categories, clip_model)
    SAVE_PATH.mkdir(exist_ok=True)
    save_embedding(text_features, categories)



if __name__ == "__main__":
    main()