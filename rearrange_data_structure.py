from parameters import *
from all_functions import *
from tqdm import tqdm

old_data = open_json(ALL_DATA_FILE)
labels = open_json(LABELS_FILE)
patterns = open_json(PATTERNS_FILE)
materials = open_json(MATERIAL_FILE)
sleeves = open_json(SLEEVE_TYPE_FILE)
categories = open_json(CATEGORY_FILE)
fit = open_json(FIT_FILE)
gender = open_json(GENDER_FILE)
occasion = open_json(OCCASION_FILE)
style = open_json(STYLE_FILE)
new_data = {}

for data in tqdm(old_data):
    print(data, '---------->')
    new_data[data] = dict()
    items = list()
    for item in old_data[data]['Items']:
        item_info = dict()
        item_info['cate3_id'] = item['Cate3_ID']
        item_info['sku_id'] = item['SKU_ID']
        item_info['image_name'] = item['Image']
        item_info['item_name'] = item['Name_Short']
        tags = item['Tags']
        item_info['pattern'] = list()
        if '93' in tags.keys():
            pat_list = []
            for t in tags['93']:
                pat_list.append(t['label_name'])
            pat_list = list(set(pat_list))
            for p in pat_list:
                item_info['pattern'].append({patterns[p]:p})
        item_info['material'] = list()
        if '94' in tags.keys():
            mat_list = []
            for t in tags['94']:
                mat_list.append(t['label_name'])
            mat_list = list(set(mat_list))
            for m in mat_list:
                item_info['material'].append({materials[m]:m})
        item_info['sleeve_type'] = list()
        if '95' in tags.keys():
            sleeve_list = []
            for t in tags['95']:
                sleeve_list.append(t['label_name'])
            sleeve_list = list(set(sleeve_list))
            for s in sleeve_list:
                item_info['sleeve_type'].append({sleeves[s]:s})
        item_info['category'] = list()
        if '97' in tags.keys():
            cat_list = []
            for t in tags['97']:
                cat_list.append(t['label_name'])
            cat_list = list(set(cat_list))
            for c in cat_list:
                item_info['category'].append({categories[c]:c})
                item_info['category'] = list()
        item_info['label'] = list()
        label_list = list()
        if '98' in tags.keys():
            for t in tags['98']:
                label_list.append(t['label_name'])
        if '99' in tags.keys():
            for t in tags['99']:
                label_list.append(t['label_name'])
        label_list = list(set(label_list))
        for c in label_list:
            item_info['label'].append({labels[c]:c})
        items.append(item_info)
    new_data[data]['items'] = items
    new_data[data]['outfit_fit'] = {fit[old_data[data]['Outfit_Fit']]:old_data[data]['Outfit_Fit']}
    new_data[data]['outfit_gender'] = {gender[old_data[data]['Outfit_Gender']]:old_data[data]['Outfit_Gender']}
    new_data[data]['outfit_occasion'] = {occasion[old_data[data]['Outfit_Occasion']]:old_data[data]['Outfit_Occasion']}
    new_data[data]['outfit_style'] = {style[old_data[data]['Outfit_Style']]:old_data[data]['Outfit_Style']}
    new_data[data]['outfit_id'] = old_data[data]['Outfit_ID']
    new_data[data]['outfit_images'] = old_data[data]['Outfit_Images']

write_json(REARRANGED_DATA, new_data)
print('Data file is ready!')