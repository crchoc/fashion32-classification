from parameters import *
from all_functions import *
from tqdm import tqdm

# Get labels and put numbers on it
data_dict = open_json(ALL_DATA_FILE)
labels_type = []
image_to_label = {}
pattern_type = list()
material_type = list()
sleeve_type = list()
cat_type = list()
fit = list()
gender = list()
occasion = list()
style = list()

for data in tqdm(data_dict):
    print(data, '-------->')
    items = data_dict[data]['Items']
    for item in items:
        tags = item['Tags']
        image_to_label[item['Image']] = []
        if '98' in tags.keys(): # women's clothes
            label_list = []
            for t in tags['98']:
                label_list.append(t['label_name'])
            labels_type.extend(label_list)
            if len(image_to_label[item['Image']])==0:
                image_to_label[item['Image']] = label_list
        if '99' in tags.keys(): # men's clothes
            label_list = []
            for t in tags['99']:
                label_list.append(t['label_name'])
            labels_type.extend(label_list)
            if len(image_to_label[item['Image']])==0:
                image_to_label[item['Image']] = label_list
        if '93' in tags.keys(): # women's clothes
            patt_list = []
            for t in tags['93']:
                patt_list.append(t['label_name'])
            pattern_type.extend(patt_list)
        if '94' in tags.keys(): # women's clothes
            mat_list = []
            for t in tags['94']:
                mat_list.append(t['label_name'])
            material_type.extend(mat_list)
        if '95' in tags.keys(): # women's clothes
            sleeve_list = []
            for t in tags['95']:
                sleeve_list.append(t['label_name'])
            sleeve_type.extend(sleeve_list)
        if '97' in tags.keys(): # women's clothes
            cat_list = []
            for t in tags['97']:
                cat_list.append(t['label_name'])
            cat_type.extend(cat_list)
    fit.append(data_dict[data]['Outfit_Fit'])
    gender.append(data_dict[data]['Outfit_Gender'])
    occasion.append(data_dict[data]['Outfit_Occasion'])
    style.append(data_dict[data]['Outfit_Style'])

labels_type = list(set(labels_type))
labels_type.sort()
labels_dict = {}
i = 0
for l in labels_type:
    labels_dict[l] = i
    i += 1

pattern_type = list(set(pattern_type))
pattern_type.sort()
patt_dict = {}
i = 0
for l in pattern_type:
    patt_dict[l] = i
    i += 1

material_type = list(set(material_type))
material_type.sort()
mat_dict = {}
i = 0
for l in material_type:
    mat_dict[l] = i
    i += 1

sleeve_type = list(set(sleeve_type))
sleeve_type.sort()
sleeve_dict = {}
i = 0
for l in sleeve_type:
    sleeve_dict[l] = i
    i += 1

cat_type = list(set(cat_type))
cat_type.sort()
cat_dict = {}
i = 0
for l in cat_type:
    cat_dict[l] = i
    i += 1

fit = list(set(fit))
fit.sort()
fit_dict = dict()
i = 0
for f in fit:
    fit_dict[f] = i
    i += 1

gender = list(set(gender))
gender.sort()
gender_dict = dict()
i = 0
for g in gender:
    gender_dict[g] = i
    i += 1

occasion = list(set(occasion))
occasion.sort()
occasion_dict = dict()
i = 0
for o in occasion:
    occasion_dict[o] = i
    i += 1

style = list(set(style))
style.sort()
style_dict = dict()
i = 0
for s in style:
    style_dict[s] = i
    i += 1

for image in image_to_label:
    image_to_label[image] = labels_dict[image_to_label[image][0]]
write_json(LABELS_FILE, labels_dict)
write_json(IMGS_LABEL, image_to_label)
write_json(PATTERNS_FILE, patt_dict)
write_json(MATERIAL_FILE, mat_dict)
write_json(SLEEVE_TYPE_FILE, sleeve_dict)
write_json(CATEGORY_FILE, cat_dict)
write_json(FIT_FILE, fit_dict)
write_json(GENDER_FILE, gender_dict)
write_json(OCCASION_FILE, occasion_dict)
write_json(STYLE_FILE, style_dict)

print('Number of labels: ', len(labels_dict), 
      '\nNumber of patterns: ', len(patt_dict), 
      '\nNumber of materials: ', len(mat_dict), 
      '\nNumber of sleeve types: ', len(sleeve_dict), 
      '\nNumber of images: ', len(image_to_label))
print('Labels ready, DONE!')