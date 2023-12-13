import os
import cv2
import selectivesearch
import argparse


def selective_search(img, min_area=600, min_size=30, scale=600):
    img_lbl, regions = selectivesearch.selective_search(img, scale=scale, sigma=0.9, min_size=min_size)
    
    temp = set()
    for i in range(img_lbl.shape[0]):
        for j in range(img_lbl.shape[1]):    
            temp.add(img_lbl[i,j,3]) 
            
    candidates = set()
    for r in regions:
        if r['rect'] in candidates:
            continue
        if r['size'] < min_area:
            continue
        x, y, w, h = r['rect']
        if w < 40 or h < 40:
            continue
        if w / h > 3 or h / w > 3: 
            continue
        candidates.add(r['rect'])
    return candidates


if __name__ == '__main__':
    parser = argparse.ArgumentParser('parse data', add_help=False)
    parser.add_argument("--base_path", type=str, default=' ')
    parser.add_argument("--dataset", type=str, default=' ')
    args = parser.parse_args()
    
    base_path = args.base_path
    img_num = 0
    roi_num = 0
    dataset = args.dataset
    
    print(f"Start parse dataset {dataset}.")
    img_folder_path = os.path.join(base_path, dataset, 'thermal')
    txt_folder_path = os.path.join(base_path, dataset, 'roi')
    assert os.path.exists(img_folder_path), "The img folder does not exist!"
    os.makedirs(txt_folder_path, exist_ok=True)
    
    img_path_list = os.listdir(img_folder_path)
    for img_path in img_path_list:
        img_num += 1
        img = cv2.imread(os.path.join(img_folder_path, img_path))
        candidates = selective_search(img)
        roi_num += len(candidates)
        img_name = img_path.split('.')[0]
        txt_path = os.path.join(txt_folder_path, img_name + '.txt')
        with open(txt_path, 'w') as f:
            for x, y, w, h in candidates:
                f.write(f"{x},{y},{w},{h}\n")
        
        if img_num % 1000 == 0:
            print(f"Parse {img_num} images.")
    avg_roi = roi_num / img_num
    print(f"Parse {img_num} images totally, average {avg_roi:.1f} rois per image.")
        