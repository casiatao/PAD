import torch
import re

cpk_path = 'mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth'
checkpoint = torch.load(cpk_path, map_location='cpu')

new_checkpoint = {'model': {}}
for key, value in checkpoint['state_dict'].items():
    if key == 'backbone.ln1.weight':
        new_key = 'norm.weight'
        new_checkpoint['model'][new_key] = value
    elif key == 'backbone.ln1.bias':
        new_key = 'norm.bias'
        new_checkpoint['model'][new_key] = value
    elif key == 'neck.ln1.weight':
        new_key = 'decoder_norm.weight'
        new_checkpoint['model'][new_key] = value
    elif key == 'neck.ln1.bias':
        new_key = 'decoder_norm.bias'
        new_checkpoint['model'][new_key] = value
    
    elif 'backbone' in key or 'neck' in key:
        new_key = key.replace('backbone.', '')
        new_key = new_key.replace('neck.', '')
        new_key = re.sub(r'^layers(.*)', r'blocks\1', new_key)
        new_key = re.sub(r'(.*)\.(\d+)\.ffn\.layers\.0\.0\.(\w+)$', r'\1.\2.mlp.fc1.\3', new_key)
        new_key = re.sub(r'(.*)\.(\d+)\.ffn\.layers\.1\.(\w+)$', r'\1.\2.mlp.fc2.\3', new_key)
        new_key = re.sub(r'(.*)ln(\d+)(.*)', r'\1norm\2\3', new_key)
        new_key = re.sub(r'(.*)projection(.*)', r'\1proj\2', new_key)
        
        new_checkpoint['model'][new_key] = value

torch.save(new_checkpoint, "mae_pretrain_vit_mm.pth")
