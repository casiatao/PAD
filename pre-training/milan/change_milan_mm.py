import torch
import re

cpk_path = 'milan_vit-base-p16_16xb256-amp-coslr-400e_in1k_20221129-180922e8.pth'
checkpoint = torch.load(cpk_path, map_location='cpu')

new_checkpoint = {'model': {}}
clip_checkpoint = {'model': {}}
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
        
    if 'target_generator' in key:
        new_key = key.replace('target_generator.tokenizer.', '')
        clip_checkpoint['model'][new_key] = value

torch.save(new_checkpoint, "milan_vit_base_init.pth")
torch.save(clip_checkpoint, "clip_vit_base_mm.pth")


