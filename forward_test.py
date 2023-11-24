import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from deplot_vidion_encoder_models.pix2struct_image_processor import Pix2StructImageProcessor
from deplot_vidion_encoder_models.pix2struct_vision_model import Pix2StructVisionModel
from deplot_vidion_encoder_models.pix2struct_vision_config import Pix2StructVisionConfig
from moonshot_dataset import Related_Text_Dataset

from torchinfo import summary

# check GPU usage
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()==0: print('Use 1 GPU')
else: print(f'Use {torch.cuda.device_count()} GPUs')



vision_config = Pix2StructVisionConfig.from_pretrained('google/deplot')
model = Pix2StructVisionModel(vision_config).to(device)
processor = Pix2StructImageProcessor()


msg = model.load_state_dict(torch.load('model.pth', map_location=torch.device(device)))
print("model.load_state_dict message : ", msg)
print("model.device : ", model.device)

root_dir = '/taiga/Datasets/moonshot-dataset'
dataset = Related_Text_Dataset(root_dir=root_dir, 
                               image_processor=None, 
                               tokenizer=None, 
                               is_train=False, 
                               split_caption=True, 
                               )

i=7
image_path, _, _, _ = dataset[i]


image = Image.open(os.path.join(root_dir, image_path)).convert('RGB')
print(np.shape(image))

summary(model)


# Step1 画像をprocessorに入力し，パッチ分割 and 埋め込み した状態に変換する
inputs, info = processor(images=image, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()} # 入力をdeviceに送る

# processorの出力の形状を確認 (1バッチごとに，タプル型で提供されているため，0番目のデータの形状を確認する)
print("inputs['flattened_patches'][0].shape : ", inputs['flattened_patches'][0].shape)  #torch.Size([2048, 770]) >> [max_patches, dim]

outputs = model(flattened_patches = inputs['flattened_patches'], 
                attention_mask = inputs['attention_mask'],
                output_attentions = False,
                output_hidden_states = False,
                return_dict = False,
                )

# 出力の特徴量を確認
print("hidden_states.shape : ", outputs[0].shape) # torch.Size([1, 2048, 768]) >> [batch, max_patches, dim]
