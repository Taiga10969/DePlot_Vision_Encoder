import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from deplot_vidion_encoder_models.pix2struct_image_processor import Pix2StructImageProcessor
from deplot_vidion_encoder_models.pix2struct_vision_model import Pix2StructVisionModel
from deplot_vidion_encoder_models.pix2struct_vision_config import Pix2StructVisionConfig
from moonshot_dataset import Related_Text_Dataset

from utils import Attention_Rollout

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

# batch_size = 2 の画像データを作成
image_indexs = [7, 11]
image_paths = []
for i in image_indexs:
    image_path, _, _, _ = dataset[i]
    image_paths.append(image_path)

images=[]
for image_path in image_paths:
    image = Image.open(os.path.join(root_dir, image_path)).convert('RGB')
    images.append(image)

summary(model)


# Step1 画像をprocessorに入力し，パッチ分割 and 埋め込み した状態に変換する
inputs, info = processor(images=images, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()} # 入力をdeviceに送る

outputs = model(flattened_patches = inputs['flattened_patches'], 
                attention_mask = inputs['attention_mask'],
                output_attentions = True,
                output_hidden_states = False,
                return_dict = True,
                )

print("outputs['last_hidden_state'].shape : ", outputs['last_hidden_state'].shape) # torch.Size([2, 2048, 768]) >> [batch, max_patchs, dim]
print("len(outputs['attentions']) : ", len(outputs['attentions']))                 # 12 >> 12layer分のAttentionデータがタプル型で返されてる．
# 最終layerのAttention情報の形状を確認
print("outputs['attentions'][-1].shape : ", outputs['attentions'][-1].shape)       # torch.Size([2, 12, 2048, 2048]) >> [batch, head, max_patchs, max_patchs]



# Attention data processing
batch = 0 # Attentionデータを収集する画像のbatch_index
attention_list = []
for i in range(len(outputs['attentions'])):
    attention_data = outputs['attentions'][i][batch]
    attention_list.append(outputs['attentions'][i][batch].cpu().numpy())

print('np.shape(attention_list) : ', np.shape(attention_list)) # (12, 12, 2048, 2048) >> (layer, head, max_patchs, max_patchs)

# layer=0, head=0 を可視化
plt.imshow(attention_list[0][0])
plt.savefig('./output_images/attention_list_layer0_head0.png')
plt.close()


vision_attn = Attention_Rollout(attention_list)

print('np.shape(vision_attn) : ', np.shape(vision_attn))

plt.imshow(vision_attn)
plt.savefig('./output_images/attention_rollout.png')
plt.close()
