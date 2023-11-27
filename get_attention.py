import os
import cv2
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from deplot_vidion_encoder_models.pix2struct_image_processor import Pix2StructImageProcessor
from deplot_vidion_encoder_models.pix2struct_vision_model import Pix2StructVisionModel
from deplot_vidion_encoder_models.pix2struct_vision_config import Pix2StructVisionConfig
from moonshot_dataset import Related_Text_Dataset

from utils import Attention_Rollout, heatmap_to_rgb

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
image_indexs = [7, 70] # 10
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


# Attention data processing (モデルの出力からAttention情報を取得)

batch = 1 # Attentionデータを収集する画像のbatch_index
save_format = 'png'

attention_list = []
for i in range(len(outputs['attentions'])):
    attention_data = outputs['attentions'][i][batch]
    attention_list.append(outputs['attentions'][i][batch].detach().cpu().numpy())
print('np.shape(attention_list) : ', np.shape(attention_list)) # (12, 12, 2048, 2048) >> (layer, head, max_patchs, max_patchs)

# processorが出力する画像情報の詳細(info)を取得
patch_nums = info['extract_flattened_patches_info'][batch]
patch_num_rows = patch_nums['rows']         # パッチの行と列の数
patch_num_columns = patch_nums['columns']   # パッチの行と列の数 
active_patch_num = patch_num_rows * patch_num_columns

print(f'processor info \npatch_num_rows = {patch_num_rows}\npatch_num_columns = {patch_num_columns}\nactive_patch_num = {active_patch_num}')


# 入力画像の可視化
img = images[batch]
#print('type(img) : ', type(img)) # <class 'PIL.Image.Image'>
img_width, img_height = img.size
print(f'input_image :: width = {img_width}, height = {img_height}') # width = 523  height = 554
plt.imshow(img)
plt.savefig(f'./get_attention_output/input_image.{save_format}')
plt.close()


# Attention Rollout
attn_rollouted = Attention_Rollout(attention_list)
print('np.shape(attn_rollouted) : ', np.shape(attn_rollouted)) # (2048, 2048) >> (max_patchs, max_patchs)

plt.imshow(attn_rollouted)
plt.savefig(f'./get_attention_output/rollouted_attention_weights.{save_format}')
plt.close()


attention = attn_rollouted[:active_patch_num, :active_patch_num] # processorでのpadding部分を削除
attention = attention[batch]
attention = np.reshape(attention, (patch_num_rows, patch_num_columns)) # 画像のサイズにreshape 縦横で取るpatch数が異なる．

# 画像サイズに拡大し，RGB3次元画像に変換し，画像と重ね合わせる
attention_map = cv2.resize(attention, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
attention_map_rgb = heatmap_to_rgb(attention_map)
attention_map = cv2.addWeighted(np.array(img), 0.25, attention_map_rgb, 0.75, 0)

plt.imshow(attention_map)
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
plt.savefig(f'./get_attention_output/attention_map.{save_format}')
plt.close()


# 各パッチの幅と高さ
patch_width = img_width / patch_num_columns
patch_height = img_height / patch_num_rows

print(f"patch_size :: width = {patch_width}, height = {patch_height}")

# 画像の表示
plt.imshow(img)

# 格子状の枠を描画
for i in range(patch_num_rows):
    plt.axhline(y=i * patch_height, color='r', linestyle='solid', linewidth=0.5)

for j in range(patch_num_columns):
    plt.axvline(x=j * patch_width, color='r', linestyle='solid', linewidth=0.5)

# 結果の保存
plt.savefig(f'./get_attention_output/input_image_with_patch_grid.{save_format}')
plt.close()
