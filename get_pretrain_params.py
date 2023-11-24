import torch
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
model.to('cpu')

# DePlot Modelの内画像エンコーダ部分のパラメータのみ保存する．
vision_model_state_dict = model.encoder.state_dict()
torch.save(vision_model_state_dict,'model.pth')
