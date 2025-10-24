import torch
from model import DonutModel, DonutConfig
from PIL import Image
import numpy as np


# Config 값 설정 및 Fusion 활성화
config = DonutConfig(
    input_size= [2560, 1920],
    use_fusion=True,
    fusion_hidden_dim=1024 ,
    fusion_num_heads=8
) 

print("=" * 60)

model = DonutModel(config)
dummy_pli_ima = Image.fromarray(
    np.random.randint(0, 255, (1920, 2560, 3), dtype=np.uint8)
)
image_tensor = model.encoder.prepare_input(dummy_pli_ima).unsqueeze(0)

# 텍스트 데이터
dummy_input_ids = torch.randint(0,1000, (1,50))
dummy_labels = torch.randint(0,1000, (1,50))


# forword 확인
try:
    outputs = model(
        image_tensors=image_tensor,
        decoder_input_ids=dummy_input_ids,
        decoder_labels=dummy_labels
    )
    print("성공")
    print({outputs.loss})
    print({outputs.logits.shape})    
except Exception as e:
    print("실패", {e})
    import traceback
    traceback.print_exc()

if hasattr(model, 'fusion_layer'):
    print("Fusion layer 테스트")

    encoder_out = model.encoder(image_tensor)
    print("encoder 값", {encoder_out.shape})

    decoder_embeds = model.decoder.model.model.decoder.embed_tokens(dummy_input_ids)
    print("임베딩", {decoder_embeds.shape})

    fused = model.fusion_layer(encoder_out, decoder_embeds)
    print("fusion 출력", {fused.shape})

    #실패 {AttributeError("'SwinTransformer' object has no attribute 'pos_drop'")}