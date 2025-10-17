import torch
from model import DonutModel, DonutConfig

config = DonutConfig()

model = DonutModel(config)

# encoder 
print("=" * 50)
print("1. Encoder 출력 확인")
print("=" * 50)

dummy_image = torch.randn(1,3,2560,1920)
encoder_out = model.encoder(dummy_image)
print(f"Encoder출력: {encoder_out.shape}")


# decoder 임베딩
print("\n" + "=" * 50) 
print("2. Decoder 임베딩 차원 확인")
print("=" * 50)
dummy_ids = torch.randint(0, 1000, (1,50))
decoder_embeds = model.decoder.model.model.decoder.embed_tokens(dummy_ids)
print(f"Decoder 임베딩: {decoder_embeds.shape}")

# 차원이 같은지 확인
print("\n" + "=" * 50) 
print("3. 호환성 확인")
print("=" * 50)

if encoder_out.shape[-1] == decoder_embeds.shape[-1]:
    print("차원 확인, 호환 가능")
    hidden_dim = encoder_out.shape[-1]
    print(f"Hidden dimension: {hidden_dim}")
else:
    print("실패")
    print(f"Encoder 마지막 차원: {encoder_out.shape[-1]}")
    print(f"Decoder 마지막 차원: {decoder_embeds.shape[-1]}")