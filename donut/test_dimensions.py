import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np


processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

print("모델 로드 완료!\n")

# 1. Encoder 출력 확인
print("=" * 60)
print("1. Encoder 출력 확인")
print("=" * 60)

# PIL Image 생성 (랜덤 이미지)
dummy_pil_image = Image.fromarray(
    np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
)

# Processor로 이미지 전처리
pixel_values = processor(dummy_pil_image, return_tensors="pt").pixel_values

print(f"원본 이미지 크기: 480x640")
print(f"전처리된 이미지: {pixel_values.shape}")

# Encoder forward
encoder_out = model.encoder(pixel_values).last_hidden_state
print(f"Encoder 출력: {encoder_out.shape}")
print(f"  - Batch 크기: {encoder_out.shape[0]}")
print(f"  - 패치 수: {encoder_out.shape[1]}")
print(f"  - 특징 차원: {encoder_out.shape[2]}")




# 2. Decoder 임베딩 확인
print("\n" + "=" * 60)
print("2. Decoder 임베딩 차원 확인")
print("=" * 60)

# Decoder vocabulary 크기 확인
vocab_size = model.decoder.config.vocab_size
print(f"Vocabulary 크기: {vocab_size}")

# 유효한 토큰 ID 생성
dummy_ids = torch.randint(0, min(vocab_size, 1000), (1, 50))
decoder_embeds = model.decoder.model.decoder.embed_tokens(dummy_ids)

print(f"입력 토큰 ID: {dummy_ids.shape}")
print(f"Decoder 임베딩: {decoder_embeds.shape}")
print(f"  - Batch 크기: {decoder_embeds.shape[0]}")
print(f"  - 토큰 수: {decoder_embeds.shape[1]}")
print(f"  - 임베딩 차원: {decoder_embeds.shape[2]}")

# 3. 호환성 확인
print("\n" + "=" * 60)
print("3. CrossAttention 호환성 확인")
print("=" * 60)

encoder_dim = encoder_out.shape[-1]
decoder_dim = decoder_embeds.shape[-1]

if encoder_dim == decoder_dim:
    print("차원 호환 가능! CrossAttention 사용 가능")
    print(f"\n 상세 정보:")
    print(f"  - Hidden Dimension: {encoder_dim}")
    print(f"  - Encoder 패치 수: {encoder_out.shape[1]}")
    print(f"  - Decoder 토큰 수: {decoder_embeds.shape[1]}")
    print(f"\n CrossAttentionFusionLayer 설정:")
    print(f"   CrossAttentionFusionLayer(")
    print(f"       hidden_dim={encoder_dim},")
    print(f"       num_heads=8")
    print(f"   )")
    print(f"\n 사용 예시:")
    print(f"   vision_feats: ({encoder_out.shape[0]}, {encoder_out.shape[1]}, {encoder_dim})")
    print(f"   text_embeds:  ({decoder_embeds.shape[0]}, {decoder_embeds.shape[1]}, {decoder_dim})")
else:
    print("차원 불일치!")
    print(f"  - Encoder 차원: {encoder_dim}")
    print(f"  - Decoder 차원: {decoder_dim}")
    print(f"  - 차이: {abs(encoder_dim - decoder_dim)}")