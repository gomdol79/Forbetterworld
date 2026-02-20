"""
nanoGPT 모델 코드 - 한글 주석 버전
========================================
GPT (Generative Pre-trained Transformer) 구현

참조:
1) OpenAI의 공식 GPT-2 TensorFlow 구현
2) Hugging Face Transformers PyTorch 구현

핵심 개념 설명:
- Token Embedding: 단어를 벡터로 변환 (각 단어를 고유한 숫자로 표현)
- Position Embedding: 단어의 위치 정보 추가 (문장에서 순서 정보)
- Self-Attention: 문장의 각 단어가 다른 단어들과 얼마나 관련 있는지 계산
- Causal Mask: 미래의 정보를 보지 못하도록阻挡 (왼쪽 방향만 참고)
- Residual Connection: Gradient 소실 방지, 학습 안정성 향상
- Layer Normalization: 학습 안정화와 수렴 속도 향상
- MLP (Feed Forward): 비선형 변환으로 표현력增强
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """
    Layer Normalization (층 정규화)
    ========================================
    물리적 의미: 
    - 각 층의 입력값들을 평균 0, 분산 1로 정규화
    - 학습 중数值 불안정성防止 (gradient exploding/vanishing 문제 해결)
    - 입력 범위를 일정하게 유지하여 학습 안정화
    
    선택적 bias: PyTorch의 LayerNorm은 bias를 지원하지 않아 직접 구현
    """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))  # gamma: 학습 가능한尺度 파라미터
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # beta: 학습 가능한 이동 파라미터

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    causal Self-Attention (인과적 자기 어텐션)
    ========================================
    물리적 의미:
    - 입력 시퀀스의 각 위치가 이전 위치들과 어떤 관계를 갖는지 계산
    - Q (Query): "내가 찾고 싶은 정보"
    - K (Key): "내가 가지고 있는 정보의 라벨"
    - V (Value): "실제 정보 내용"
    
    어텐션 계산: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    causal mask의 의미:
    - 현재 위치에서 미래 위치(오른쪽)의 정보를 참조하지 않도록遮斷
    - "나는 어제 것을 볼 수 있지만 오늘은 볼 수 없다"는 時系列 특성 반영
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Q, K, V를 하나의 선형 변환으로 동시에 계산 (효율성)
        # 입력: n_embd 차원 → 출력: 3 * n_embd (Q, K, V 각 하나씩)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # 어텐션 결과 projection (다시 n_embd 차원으로)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # 정규화 dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head  # 어텐션 헤드 수 (병렬 어텐션)
        self.n_embd = config.n_embd   # 임베딩 차원
        self.dropout = config.dropout
        
        # Flash Attention: GPU에서 효율적인 어텐션 계산 (PyTorch 2.0+)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("경고:低速 어텐션 사용中. Flash Attention은 PyTorch 2.0 이상이 필요합니다.")
            
            # Causal Mask 생성: 삼각형 형태 (미래 정보遮斷)
            # torch.tril: 아래 삼각행렬 (대각선 포함)
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        """
        순전파 (Forward Pass)
        -------------------
        입력 x: (B, T, C) - 배치크기, 시퀀스길이, 임베딩 차원
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # Q, K, V 계산: 하나의 선형층에서 세 값으로 분할
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # 헤드별로 분리: (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
        # head_dim = n_embd / n_head: 각 헤드의 차원수
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Causal Self-Attention 계산
        if self.flash:
            # Flash Attention 사용 (효율적 CUDA 커널)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=True
            )
        else:
            # 수동 어텐션 계산
            # 1. Q와 K의 유사도 계산 (스케일링 포함)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            # 2. Causal Mask 적용 (미래 위치 = -inf → 확률 0)
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            
            # 3. Softmax로 확률 변환
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            
            # 4. 가중합으로 최종 값 계산
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # 헤드 결과 결합: (B, nh, T, hs) → (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 출력 projection + dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    MLP (Multi-Layer Perceptron) - 피드포워드 네트워크
    ========================================
    물리적 의미:
    - Self-Attention 후 비선형 변환 수행
    - 4배 확장: 임베딩 차원 → 4배 → 다시 축소
    - GELU 활성화 함수: ReLU보다 부드러운 비선형성
    
    수식: MLP(x) = GELU(xW_1 + b_1)W_2 + b_2
    """
    def __init__(self, config):
        super().__init__()
        # 입력 → 4배 확장 (표현력增强)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        
        # GELU 활성화 함수 (Gaussian Error Linear Unit)
        # ReLU보다 자연스러운 그래프 형태, 더 나은 성능
        self.gelu = nn.GELU()
        
        # 다시 원래 차원으로 축소
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)    # 확장
        x = self.gelu(x)    # 비선형 활성화
        x = self.c_proj(x)  # 축소
        x = self.dropout(x) # 정규화
        return x


class Block(nn.Module):
    """
    Transformer Block (트랜스포머 블록)
    ========================================
    물리적 의미:
    - Transformer의 핵심 단위
    - Attention + MLP + Residual Connection + LayerNorm
    
    수식:
    x = x + Attention(LayerNorm(x))  # 잔차 연결 (Gradient 흐름 개선)
    x = x + MLP(LayerNorm(x))        # 잔차 연결
    
    Residual Connection (잔차 연결)의 의미:
    - Gradient가 직접 흐를 수 있는 경로 제공 (기울기 소실防止)
    - 학습이 깊어져도 정보 손실 최소화
    - 각 층이 입력 정보를 직접 학습할 수 있음
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)  # Attention 전 정규화
        self.attn = CausalSelfAttention(config)                  # 자기 어텐션
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)  # MLP 전 정규화
        self.mlp = MLP(config)                                   # 피드포워드 네트워크

    def forward(self, x):
        # Attention with Residual Connection
        x = x + self.attn(self.ln_1(x))
        # MLP with Residual Connection
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    """
    GPT 모델 하이퍼파라미터 설정
    ========================================
    기본값: GPT-2 small (124M 파라미터)
    """
    block_size: int = 1024       # 입력 시퀀스 최대 길이
    vocab_size: int = 50304      # 어휘 크기 (GPT-2: 50257, 64의 배수로 패딩)
    n_layer: int = 12            # Transformer 블록 수 (Layer)
    n_head: int = 12             # 어텐션 헤드 수
    n_embd: int = 768            # 임베딩 차원 (각 토큰의 벡터 크기)
    dropout: float = 0.0         # 드롭아웃 비율 (정규화)
    bias: bool = True            # Linear/LayerNorm에서 bias 사용 여부


class GPT(nn.Module):
    """
    GPT 모델 전체 정의
    ========================================
    구조:
    1. Token Embedding (wte): 토큰 → 임베딩 벡터
    2. Position Embedding (wpe): 위치 → 임베딩 벡터
    3. Transformer Blocks: n_layer개의 블록 통과
    4. LayerNorm: 최종 정규화
    5. LM Head: 다음 토큰 예측 ( vocabulaire 크기 출력)
    
    Weight Tying:
    - 입력 임베딩과 출력 임베딩 가중치 공유
    - 파라미터 수 감소, 학습 효율 향상
    """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Transformer 모델 정의
        self.transformer = nn.ModuleDict(dict(
            # Token Embedding: 각 토큰을 n_embd 차원 벡터로 변환
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            
            # Position Embedding: 각 위치를 n_embd 차원 벡터로 변환
            wpe = nn.Embedding(config.block_size, config.n_embd),
            
            # Dropout: 정규화
            drop = nn.Dropout(config.dropout),
            
            # Transformer 블록들 (n_layer개)
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            
            # 최종 LayerNorm
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # Language Model Head: 임베딩 → 어휘 크기 변환
        # False bias로 Weight Tying 효과 (입력 임베딩과 공유)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight Tying: 입력 임베딩과 출력 임베딩 가중치 공유
        self.transformer.wte.weight = self.lm_head.weight

        # 가중치 초기화
        self.apply(self._init_weights)
        
        # Residual projection에 특별한 초기화 (GPT-2 논문 참조)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # 파라미터 수 출력
        print("파라미터 수: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        모델 파라미터 수 계산
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """
        가중치 초기화
        - Linear: 평균 0, 표준편차 0.02의 정규분포
        - Embedding: 평균 0, 표준편차 0.02의 정규분포
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        순전파 (Forward Pass)
        -------------------
        idx: 입력 토큰 인덱스 (B, T)
        targets: 목표 토큰 (학습 시 사용)
        
        처리 과정:
        1. 토큰을 임베딩으로 변환
        2. 위치 정보 추가
        3. Transformer 블록 통과
        4. 다음 토큰 예측
        """
        device = idx.device
        b, t = idx.size()
        
        # 시퀀스 길이가 block_size를 넘으면 오류
        assert t <= self.config.block_size, f"시퀀스 길이 {t}가 block size {self.config.block_size}를 초과"
        
        # 위치 정보: 0, 1, 2, ..., t-1
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # 토큰 임베딩: (B, T) → (B, T, n_embd)
        tok_emb = self.transformer.wte(idx)
        
        # 위치 임베딩: (T) → (T, n_embd)
        pos_emb = self.transformer.wpe(pos)
        
        # 임베딩 결합 + Dropout
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Transformer 블록 통과
        for block in self.transformer.h:
            x = block(x)
        
        # 최종 정규화
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 학습 모드: 손실 계산
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 추론 모드: 마지막 위치만 예측 (효율성)
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        텍스트 생성 (Auto-regressive)
        -------------------
        idx: 조건 시퀀스
        max_new_tokens: 생성할 토큰 수
        temperature: 생성 온도 (높을수록 무작위성 증가)
        top_k: 상위 k개 토큰만 고려
        
        방법:
        1. 현재 시퀀스로 다음 토큰 예측
        2. 예측된 토큰을 시퀀스에 추가
        3. 반복
        """
        for _ in range(max_new_tokens):
            # 시퀀스가 너무 길면 block_size로 자름
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # 다음 토큰 예측
            logits, _ = self(idx_cond)
            
            # 마지막 위치의 로짓만 사용
            logits = logits[:, -1, :] / temperature
            
            # Top-k 필터링
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # 확률 변환
            probs = F.softmax(logits, dim=-1)
            
            # 샘플링
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 시퀀스에 추가
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
