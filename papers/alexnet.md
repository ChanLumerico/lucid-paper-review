# [AlexNet] ImageNet Classification with Deep Convolutional Neural Networks

이 글은 Krizhevsky, Sutskever, Hinton(2012)의 _“ImageNet Classification with Deep Convolutional Neural Networks”_ 를 “요약”하기보다, 원 논문의 전개(문제 제기 → 데이터셋/평가 → 아키텍처/수식 → 학습/정규화 → 실험 결과 → 해석)를 최대한 그대로 따라가며 한국어로 촘촘히 풀어쓴 상세 리뷰다.

AlexNet이 유명한 이유는 단순히 “큰 CNN을 돌렸더니 성능이 좋았다”가 아니다. 논문은 당시의 컴퓨터 비전 학습 관행에서 **(1) 비포화 활성화(ReLU), (2) GPU 기반 고효율 합성곱 구현, (3) 데이터 증강과 dropout 중심의 과적합 억제, (4) 대규모 데이터셋(ImageNet)에서의 끝단 성능**을 하나의 설계로 묶어 “현실적으로 큰 모델을 끝까지 학습시키는 방법”을 제시한다. 특히 ILSVRC-2010/2012에서 **top-5 error를 큰 폭으로 낮춘** 것이 이 논문의 정량적 임팩트다.

---

## 1️⃣ Introduction

### 🔹 문제의식: 대규모 데이터·고해상도 이미지에서의 일반화
논문은 ImageNet처럼 **클래스 수가 많고(1000 클래스), 이미지가 고해상도이며(변형이 큼), 데이터 규모가 큰(약 120만 학습 샘플)** 환경에서 기존 기법들이 충분히 잘 작동하지 않는다는 점에서 출발한다. 이전 SOTA가 존재하더라도, 데이터 규모가 커질수록 “모델 용량이 충분히 크고, 최적화가 가능하며, 과적합을 제어할 수 있는” 학습 시스템이 필요해진다.

여기서 AlexNet이 제시하는 방향은 명확하다.

- 특징을 손으로 설계하는 대신, **심층 CNN이 이미지에서 표현을 자동으로 학습**하도록 한다.
- 다만 이를 가능하게 하려면, (1) 계산량과 메모리를 감당할 수 있어야 하고, (2) 과적합을 억제할 수 있어야 하며, (3) 학습이 수렴할 만큼 최적화가 잘 되어야 한다.

### 🔸 핵심 기여(논문 Abstract 기반)
논문이 스스로 정리하는 구성 요소를 먼저 명시해두면 이후 섹션이 깔끔해진다.

1) **5개의 합성곱 층 + 3개의 완전연결 층 + 1000-way softmax**로 구성된 대규모 CNN(약 60M 파라미터, 650k 뉴런).  
2) 학습 가속을 위한 **비포화 활성화(ReLU)** 와 **GPU 최적화 합성곱 구현**.  
3) 완전연결 층의 과적합 억제를 위한 **dropout**.  
4) ILSVRC에서의 성능: 2010 test에서 top-1 37.5%, top-5 17.0% error(에러율 기준), 2012에서도 우승 수준 top-5 error.

논문은 이후 섹션에서 이 기여들을 “데이터셋/평가 정의 → 아키텍처 설계 → 정규화/증강 → 학습 디테일 → 결과 해석” 순으로 확장한다.

---

## 2️⃣ The Dataset

### 🔹 ILSVRC-2010: 규모, 클래스, 평가 지표와 크롭 전략
논문이 다루는 ILSVRC-2010(ImageNet Large Scale Visual Recognition Challenge)은

- 1000개의 클래스,
- 약 120만 개의 학습 이미지,
- 5만 개의 검증 이미지,
- 15만 개의 테스트 이미지(라벨 비공개)

로 구성된다.

평가 지표는 **top-1 error**와 **top-5 error**다.

- top-1 error: 모델이 가장 높은 확률로 예측한 클래스가 정답이 아니면 오류.  
- top-5 error: 모델이 확률 상위 5개로 제시한 클래스들 안에 정답이 없으면 오류.

ImageNet처럼 클래스가 1000개로 많고 클래스 간 시각적 유사성이 존재하는 환경에서는, top-5가 “모델이 의미 있게 후보를 좁혔는지”를 보여주는 현실적인 지표로 자주 쓰인다.

논문은 이미지 입력을 네트워크에 넣을 때 “한 장을 그대로” 넣지 않는다. 데이터 증강과 테스트-time averaging을 위해 **크롭(crop)** 을 적극적으로 활용한다(자세한 방법은 4.1에서).

핵심만 먼저 잡으면:

- 학습 시: 원본에서 일정 크기(논문에서는 224×224) 패치를 랜덤하게 뽑아 학습 샘플로 사용한다.  
- 테스트 시: 여러 위치의 크롭(예: 코너 + 중앙)과 좌우반전까지 포함한 여러 뷰를 평가하고, 예측을 평균하여 최종 확률을 만든다.

이 전략은 “한 이미지 안에서도 객체 위치/스케일이 다양하다”는 사실을 데이터 증강과 앙상블(평균) 형태로 반영하는 방식이다.

---

## 3️⃣ The Architecture

논문의 3장은 AlexNet의 핵심이다. 단순히 “층을 쌓았다”가 아니라, **큰 모델을 학습 가능하게 만드는 설계 요소**들을 하나씩 분해해 설명한다. 이 섹션을 읽을 때는 “각 요소가 (1) 수렴 속도, (2) 일반화, (3) 계산 효율에 어떤 영향을 주는가”를 계속 따라가는 게 좋다.

### 🔸 3.1 ReLU Nonlinearity: 왜 tanh/sigmoid 대신 ReLU인가
논문은 비선형 활성화로
$$
f(x) = \max(0, x)
$$
를 사용하는 **ReLU(Rectified Linear Unit)** 를 채택한다.

수식적으로는 간단하지만, 논문이 강조하는 포인트는 “학습 속도”다.

- sigmoid/tanh 같은 포화(saturating) 비선형은 입력이 커지면 기울기가 0에 가까워져 학습이 느려질 수 있다.  
- ReLU는 양의 구간에서 기울기가 1로 유지되어, 깊은 네트워크에서도 기울기 소실 문제를 완화하고 수렴을 빠르게 만든다.

논문은 동일한 아키텍처에서 tanh 대비 ReLU가 훨씬 빠르게 학습 손실을 낮추는 학습 곡선(훈련 반복 수 대비 에러)을 제시한다.

(Fig. 1: CIFAR-10에서 ReLU를 쓰는 4-layer CNN과 tanh를 쓰는 동일 구조 CNN의 학습 곡선을 비교한다. 같은 반복 수에서 ReLU가 훨씬 빨리 낮은 training error에 도달함을 보여준다.)

### 🔹 3.2 Training at Scale: Multi-GPU와 Overlapping Pooling
AlexNet이 당시 이슈였던 이유 중 하나는 모델이 매우 크고 합성곱 계산이 무겁다는 점이다. 논문은 두 GPU에 네트워크를 분할해 올려 학습한다.

여기서 중요한 건 단순 병렬화가 아니라, **분할 방식이 아키텍처에 직접 반영**된다는 점이다.

- 어떤 층의 feature map들은 GPU1에, 나머지는 GPU2에 둔다.  
- 특정 합성곱 층은 같은 GPU에 있는 이전 층의 feature map만 연결하고, 일부 층은 두 GPU 사이를 “교차(cross-GPU)”로 연결한다.

이 분할은 메모리 제약을 넘기기 위한 실용적 설계이며, 동시에 부분 연결(partial connectivity)을 만들어 모델의 대칭성을 깨는 효과도 갖는다(이 점은 Fig. 2의 구조 설명에서 더 명확해진다).

또 하나의 디테일은 **overlapping pooling**이다. 논문은 max pooling에서 윈도우 크기와 stride를 다르게 둬서 **인접 pooling window가 겹치도록** 설정한다(예: kernel size 3, stride 2). 직관은 “너무 거친 다운샘플링으로 정보가 버려지는 것을 완화”하고, 경험적으로 error를 소폭 낮추는 데 도움이 된다는 주장이다.

### 🔸 3.3 Local Response Normalization (LRN): “lateral inhibition”을 흉내낸 정규화
논문은 ReLU를 쓰면 활성값이 커질 수 있고, 인접 채널 간 경쟁을 유도해 일반화에 도움을 주기 위해 **LRN**을 도입한다. LRN은 같은 공간 위치 $(x, y)$에서 채널 방향으로 주변 activation의 제곱합을 이용해 정규화한다.

논문 표기대로, $a_{x,y}^i$를 (정규화 전) 채널 $i$의 activation이라고 하면, 정규화 후 $b_{x,y}^i$는
$$
b_{x,y}^i
=
\frac{a_{x,y}^i}{\left(k + \alpha \sum_{j=\max(0, i-n/2)}^{\min(N-1, i+n/2)} (a_{x,y}^j)^2\right)^{\beta}}.
$$
여기서

- $N$: 총 채널 수(feature map 수),
- $n$: 정규화에 포함할 이웃 채널의 폭(윈도우 크기),
- $k, \alpha, \beta$: 하이퍼파라미터

이다.

이 식의 의미를 직관적으로 보면:

- 분모가 커지면(주변 채널 활성들이 크면) 현재 채널의 값이 상대적으로 눌린다.  
- 특정 위치에서 “특히 강하게 반응하는 채널”이 있으면, 그 주변 채널은 상대적으로 억제된다(채널 간 경쟁).

논문은 LRN이 top-1/top-5 error를 소폭 낮추는 경험적 이득이 있다고 보고한다(절대 필수라기보다, 당시의 최적 조합에 포함된 컴포넌트로 이해하면 좋다).

### 🔹 3.5 Overall Architecture: 5 conv + 3 fc + softmax의 구체
이제 전체 네트워크를 층 단위로 정리한다. 논문은 Fig. 2로 구조를 요약한다.

(Fig. 2: AlexNet 아키텍처를 도식화한다. 2-GPU 분할(두 컬럼)과, 각 층의 kernel size/stride, feature map 수, 그리고 FC 층(4096-4096-1000) 구조를 함께 보여준다.)

논문 텍스트를 따라 “스테이지 단위”로 구조를 정리하면 아래 표처럼 한 번에 보는 게 깔끔하다(핵심 하이퍼파라미터만 추려서 적는다).

| Stage | Type | Kernel/Stride/Pad | Channels/Units | Notes |
|---|---|---:|---:|---|
| Input | Crop | 224×224 (from 256 resize) | 3 | 학습: 랜덤 크롭/플립, 테스트: 여러 크롭 평균(4.1) |
| Conv1 | Conv + ReLU | 11×11 / s4 / p0 | 96 | 뒤에 LRN(3.3) 및 MaxPool |
| Pool1 | MaxPool | 3×3 / s2 | - | Overlapping pooling(3.2) |
| Conv2 | Conv + ReLU | 5×5 / s1 / p2 | 256 | 뒤에 LRN 및 MaxPool |
| Pool2 | MaxPool | 3×3 / s2 | - | Overlapping pooling(3.2) |
| Conv3 | Conv + ReLU | 3×3 / s1 / p1 | 384 | - |
| Conv4 | Conv + ReLU | 3×3 / s1 / p1 | 384 | - |
| Conv5 | Conv + ReLU | 3×3 / s1 / p1 | 256 | 뒤에 MaxPool |
| Pool5 | MaxPool | 3×3 / s2 | - | - |
| FC6 | FC + ReLU + Dropout | - | 4096 | dropout으로 FC 과적합 억제(4.2) |
| FC7 | FC + ReLU + Dropout | - | 4096 | dropout으로 FC 과적합 억제(4.2) |
| FC8 | FC | - | 1000 | 마지막 로짓 |
| Output | Softmax | - | 1000 | 교차엔트로피로 학습(5) |

이 구조는 지금 관점에서는 “기본 CNN 스택”처럼 보일 수 있지만, 당시엔 (1) ReLU 채택, (2) 대규모 입력/대규모 클래스, (3) 큰 FC 층(4096)과 dropout, (4) GPU 분할까지 포함한 “학습 가능한 규모”가 중요한 포인트였다.

---

## 4️⃣ Reducing Overfitting

논문은 모델이 큰 만큼 과적합을 강하게 경계한다. 특히 FC 층이 파라미터 수 대부분을 차지하므로, 정규화/증강이 없으면 training error는 내려가도 test error가 크게 나빠질 수 있다. AlexNet은 과적합 억제를 크게 두 축으로 해결한다: **데이터 증강**과 **dropout**.

### 🔸 4.1 Data Augmentation: 라벨 보존 변환을 학습 데이터로
논문이 말하는 “가장 효과적이면서 가장 간단한” 과적합 억제는 데이터 증강이다. 두 종류를 쓴다.

#### (1) 기하학적 증강: 랜덤 크롭 + 좌우 반전
입력을 256×256으로 리사이즈한 뒤, 학습 시에는

- 무작위 위치에서 224×224 패치를 뽑고,
- 그 패치를 좌우 반전할 수도 있도록 한다.

이렇게 하면 한 이미지에서 많은 변형 샘플이 만들어진다. 논문은 이 방법이 학습 데이터를 실질적으로 크게 늘리고(특히 위치 변형), 과적합을 눈에 띄게 줄인다고 설명한다.

테스트 시에는 랜덤이 아니라 고정된 여러 뷰(예: 네 코너 + 중앙, 그리고 좌우 반전)를 평가해 예측을 평균한다. 즉, 테스트도 작은 앙상블처럼 동작한다.

#### (2) 광학적 증강: RGB 채널의 조명 변화(색상 jitter)
두 번째는 조명/색상 변화에 대한 불변성을 주기 위한 색상 증강이다. 논문은 학습 이미지의 RGB 픽셀들에 대해 PCA를 계산하고, 주성분 방향으로 작은 노이즈를 더한다.

표기를 정리하면:

- RGB 3차원 데이터의 공분산을 PCA로 분해해 고유벡터 $p_1, p_2, p_3$와 고유값 $\lambda_1, \lambda_2, \lambda_3$를 얻는다.  
- 각 학습 이미지에 대해 $\alpha_i \sim \mathcal{N}(0, \sigma)$ 를 샘플링하고, 다음을 픽셀에 더한다:
$$
\Delta I = \sum_{i=1}^{3} p_i \, \alpha_i \, \lambda_i.
$$
이렇게 하면 “전체 이미지의 조명/색조가 약간 변한” 버전이 만들어진다.

이 증강의 직관은 “같은 객체라도 조명에 따라 RGB 분포가 달라질 수 있으니, 모델이 그 변화에 덜 민감해지도록” 학습 분포를 넓히는 것이다.

### 🔹 4.2 Dropout: co-adaptation을 막는 확률적 정규화
dropout은 논문이 “최근 개발된 정규화”라고 표현하는 핵심 요소다. FC 층 뉴런을 학습 중 확률적으로 꺼서(co-adaptation 방지), 특정 특징 조합에 과하게 의존하는 것을 막는다.

학습 관점에서 dropout은 다음 성질을 갖는다.

- 매 iteration마다 서로 다른 “부분 네트워크(subnetwork)”를 샘플링해 학습하는 셈이다.  
- 테스트 시에는 전체 네트워크를 쓰되, dropout으로 사라졌던 기대값을 보정하기 위해 스케일링을 적용한다(프레임워크마다 구현 방식은 다르지만, 핵심은 “학습 시의 랜덤 마스크의 기대 효과를 테스트에서 맞춘다”는 것).

논문은 dropout이 없으면 FC 층이 빠르게 과적합하고, dropout이 이를 크게 줄인다고 설명한다.

---

## 5️⃣ Details of Learning

이 섹션은 AlexNet이 단순히 “구조가 좋아서”가 아니라, **학습 설정 자체가 대규모 학습을 가능하게 했음**을 보여주는 파트다. 구현/재현 관점에서 중요하므로, 변수들을 정리해가며 읽자.

### 🔸 목적 함수와 최적화: softmax/교차엔트로피 + SGD(mom.)/weight decay
마지막 층이 1000-way softmax이므로, 입력 $x$에 대해 로짓(logit) 벡터를 $z(x) \in \mathbb{R}^{1000}$라 하고, softmax 확률을
$$
p_k(x) = \frac{\exp(z_k(x))}{\sum_{j=1}^{1000} \exp(z_j(x))}
$$
로 두면, 정답 클래스 $y$에 대한 음의 로그우도(교차엔트로피) 손실은
$$
\mathcal{L}(x, y) = -\log p_y(x)
$$
가 된다. 전체 데이터셋에 대해 평균을 취하면 경험 위험(empirical risk)이다.

이 손실을 최소화하기 위해 논문은 SGD(모멘텀 포함)와 가중치 감쇠(weight decay)를 사용한다.

논문은 미니배치 SGD를 사용하며, 모멘텀과 weight decay를 포함한다. 일반적인 형태로 적으면:

$$
v_{t+1} = \mu v_t - \eta \left(\nabla_W \mathcal{L}_t(W_t) + \lambda W_t\right),
$$
$$
W_{t+1} = W_t + v_{t+1},
$$

- $W$: 파라미터,
- $v$: 모멘텀 버퍼,
- $\mu$: momentum 계수(논문은 0.9),
- $\eta$: learning rate(초기값 후 점차 감소),
- $\lambda$: weight decay 계수(논문은 $5\times 10^{-4}$)

로 이해하면 된다.

논문은 또한 특정 층(특히 일부 bias)에 대해 학습률을 다르게 두거나 초기화에 대한 경험적 팁을 적는다. 핵심은 “큰 모델에서도 안정적으로 학습되도록” 수치적 디테일을 맞춘다는 점이다.

### 🔹 학습 파이프라인(의사코드)
논문은 텍스트로 학습 설정을 풀어 쓰지만, 이를 절차로 정리하면 다음처럼 된다.

```pseudocode
Algorithm 1: AlexNet Training (paper-level procedure)
Input: training set D = {(I, y)}, initial parameters W
Hyperparams: batch size B, lr η0, momentum μ, weight decay λ
Augment: random crop/flip, PCA color jitter

1: initialize momentum buffer v ← 0
2: for epoch = 1..E do
3:     for each minibatch {(I_b, y_b)}_{b=1..B} do
4:         for each image I_b do
5:             I'_b ← RandomCropAndFlip(Resize256(I_b))   # 224×224
6:             I''_b ← ColorJitterPCA(I'_b)               # optional
7:             x_b ← Preprocess(I''_b)                    # mean subtraction, etc.
8:         end for
9:         z ← Forward(W, {x_b})
10:        L ← CrossEntropySoftmax(z, {y_b})
11:        g ← Backward(W, L)
12:        v ← μ v − η (g + λ W)
13:        W ← W + v
14:    end for
15:    η ← Schedule(η)   # reduce when validation plateaus
16: end for
17: return W
```

이 의사코드는 “뼈대”만 보여준다. AlexNet의 실제 기여는 여기에 (1) ReLU로 빠른 수렴, (2) dropout으로 FC 과적합 억제, (3) GPU 최적화로 학습 시간을 현실화하는 요소들이 결합된다는 점이다.

---

## 6️⃣ Results

### 🔸 정량 결과: ILSVRC-2010/2012 성능
논문은 ILSVRC-2010 test set에서 top-1 37.5%, top-5 17.0% error를 보고한다. 또한 당시 SOTA 대비 큰 폭의 개선임을 강조한다.

(Table 1: ILSVRC-2010에서 AlexNet과 기존 방법들의 top-1/top-5 error를 비교한다. 논문은 AlexNet이 가장 낮은 error를 달성했음을 수치로 보여준다.)

또한 2012 대회에서도 이 모델의 변형을 제출해 top-5 error에서 우승 수준 성능을 달성했다고 언급하며, 같은 대회 상위권 모델들과의 비교를 표로 제시한다.

(Table 2: ILSVRC-2012 validation/test에 대한 여러 방법들의 error rate를 비교한다. AlexNet 계열 모델이 당시 가장 경쟁력 있는 성능을 보였음을 수치로 제시한다.)

### 🔹 정성 분석: 무엇을 학습했는가
논문은 정량 지표만 아니라, 모델이 첫 층에서 어떤 필터를 학습했는지 시각적으로 보여준다.

(Fig. 3: 첫 합성곱 층에서 학습된 96개의 11×11 커널을 시각화한다. 색상/방향/에지/블롭 같은 저수준 특징들이 나타남을 보여준다.)

또한 마지막 은닉층의 특징 공간에서 이미지 간 “유사도”가 의미 있게 형성되었는지를 예시로 보여준다.

(Fig. 4: (Left) 테스트 이미지 몇 장과 모델이 높은 확률로 제시한 상위 라벨들을 보여준다. (Right) 마지막 은닉층 표현에서 가까운 이웃(nearest neighbors) 이미지들을 보여주며, 의미적으로 유사한 이미지들이 가까이 모임을 시각적으로 확인한다.)

이 파트의 요지는 “모델이 단순히 훈련 데이터를 암기한 것이 아니라, 일반화 가능한 표현 공간을 학습했다”는 정성적 근거를 보이려는 것이다.

---

## 7️⃣ Discussion

논문은 마지막에 “왜 이게 되었는가”를 정리한다. 현대 관점에서 다시 읽어도, AlexNet이 성공한 이유는 단일 요소가 아니라 **조합**이다.

1) **대규모 데이터셋**: 120만 장 규모의 데이터가 큰 모델 용량을 뒷받침했다.  
2) **학습 가속(최적화/구현)**: ReLU와 GPU 최적화가 없었다면, 동일한 규모의 실험을 현실적인 시간에 끝내기 어려웠다.  
3) **일반화 제어**: 데이터 증강과 dropout이 없었다면, 특히 FC 층이 과적합으로 무너졌을 가능성이 크다.  
4) **아키텍처의 적절한 귀납편향(inductive bias)**: 합성곱/풀링이 이미지의 국소 구조와 이동 변형에 맞는 prior를 제공한다.

이 논문은 “심층 CNN이 대규모 인식에서 통한다”는 사실을 결과로 보여주는 동시에, 그걸 가능하게 만드는 실용적인 레시피를 제공했다는 점에서 의미가 크다.

---

## 💡 해당 논문의 시사점과 한계 혹은 의의

AlexNet은 이후 CNN 설계의 기본이 되는 요소들을 대중화했다.

- ReLU의 표준화(학습 속도/안정성).  
- 데이터 증강과 dropout의 결합으로 “큰 FC”를 다루는 방법.  
- 테스트에서 multi-crop 평균을 쓰는 관행.  
- GPU가 사실상 필수인 규모의 모델을 “현실적으로” 구현/학습.

한편 시간이 지나며 AlexNet의 일부 구성은 다른 기법으로 대체되거나 단순화되었다.

- LRN은 이후 BatchNorm 등으로 대체되거나 불필요해지는 경우가 많았다.  
- 큰 FC(4096×4096)는 파라미터/메모리 부담이 크고, 이후에는 GAP(Global Average Pooling) 중심 구조로 이동하는 흐름이 생겼다.  
- 2-GPU 분할은 당시 제약(메모리/속도)에 대한 해법이었지만, 현대 프레임워크에서는 더 다양한 병렬화/분산 학습 기법이 존재한다.

그럼에도 AlexNet의 “조합 설계”는 여전히 역사적으로 중요하다. “큰 CNN을 끝까지 학습시키기 위해 무엇이 동시에 필요했는가(최적화/구현/정규화/증강)”를 논문 단위로 한 번에 묶어 보여준다는 점에서, 이후의 거의 모든 CNN 계열 연구를 이해하는 출발점 역할을 한다고 느꼈다.

---

## 👨🏻‍💻 AlexNet Lucid 구현

아래는 `lucid/models/`에 존재하는 AlexNet 구현을 실제로 읽고, 논문 구조와 1:1로 대응시켜 해설하는 파트다. 여기서는 논문의 “정확히 동일한 구현”을 재현하기보다, Lucid 코드가 논문에서 제시한 설계 요소들을 어떤 방식으로 담고 있는지(또는 생략했는지)를 코드 단위로 확인한다.

### 0️⃣ 구현 위치와 엔트리포인트
Lucid의 AlexNet은 `lucid/models/imgclf/alex.py`에 구현되어 있고, 외부에서 불러올 수 있도록 `alexnet()` 팩토리 함수가 `@register_model`로 등록되어 있다.

```python
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = ["AlexNet", "alexnet"]


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


@register_model
def alexnet(num_classes: int = 1000, **kwargs) -> AlexNet:
    return AlexNet(num_classes, **kwargs)
```

이 파일에는 크게 세 덩어리가 있다.

1) `AlexNet` 클래스의 `__init__`: 합성곱 블록(`self.conv`) + 풀링/평균풀(`self.avgpool`) + 분류기(`self.fc`) 구성  
2) `AlexNet.forward`: conv → avgpool → flatten → fc 순서의 전방 계산  
3) `alexnet()` 팩토리: `register_model`로 모델 레지스트리에 등록

이제 이 구조를 논문 섹션과 연결해보자.

### 1️⃣ 아키텍처 구성요소 매핑: `conv`/`avgpool`/`fc`/`forward`
Lucid 구현의 `self.conv`는 논문의 “conv + ReLU + (일부에 pooling)” 패턴을 직접 반영한다.

```python
self.conv = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(64, 192, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(192, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(256, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
)
```

- `Conv2d(3, 64, kernel_size=11, stride=4, padding=2)`는 논문의 첫 층(11×11, stride 4)을 반영한다.  
  - 다만 논문은 첫 층 출력 채널을 96으로 두는데, Lucid는 64로 설정되어 있다. 즉 “AlexNet 스타일의 첫 층”이지만 채널 수는 줄인 변형이다.  
- 중간의 `Conv2d(in_channels, out_channels, kernel_size=5, padding=2)`, `Conv2d(in_channels, out_channels, kernel_size=3, padding=1)` 반복은 논문의 5×5, 3×3 커널 설계를 그대로 따른다.  
- `MaxPool2d(kernel_size=3, stride=2)`는 논문의 3×3, stride 2 풀링을 반영한다(오버랩 여부는 앞 단계 feature map 크기와 stride로 결정).

논문에서 강조한 ReLU(3.1)와 pooling(3.4) 설계 요소는 Lucid 코드에 명확히 드러난다.

반면 논문에 있던 LRN(3.3)은 `self.conv` 내부에 등장하지 않는다. 즉 Lucid의 AlexNet은 LRN을 생략한 변형이다(현대 구현들에서도 흔한 선택).

또한 Lucid는 `self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))`로 conv 출력의 공간 크기를 **항상 6×6으로 맞춘다**. 논문 AlexNet은 특정 입력 크기(224/227 계열)에서 마지막 conv 출력이 6×6이 되도록 “입력 크기-아키텍처”를 맞춘 반면, Lucid는 adaptive pooling으로 입력 크기 변화에 더 유연하게 대응한다.

```python
self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
```

분류기(`self.fc`)는 논문의 FC6/FC7/FC8 패턴을 따르며, FC 앞단에 dropout을 배치한다.

- `Dropout → Linear(256*6*6, 4096) → ReLU`  
- `Dropout → Linear(4096, 4096) → ReLU`  
- `Linear(4096, num_classes)`

```python
self.fc = nn.Sequential(
    nn.Dropout(),
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Linear(4096, num_classes),
)
```

마지막으로 `forward`는 `conv → avgpool → flatten → fc`를 그대로 수행한다. 즉 논문 Fig. 2의 큰 흐름(특징 추출 → 고정 차원화 → 분류)을 코드에서 1:1로 읽을 수 있다.

```python
def forward(self, x: Tensor) -> Tensor:
    x = self.conv(x)
    x = self.avgpool(x)

    x = x.reshape(x.shape[0], -1)
    x = self.fc(x)
    return x
```

그리고 파일 마지막의 팩토리 함수는 논문 설정과 같은 기본 클래스 수(1000)를 기본값으로 두고, Lucid의 모델 레지스트리에 등록한다.

```python
@register_model
def alexnet(num_classes: int = 1000, **kwargs) -> AlexNet:
    return AlexNet(num_classes, **kwargs)
```

### 2️⃣ 논문과 Lucid 구현의 차이(중요)
마지막으로, 논문을 기준으로 Lucid 구현을 읽을 때 “없거나 다른 것”을 명확히 적어두면 혼동이 줄어든다.

1) **LRN이 없다.** (논문 3.3)  
2) **2-GPU 분할/부분 연결을 구현하지 않는다.** (논문 3.2)  
3) **채널 수가 일부 다르다.** (논문은 96-256-384-384-256, Lucid는 64-192-384-256-256)  
4) **논문은 softmax를 명시하지만, Lucid 모델은 로짓만 반환**한다. 실제 softmax/교차엔트로피는 학습 코드(손실 함수)에서 적용되는 것이 일반적이므로, 모델이 로짓을 반환하는 설계는 실용적이다.

즉 Lucid의 `AlexNet`은 논문 AlexNet의 핵심 아이디어(큰 커널로 시작하는 conv 스택 + ReLU + maxpool + 큰 FC + dropout)를 유지하되, 일부 요소(LRN/멀티GPU)는 생략하고 입력 차원 처리를 `AdaptiveAvgPool`로 더 유연하게 만든 “실전형 AlexNet 변형”이라고 해석할 수 있다.

---

## ✅ 정리

AlexNet 논문은 “대규모 이미지 분류”에서 심층 CNN이 실질적으로 경쟁력을 갖는다는 사실을 ILSVRC라는 공인 벤치마크에서 강하게 입증했고, 그 성능을 가능하게 만든 구현·학습 레시피(ReLU, GPU, 데이터 증강, dropout)를 함께 제시했다. 특히 이 논문을 꼼꼼히 읽으면, 단순히 층 수를 늘리는 것만으로는 부족하고, 학습을 안정적으로 끝내기 위해 어떤 형태의 정규화/증강/최적화 디테일이 함께 필요했는지를 이해하게 된다. Lucid의 `alex.py` 구현은 이 레시피 중 “아키텍처/정규화(드롭아웃)/비선형(ReLU)/풀링” 축을 코드로 반영하며, LRN이나 멀티GPU 분할 같은 당시의 제약 기반 요소는 현대적 관점에서 과감히 생략한 형태다. 논문을 읽고 코드를 다시 보면, AlexNet이 단순한 역사적 레퍼런스가 아니라 “큰 CNN을 설계하고 학습시키는 기본 원리”를 담은 교재로 여전히 유효하다는 점이 더 또렷해진다.

#### 📄 출처
Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. “ImageNet Classification with Deep Convolutional Neural Networks.” *Advances in Neural Information Processing Systems 25 (NeurIPS 2012)*, 2012. https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf.
