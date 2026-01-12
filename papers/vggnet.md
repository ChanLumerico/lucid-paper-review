# [VGGNet] Very Deep Convolutional Networks for Large-Scale Image Recognition

이 글은 Simonyan, Zisserman(2015 ICLR)의 _"Very Deep Convolutional Networks for Large-Scale Image Recognition"_ 을 원 논문이 실제로 전개하는 흐름(문제 제기 → 아키텍처 설계 원칙 → 학습/평가 프로토콜 → 실험 결과 해석 → 결론 + 부록)을 최대한 그대로 따라가며 상세하게 분석해본 리뷰이다.

VGGNet이 유명한 이유를 "3×3 conv를 많이 쌓아서 깊게 만들었다" 정도로만 기억하면, 논문이 강조하는 핵심을 놓치기 쉽다. 이 논문은 **깊이(depth)** 라는 축을 가능한 한 깨끗하게 분리해서 실험하기 위해

- 첫 conv부터 끝까지 **3×3, stride 1** 합성곱을 기본으로 고정하고,
- **2×2, stride 2** max pooling으로만 다운샘플링하며,
- 깊이만 11 → 13 → 16 → 19 weight layers로 늘려가면서

정량적으로 얼마나 이득이 나는지 보여준다. 그리고 깊게 하면 좋아진다를 주장하는 데서 끝나지 않고, **훈련 스케일 지터링(scale jittering)**, **테스트 멀티스케일**, **dense evaluation(FC→conv 변환)**, **multi-crop**, **모델 앙상블**까지 당시 ILSVRC에서 실제로 점수를 끌어올리는 평가 프로토콜을 체계적으로 정리한다.

또한 부록에서는 localisation(단일 객체 바운딩 박스 예측), ILSVRC에서 학습한 VGG 특징을 다른 데이터셋(VOC/Caltech/Action)에 전이해 선형 SVM으로 쓰는 방식까지 다룬다. 즉 VGGNet은 아키텍처만이 아니라 평가/활용 레시피까지 포함해서 후대의 표준을 만든 논문이다.

---

## 1️⃣ 시대적 배경

### 🔹 ILSVRC를 중심으로 깊이가 다시 중요해진 배경

논문은 ConvNet이 대규모 이미지/비디오 인식에서 큰 성공을 거둔 직후의 상황에서 출발한다. ImageNet 같은 대규모 공개 데이터셋과 GPU 같은 고성능 연산 환경 덕분에, ConvNet은 이미 ILSVRC(이미지넷 대규모 인식 챌린지)에서 우승급 성능을 내고 있었다. 대표적으로 **AlexNet(Krizhevsky et al., 2012)** 이 2012년 우승을 하면서 "깊은 ConvNet + end-to-end 학습"이 주류로 급격히 이동했고, 2013년에도 Zeiler & Fergus(2013), OverFeat 계열(Sermanet et al., 2014)이 뒤를 이었다.

하지만 "성능이 좋아진다"는 사실과 별개로, 아키텍처 개선의 방향은 여러 갈래였다.

- 첫 층의 receptive field를 줄이고 stride를 줄여 **초기 단계에서 더 조밀한 샘플링**을 하거나(2013년 우승 계열),
- 멀티스케일/밀집 평가(dense evaluation)로 **테스트 시 계산을 더 써서** 정확도를 끌어올리는 방향(OverFeat 등),
- 네트워크 토폴로지를 더 복잡하게 설계하는 방향(이후 GoogLeNet 계열) 등이 있었다.

이 논문은 그중에서도 "ConvNet 아키텍처 설계에서 깊이(depth) 자체가 얼마나 중요한가?"를 핵심 질문으로 잡는다. 그리고 이 질문을 공정하게 보기 위해 **나머지 설계 요인을 가능한 고정**하고, 깊이만 체계적으로 늘려가며 비교한다.

### 🔸 논문의 핵심 주장과 기여

논문의 기여는 "3×3을 많이 쌓았더니 성능이 좋았다"가 아니라, 다음의 조합으로 읽는 게 정확하다.

1. **아키텍처 원칙 고정:** 모든 conv를 3×3, stride 1로 통일하고, pooling은 2×2, stride 2만 사용한다.
2. **깊이 축의 체계적 확장:** 11/13/16/19 weight layers까지 깊이를 늘리며 정확도 변화를 정량적으로 비교한다.
3. **스케일 처리 레시피:** 학습 시 스케일 지터링, 테스트 시 멀티스케일 + (dense / multi-crop) 평가를 조합하면 성능이 더 좋아짐을 보인다.
4. **SOTA 수준 정량 결과:** 단일 모델에서도 높은 성능을 내고, 2개 모델의 앙상블로 ILSVRC-2014 classification에서 6.8% top-5 test error까지 낮춘다.
5. **전이 학습의 실용성:** ILSVRC로 학습한 깊은 특징이 VOC/Caltech 등 다른 데이터셋에서도 선형 SVM만으로 강력하게 일반화함을 보인다.

이제부터는 논문의 섹션 흐름을 따라, "왜 3×3을 고정했는지", "깊이를 늘리면서 무엇이 달라지는지", "학습/평가 프로토콜이 정확도에 어떻게 영향을 주는지"를 수식/표/알고리즘 흐름대로 풀어보자.

---

## 2️⃣ ConvNet 구성

### 🔹 공통 레이아웃: 3×3 conv 스택 + 2×2 pooling + 3개의 FC
논문은 깊이 비교가 공정하려면 **레이아웃이 일관**해야 한다고 본다. 그래서 모든 구성(A~E)은 다음 공통 구조를 공유한다.

- 입력: `224 × 224` RGB 이미지 (3채널)
- 합성곱 블록: `3 × 3` conv, stride `1`, (공간 크기 유지를 위해) padding 사용
- 다운샘플링: `2 × 2` max pooling, stride `2`
- 분류기: FC-4096 → FC-4096 → FC-1000 → softmax
- 비선형성: ReLU (표에서는 생략하지만, 각 conv 뒤에 ReLU가 들어간다고 명시)

이 구조를 (논문이 독자가 머릿속으로 따라가게 만드는 방식 그대로) _"스테이지(stage)"_ 관점에서 쓰면:

1. 64 채널 stage
2. 128 채널 stage
3. 256 채널 stage
4. 512 채널 stage
5. 512 채널 stage

각 stage는 몇 번의 conv를 쌓느냐만 다르고, stage 사이를 maxpool로 구분한다.

### 🔸 구성 A~E: 깊이만 달리해 비교하기

논문은 (Table 1)에서 A~E 구성을 열(column)로 나란히 두고, 깊이가 증가할수록 "추가된 층"이 무엇인지 표시한다. 텍스트 리뷰에서는 표를 그대로 붙이기 어렵기 때문에, 각 구성의 conv 스택을 "채널 수 나열 + M(maxpool) 표시"로 재구성해보자.

아래 표기에서 `M`은 `2×2, stride 2 maxpool`을 뜻하고, 숫자들은 해당 stage에서의 `3×3 conv` 출력 채널 수를 뜻한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/3cf5bfff-885e-40f3-9f5a-4da74812e7ea/image.png" width="50%">
</p>

| Config | Weight layers | Conv stack (stage별) |
|---|---:|---|
| A | 11 | `[64, M, 128, M, 256, 256, M, 512, 512, M, 512, 512, M]` |
| B | 13 | `[64, 64, M, 128, 128, M, 256, 256, M, 512, 512, M, 512, 512, M]` |
| C | 16 | B에 더해 `1×1 conv`를 일부 stage에 삽입(논문 (Table 1) 참조) |
| D | 16 | `[64, 64, M, 128, 128, M, 256, 256, 256, M, 512, 512, 512, M, 512, 512, 512, M]` |
| E | 19 | `[64, 64, M, 128, 128, M, 256, 256, 256, 256, M, 512, 512, 512, 512, M, 512, 512, 512, 512, M]` |

여기서 C는 깊이를 늘리되 receptive field는 늘리지 않고 비선형성을 더 넣는 실험이다. 즉 3×3 대신 1×1을 끼워 넣어서 **공간 문맥**을 보지 않는 층이 섞이게 된다. 이 C vs D 비교가 뒤에서 중요한 해석 포인트로 쓰인다.

다음으로 (Table 2)는 각 구성의 파라미터 수를 보여주며, 깊어져도 파라미터가 폭발적으로 늘지 않는다는 관찰을 주기 위해 배치되어 있다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/405c9701-d82f-4fb1-8fd2-1f62f7fa412e/image.png" width="50%">
</p>

이 표에서 특히 흥미로운 점은 "B가 A보다 깊지만 파라미터 수가 같다(133M)"는 것이다. 깊이를 늘리면서 파라미터가 크게 늘지 않은 이유는, 큰 커널 하나로 때우는 대신 작은 커널을 여러 번 쓰는 구조가 파라미터 효율 면에서 유리하기 때문이다.

### 🔹 작은 커널을 깊게 쌓은 이유
논문이 이 섹션에서 하고 싶은 말은 크게 두 가지다.

1. 첫 층부터 큰 커널(예: 11×11, stride 4)을 쓰지 않고, **전 구간을 3×3**으로 통일한 이유  
2. 3×3을 여러 층 쌓는 것이 "표현력"과 "파라미터 효율"에서 왜 유리한지

#### 3×3 스택의 Receptive Field와 파라미터 수 비교
논문은 "pooling 없이" 3×3 conv를 여러 번 쌓으면 effective receptive field가 커진다는 점을 상기시킨다.

- `3×3` 두 번 쌓기 → effective `5×5`
- `3×3` 세 번 쌓기 → effective `7×7`

그리고 "그렇다면 7×7 하나 쓰면 되지 않나?"라는 질문에 대해, 3×3을 쌓는 게 더 낫다고 주장한다.

**첫째, 비선형성이 더 많이 들어간다.**  
7×7 한 층이면 ReLU가 한 번이지만, 3×3 세 층이면 ReLU가 세 번 들어가므로 더 "discriminative"한 결정함수를 만들 수 있다.

**둘째, 파라미터 수가 줄어든다.**  
논문은 입력/출력 채널 수가 모두 $C$라고 단순화했을 때,

- 3×3 conv 한 층의 파라미터: $3 \times 3 \times C \times C = 9C^2$
- 3×3 세 층 스택: $3 \cdot 9C^2 = 27C^2$
- 7×7 한 층: $7 \times 7 \times C \times C = 49C^2$

즉,
$$
49C^2 - 27C^2 = 22C^2
$$
만큼 7×7이 더 크고, 비율로 보면 7×7이 약 **81% 더 많은 파라미터**를 요구한다고 말한다. 논문은 이 관점을 "7×7 필터에 대한 일종의 regularisation"로도 해석한다. 즉 7×7을 자유롭게 학습하게 두는 대신, 3×3의 조합(중간 비선형 포함)으로 분해하도록 강제한다는 관점이다.

#### 1×1 conv 삽입(C 구성)의 의미
논문은 C 구성에서 1×1 conv를 넣는 이유를 "receptive field는 그대로 두고 비선형성을 늘리는 방법"으로 설명한다. 1×1 conv 자체는 공간 문맥을 보지 못하지만, 그 뒤의 ReLU가 하나 더 들어가므로 결정함수가 더 비선형적이 된다. 이 아이디어는 "Network in Network" 계열(Lin et al., 2014)과 연결된다고 언급한다.

다만 이후 실험(sect. 4.1)에서 **C가 D보다 나쁘다**는 결과가 나오며, 논문은 "비선형성 증가도 도움이 되지만, 공간 문맥을 보는 3×3 receptive field를 유지하는 것이 더 중요할 수 있다"는 해석으로 연결한다.

---

## 3️⃣ 분류 프레임워크

### 🔸 목적함수, 최적화, 정규화, 초기화
이 섹션은 VGG의 성능이 구조만 깊게 해서 나오는 것이 아니라, **학습 레시피가 꽤 구체적**이라는 점을 보여준다. 논문은 기본적으로 AlexNet 계열의 훈련 절차를 따르되, 스케일 샘플링만 다르게 한다고 말한다.

#### 목적함수: Multinomial Logistic Regression
논문은 분류 학습을 multinomial logistic regression objective를 최소화한다고 표현한다. 실무적으로는 **softmax + cross-entropy**를 떠올리면 된다.

입력 이미지 $x$에 대해 네트워크가 클래스별 로짓(logit) $z \in \mathbb{R}^K$를 출력한다고 하자($K=1000$).

softmax 확률은
$$
p(y=k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}.
$$
정답 레이블이 $y$일 때 교차 엔트로피 손실은
$$
\mathcal{L}_{CE}(x,y) = -\log p(y \mid x).
$$

이 식의 직관은 _"정답 클래스 확률을 1에 가깝게 만들도록"_ 로짓을 상대적으로 키우는 것이다. 논문 본문에서는 이 목적함수의 형태를 상세히 전개하진 않지만, **logistic regression objective**를 명시하고 이후 SGD+momentum으로 최적화한다고 설명한다.

#### 최적화: Mini-Batch SGD + Momentum
하이퍼파라미터는 다음과 같다.

- batch size: $256$
- momentum: $0.9$
- learning rate: 초기 $10^{-2}$, 검증 성능이 멈추면 10배 감소
  - 총 3번 감소
  - 370K iterations(74 epochs)에서 종료

momentum SGD는 보통 다음 업데이트로 쓸 수 있다.
$$
v_t = \mu v_{t-1} - \eta \nabla_w \mathcal{L}(w_{t-1}), \quad
w_t = w_{t-1} + v_t
$$
여기서 $\mu=0.9$, $\eta$는 learning rate다.

#### 정규화: Weight Decay + Dropout
논문은 두 가지 정규화를 명시한다.

- weight decay(L2 penalty) 계수: $5 \cdot 10^{-4}$
- dropout: 첫 두 FC(4096) 층에 적용, dropout ratio $0.5$

L2 정규화는 손실에 $\lambda \lVert w\rVert_2^2$를 더하는 형태로 이해할 수 있고($\lambda=5\cdot 10^{-4}$), 이는 큰 가중치를 억제해 과적합을 줄이는 역할을 한다. dropout은 학습 시 일부 뉴런을 확률적으로 제거해 **"공동 적응(co-adaptation)"** 을 줄여 일반화를 높인다.

#### 초기화: Pre-Initialisation(얕은 모델로부터 복사) + 랜덤 초기화
깊은 네트워크에서 나쁜 초기화가 학습을 멈출 수 있다는 점을 논문은 강조한다(gradient instability). 그래서 논문은 다음 절차로 초기화를 한다.

1. 먼저 얕은 A 구성을 **랜덤 초기화**로 학습한다.
2. 더 깊은 모델을 학습할 때:
   - 앞쪽 4개의 conv layer + 뒤쪽 3개의 FC layer를 A에서 가져와 초기화한다.
   - 중간에 추가된 conv들은 랜덤 초기화한다.
3. 랜덤 초기화는 평균 $0$, 분산 $10^{-2}$인 정규분포에서 샘플링하고, bias는 $0$으로 둔다.

논문은 추가로 논문 제출 이후에는 Glorot & Bengio(2010) 방식으로도 pre-training 없이 초기화 가능함을 발견했다고 덧붙인다.

#### 입력 크롭과 스케일 지터링: 학습 데이터 증강의 핵심
네트워크 입력은 항상 `224×224`로 고정이다. 학습 이미지는 먼저 가장 짧은 변이 S가 되도록 **isotropic resizing**을 하고, 그 안에서 `224×224`를 랜덤 크롭한다. 그리고 랜덤 horizontal flip과 RGB color shift(Krizhevsky et al., 2012)도 적용한다.

여기서 중요한 차별점이 훈련 스케일 $S$를 고정할지, 범위에서 샘플링할지다.

- **single-scale training**: $S$를 256 또는 384로 고정
- **multi-scale training(scale jittering)**: $S \sim \mathcal{U}([S_{\min}, S_{\max}])$, 논문은 $S_{\min}=256, S_{\max}=512$ 사용

논문은 이를 multi-scale object statistics를 학습에 반영하는 data augmentation으로 설명한다.

#### 학습 절차 의사코드
논문의 설명을 그대로 알고리즘 형태로 정리하면 다음과 같다.

```text
Algorithm: VGG classification training (Sect. 3.1)
Inputs:
  - Training set D = {(I_i, y_i)}
  - Network configuration (A/B/C/D/E)
  - Training scale policy: fixed S or jittered S in [Smin, Smax]
Hyperparams:
  - batch_size = 256, momentum = 0.9
  - weight_decay = 5e-4
  - dropout = 0.5 on first two FC layers
  - lr = 1e-2, decay by 10 when val stops improving (3 times total)
For iter = 1..370K:
  1. Sample a mini-batch of images
  2. For each image:
       a) Choose scale S (fixed or random in [Smin,Smax])
       b) Resize isotropically so min side = S
       c) Random crop 224x224
       d) Random horizontal flip
       e) Random RGB colour shift
       f) Subtract per-channel mean RGB (computed on training set)
  3. Forward -> logits -> softmax
  4. Compute multinomial logistic regression loss (+ weight decay)
  5. Backprop gradients
  6. SGD with momentum update
  7. If validation accuracy plateaus: lr /= 10
Output: trained weights
```

### 🔹 Dense Evaluation, Multi-Crop, Multi-Scale
테스트는 정확도를 올리기 위해 계산을 더 쓰는 레시피가 매우 중요하다. 논문은 먼저 테스트 스케일 $Q$를 정의한다.

- 훈련 스케일: $S$
- 테스트 스케일: $Q$ (min side를 $Q$로 리사이즈)

논문은 단일 스케일 평가에서

- fixed $S$인 모델: $Q=S$
- jittered $S \in [S_{\min}, S_{\max}]$로 학습한 모델: $Q=0.5(S_{\min}+S_{\max})$

로 시작한 뒤, 멀티스케일에서는 $Q$를 여러 값으로 두고 평균한다.

#### Dense evaluation(fully-convolutional)
논문의 핵심 트릭은 FC를 conv로 바꿔서 이미지 전체에 네트워크를 **밀집 적용(dense apply)** 한다는 것이다.

- 첫 번째 FC(보통 `512×7×7 → 4096`)는 `7×7 conv`로 바꿀 수 있다.
- 나머지 FC는 `1×1 conv`로 바꿀 수 있다.

그 결과, 입력 이미지(크롭이 아닌 전체)를 넣으면 **공간 위치별 클래스 점수 맵(score map)** 이 나온다. 논문은 마지막에 이 score map을 spatial average pooling(sum-pooling)해 이미지 단위 클래스 점수 벡터로 만든다고 말한다.

그리고 horizontal flip도 적용해서 원본/플립 softmax posterior를 평균한다.

#### Multi-crop evaluation과의 관계
논문은 dense evaluation은 여러 크롭을 샘플링할 필요가 없어서 효율적이라고 하면서도, Szegedy et al.(2014)처럼 많은 crops를 쓰면 더 좋아질 수 있다고 인정한다. 이유는 두 가지다.

1. crop 기반은 샘플링이 더 촘촘해질 수 있다(많은 crop을 쓰면).
2. boundary condition이 다르다: crop은 padding이 $0$으로 들어가고, dense는 주변 문맥이 padding 역할을 해서 receptive field가 사실상 더 커진다.

그래서 논문은 참고용으로 3 scales에서 각 scale당 50 crops(5×5 grid × 2 flips), 총 150 crops 실험도 수행하고, dense와 결합했을 때 상보적(complementary)임을 보여준다.

#### 테스트 절차 의사코드
논문의 설명을 알고리즘으로 정리하면 다음과 같다.

```text
Algorithm: VGG dense testing at scale Q (Sect. 3.2)
Input: trained network, test image I
1. Resize I isotropically so min side = Q
2. Convert FC layers to conv layers:
     - FC1 -> 7x7 conv
     - FC2, FC3 -> 1x1 conv
3. Apply the fully-convolutional net densely to the whole image
4. Obtain class score map (H' x W' x K)
5. Spatial average pool the score map -> vector s in R^K
6. Repeat for horizontally flipped image and average posteriors
Output: class posterior for image at scale Q
```

멀티스케일은 여러 $Q$에서 위 과정을 반복하고 posterior를 평균하면 된다.

### 🔸 Implementation Details: Multi-GPU 학습
논문 구현은 Caffe 기반이지만, 여러 GPU에서 학습/평가를 하기 위해 상당한 수정을 했다고 한다. 핵심은 **data parallelism**이다.

- batch를 GPU들로 나눠 병렬 forward/backward
- 각 GPU의 gradient를 평균해 전체 batch gradient를 구성
- 동기(synchronous) 업데이트라서 단일 GPU batch 학습과 결과가 "정확히 같다"고 설명

4 GPU에서 3.75× speedup을 얻었다고 보고하며, Titan Black 4개 환경에서 단일 네트워크 학습이 2~3주 걸렸다고 적는다. (이 디테일은 깊은 네트워크를 학습하는 것이 실제로 얼마나 무겁냐를 보여주는 당시의 현실적인 문장이다.)

---

## 4️⃣ 실험 설정 및 모델 테스팅

### 🔹 실험 세팅: ILSVRC-2012, top-1/top-5 error
논문은 실험을 ILSVRC-2012 데이터셋으로 수행한다.

- training: 1.3M images
- validation: 50K images
- test: 100K images(라벨 비공개, 서버 제출)

평가는 top-1 error와 top-5 error를 사용한다. 특히 top-5 error는 정답 클래스가 모델이 제시한 상위 5개 예측 안에 없을 비율이며 ILSVRC의 주된 기준이라고 강조한다.

### 🔸 Single-Scale Evaluation: 깊이 vs 성능, LRN의 무의미, 스케일 지터링의 이득
단일 스케일 평가에서는 테스트 이미지 min side를 $Q$로 맞춘 뒤, 그 스케일에서 평가한다. 논문은 fixed $S$인 경우 $Q=S$, jittered 학습의 경우 $Q=0.5(S_{\min}+S_{\max})$로 두고 시작한다.

결과는 아래와 같다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/0703a37f-5a93-49e2-b2c2-a5f0121473d2/image.png" width="70%">
</p>

이 표에서 논문이 강조하는 해석은 다음과 같다.

1. **LRN은 도움이 안 된다.** A-LRN이 A보다 오히려 나쁘거나 비슷하다. 그래서 이후 깊은 모델에서는 normalisation을 쓰지 않는다고 선언한다.

2. **깊이가 늘수록 error가 감소한다.** A(11) → E(19)로 갈수록 top-1/top-5가 내려간다.

3. **C vs D 비교가 중요하다.** C는 1×1 conv를 포함해 비선형성이 늘었지만 D보다 성능이 떨어진다. 즉 비선형성 증가 자체도 도움은 되지만(C가 B보다 좋음), 공간 문맥을 보는 3×3 receptive field를 유지하는 게 더 중요할 수 있다(D가 C보다 좋음)라는 결론을 이 구간에서 끌어낸다.

4. **스케일 지터링 학습이 유의미하게 좋다.** $S \in [256,512]$로 학습한 모델이 단일 스케일 테스트에서도 fixed $S$ 대비 더 낫다. 논문은 이를 "multi-scale statistics를 잡는 augmentation"으로 해석한다.

또한 B 구성에 대해, 3×3 두 층을 5×5 한 층으로 바꾼 얕은 대체 네트워크와 비교했더니 center crop 기준 top-1 error가 7% 더 나빴다고 보고한다. 이는 **작은 필터로 깊게 쌓는 것이, 큰 필터로 얕게 만드는 것보다 낫다**는 주장에 추가 근거가 된다.

### 🔹 Multi-Scale Evaluation: 테스트 스케일 평균을 통한 성능 향상
멀티스케일에서는 여러 $Q$에 대해 posterior를 평균한다.

- fixed $S$ 모델: $Q=\{S-32, S, S+32\}$
- jittered $S \in [S_{\min},S_{\max}]$ 모델: $Q=\{S_{\min}, 0.5(S_{\min}+S_{\max}), S_{\max}\}$

결과는 아래와 같다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/86229e1a-b390-4ca2-aae6-3859af5a5ec0/image.png" width="70%">
</p>

논문에서의 멀티스케일 평가는 "여러 $Q$에서 posterior를 평균"하는 방식이며, 결과적으로 (Table 3)의 단일 스케일 대비 성능이 개선된다고 정리한다. 특히 본문에서 다음 두 숫자를 명시적으로 강조한다.

| 항목 | 수치 |
|---|---:|
| best single-network (val) top-1 / top-5 error | 24.8% / 7.5% |
| configuration E (test) top-5 error | 7.3% |

즉 VGG는 **"깊이 + 멀티스케일 평가"** 조합으로 단일 모델에서도 당시 기준 매우 낮은 top-5 error를 달성했고, 이후 앙상블로 더 낮춘다.

여기서 중요한 포인트는 학습이 multi-scale이든 아니든, 테스트를 multi-scale로 하면 좋아진다는 것이다. 하지만 fixed $S$ 모델은 너무 넓은 $Q$ 범위로 가면 성능이 떨어질 수 있으므로, 훈련-테스트 스케일 괴리가 커지지 않도록 $S\pm 32$ 근방만 쓴다는 세부 규칙도 같이 기억해두는 게 좋다.

### 🔸 Multi-Crop vs. Dense
논문은 3.2절에서 언급한 대로, dense evaluation과 multi-crop 평가를 비교하고 결합한다. 결과는 다음과 같으며, 이 실험은 전부 $S\in[256,512]$로 학습했고 $Q \in \{256,384,512\}$ 3 scales를 사용한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/13d76a9d-e504-4438-b70d-56e12cd9d864/image.png" width="70%">
</p>

해석은 간단하지만 중요한 결론으로 연결된다.

- multi-crop이 dense보다 약간 좋거나 비슷하다.
- **둘을 결합하면 더 좋아진다** → boundary condition 차이로 상보적이라는 논문의 추측과 맞물린다.

즉 평가 프로토콜은 단순히 _"테스트 때 많이 돌리면 좋다"_ 가 아니라, 서로 다른 평가 방식이 서로 다른 종류의 오차를 상쇄할 수 있다는 메시지가 들어 있다.

### 🔹 ConvNet Fusion: 모델 앙상블
이제 논문은 여러 모델의 softmax posterior를 평균하는 앙상블로 성능을 더 끌어올린다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/87bb5570-82c4-4f69-a7c6-91ec1adafb92/image.png" width="70%">
</p>

논문은 제출 시점에는 single-scale 위주로 학습되어 7개 모델 앙상블을 사용했고, 이후에는 multi-scale D와 E 두 개만으로 6.8%까지 내려갔다고 설명한다. 특히 **2개 모델만으로도** 6.8%를 달성했다는 점을 강조하며, 이는 다른 팀들이 훨씬 더 많은 모델을 섞는 것과 대비된다.

### 🔸 SOTA 비교
논문은 마지막으로 동시대 SOTA들과 비교한다. 핵심 메시지는 다음 문장으로 요약된다.

- VGG 2 nets 앙상블은 6.8% top-5 test error로, GoogLeNet(7 nets, 6.7%)과 경쟁적이다.
- single-net 성능에서도 VGG가 강하며, 단일 GoogLeNet보다 낫다고 주장한다.
- 복잡한 새로운 모듈(Inception 같은)을 도입하지 않고도, 고전적 ConvNet(LeCun et al., 1989; Krizhevsky et al., 2012) + 더 큰 깊이만으로 성능을 끌어올렸다는 점을 강조한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/b56abf13-99e8-4768-8249-6f73908c0d85/image.png" width="70%">
</p>

이 표는 논문 원문에서는 더 많은 방법들을 포함하지만, 핵심 비교 축은 **"VGG vs (이전 세대) vs GoogLeNet"** 이다. 그리고 VGG가 깊이라는 비교적 단순한 축을 밀어붙여 이 수준까지 왔다는 점이 이 논문의 가장 강한 메시지다.

---

## 💡 해당 논문의 시사점과 한계

VGG 논문이 남긴 의의는 크게 세 층위로 볼 수 있다.

1. **아키텍처 설계의 단순화:** "작은 커널(3×3)을 일관되게 반복 + stage별 채널 증가 + pool로 다운샘플"은 이후 수많은 모델의 기본 블록이 된다. 특히 3×3 스택의 receptive field/파라미터 효율 논증은, 왜 깊게 쌓는가를 납득시키는 매우 좋은 설명이다.

2. **평가/활용 레시피의 표준화:** 멀티스케일, dense evaluation, multi-crop, 앙상블을 체계적으로 비교해, 어떤 평가 프로토콜이 어디서 이득을 주는지를 표 형태로 남겼다. 이는 후대에 정확도 비교를 할 때도 중요한 기준점이 된다.

3. **전이 학습의 실전성:** ILSVRC pre-train 특징을 "dense + global average pooling + multi-scale"로 뽑고 선형 SVM을 붙이는 파이프라인은, 이후 수년간 다양한 비전 과제에서 강력한 베이스라인으로 자리 잡는다.

그만큼 한계도 분명하다.

- 논문은 **깊이가 중요하다**는 메시지를 강하게 밀지만, 그 과정에서 계산 비용(2~3주 학습)과 모델 크기(100M+ 파라미터)의 부담을 함께 안는다. 즉 성능은 강력하지만 **효율성 측면의 최적해**는 아니다.
- 또한 구조가 단순한 대신, 당시에는 BatchNorm 같은 안정화 기법이 없어서 초기화/사전학습 같은 트릭이 필요했고, 이는 깊어지면 무조건 쉽다가 아니라 깊게 학습시키는 공학이 필요하다는 점을 반증한다.

그럼에도 불구하고 VGG는 **깊이의 힘**을 가장 깔끔하게 보여준 대표적 논문으로서, 이후의 ResNet, DenseNet, 현대 전이 학습 레시피를 이해하는 데도 여전히 좋은 기준점이다.

---

## 👨🏻‍💻 VGGNet 구현하기
이 파트는 [`lucid`](https://github.com/ChanLumerico/lucid)를 이용해 구현한 실제 코드를 읽고, 논문 속 구성요소가 코드에서 어떤 형태로 나타나는지 살펴보자. VGGNet의 `lucid` 버전 구현은 [`vgg.py`](https://github.com/ChanLumerico/lucid/blob/main/lucid/models/imgclf/vgg.py)에 있다.

먼저 해당 파일 전체 코드는 아래와 같다.

```python
class VGGNet(nn.Module):
    def __init__(self, conv_config: list[int | str], num_classes: int = 1000) -> None:
        super().__init__()
        self.conv = self._make_layers(conv_config)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def _make_layers(self, config: list[int | str]) -> nn.Sequential:
        layers = []
        in_channels = 3
        for layer in config:
            if layer == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, layer, kernel_size=3, padding=1))
                layers.append(nn.ReLU())

                in_channels = layer

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


@register_model
def vggnet_11(num_classes: int = 1000, **kwargs) -> VGGNet:
    config = [64, "M"]
    config.extend([128, "M"])
    config.extend([256, 256, "M"])
    config.extend([512, 512, "M", 512, 512, "M"])

    return VGGNet(config, num_classes, **kwargs)


@register_model
def vggnet_13(num_classes: int = 1000, **kwargs) -> VGGNet:
    config = [64, 64, "M"]
    config.extend([128, 128, "M"])
    config.extend([256, 256, "M"])
    config.extend([512, 512, "M", 512, 512, "M"])

    return VGGNet(config, num_classes, **kwargs)


@register_model
def vggnet_16(num_classes: int = 1000, **kwargs) -> VGGNet:
    config = [64, 64, "M"]
    config.extend([128, 128, "M"])
    config.extend([256, 256, 256, "M"])
    config.extend([512, 512, 512, "M", 512, 512, 512, "M"])

    return VGGNet(config, num_classes, **kwargs)


@register_model
def vggnet_19(num_classes: int = 1000, **kwargs) -> VGGNet:
    config = [64, 64, "M"]
    config.extend([128, 128, "M"])
    config.extend([256, 256, 256, 256, "M"])
    config.extend([512, 512, 512, 512, "M", 512, 512, 512, 512, "M"])

    return VGGNet(config, num_classes, **kwargs)
```

이제 논문 관점에서 이 구현을 단계적으로 읽어보자.

### 0️⃣ 사전 설정 및 준비 단계
이 파일에서 눈여겨볼 구현 관문은 **세 가지**다.

1. `VGGNet` 클래스: 논문의 공통 레이아웃을 코드로 고정한 본체
2. `_make_layers`: 논문의 "conv 스택 + maxpool(M)"을 config 리스트로부터 생성하는 빌더
3. `vggnet_11/13/16/19`: 논문 구성 A/B/D/E에 해당하는 config를 만들어 레지스트리에 등록하는 팩토리 함수

논문에서 C 구성(1×1 conv 포함)이나 A-LRN(정규화) 같은 변형도 다루지만, Lucid 구현은 가장 표준적인 _VGG-A/B/D/E_ 만 제공하는 셈이다.

### 1️⃣ VGG 구성표를 코드로 옮기는 방식
논문에서 conv와 pooling을 번갈아 쌓는 구조는, Lucid 구현에서 `conv_config: list[int | str]`로 표현된다.

- `int`는 "해당 채널 수로 3×3 conv를 하나 쌓아라"
- `"M"`은 "2×2, stride 2 maxpool을 하나 쌓아라"

예를 들어 `vggnet_16`의 config는 논문의 D 구성과 구조적으로 동일한 나열이다.

```python
config = [64, 64, "M"]
config.extend([128, 128, "M"])
config.extend([256, 256, 256, "M"])
config.extend([512, 512, 512, "M", 512, 512, 512, "M"])
```

논문에서의 stage 개념(64/128/256/512/512)은 리스트에서 pooling을 기준으로 구간이 나뉘는 것으로 자연스럽게 구현된다.

### 2️⃣ 3×3 conv + ReLU + 2×2 maxpool의 구현
`_make_layers`는 config를 읽어 `nn.Sequential`을 만든다.

```python
def _make_layers(self, config: list[int | str]) -> nn.Sequential:
    layers = []
    in_channels = 3
    for layer in config:
        if layer == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, layer, kernel_size=3, padding=1))
            layers.append(nn.ReLU())

            in_channels = layer

    return nn.Sequential(*layers)
```

이 구현을 논문과 1:1로 연결하면 다음과 같다.

- `nn.Conv2d(..., kernel_size=3, padding=1)`:
  - 논문의 3×3 receptive field를 그대로 반영한다.
  - stride는 기본값(`1`)로 두어, 논문이 말한 "every pixel에 stride 1로 적용"과 대응한다.
  - `padding=1`은 공간 크기(H×W)를 유지시키기 위한 전형적인 선택으로, 논문 레이아웃(풀링 시에만 반으로 줄어드는 흐름)과 맞물린다.
- `nn.ReLU()`:
  - 논문에서 표에서는 생략하지만 모든 conv 뒤에 ReLU가 들어간다고 한 부분을 구현한다.
- `nn.MaxPool2d(kernel_size=2, stride=2)`:
  - 논문의 2×2, stride 2 maxpool을 직접 반영한다.

즉 논문의 깊이 증가 설계 원칙이, Lucid에서는 `_make_layers`라는 단일 함수로 깔끔하게 고정되어 있다.

### 3️⃣ 7×7로 정렬
논문 VGG의 전형적 입력은 224×224이며, `2×2` pooling을 5번 하면 공간 크기는

$$
224 \to 112 \to 56 \to 28 \to 14 \to 7
$$

로 정확히 7×7이 된다. Lucid는 이를 다음 코드로 **입력 크기가 달라도** 7×7로 맞춘다.

```python
self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
```

이 선택은 논문의 레이아웃을 보존하면서도, 입력 해상도 변화에 대해 더 견고하게 만들기 위한 _"구현 친화적"_ 장치로 읽을 수 있다. (논문은 입력 전처리로 224×224 크롭을 고정한다는 점에서 사실상 같은 목표를 달성한다.)

### 4️⃣ FC + ReLU + Dropout
VGG의 분류기 head는 다음 `nn.Sequential`로 구현된다.

```python
self.fc = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(4096, num_classes),
)
```

논문과의 대응은 명확하다.

- FC-4096 → FC-4096 → FC-1000(ILSVRC) 구조
- 첫 두 FC 뒤에 dropout(논문 dropout ratio 0.5)
- ReLU 사용

주의할 점은, 이 구현이 softmax까지 포함하지 않는다는 것이다. 논문은 목적함수를 multinomial logistic regression으로 설명하지만, 실제 구현에서는 보통 모델이 로짓을 내고, 손실 함수가 softmax+CE를 담당한다. Lucid 구현도 그 관행을 따른 것으로 볼 수 있다.

### 5️⃣ Forward 함수
`forward`는 논문 VGG의 "특징 추출 → 고정 크기 정렬 → 펼치기 → 분류" 흐름을 그대로 따른다.

```python
def forward(self, x: Tensor) -> Tensor:
    x = self.conv(x)
    x = self.avgpool(x)

    x = x.reshape(x.shape[0], -1)
    x = self.fc(x)
    return x
```

중요한 포인트는 flatten이다.

- `x.shape[0]`는 배치 크기 $N$
- `-1`은 나머지를 펼쳐 `512*7*7`로 만든다

이 flatten이 `nn.Linear(512 * 7 * 7, 4096)`과 정확히 맞물려, 논문 VGG의 FC 입력 차원을 구현 관점에서 강제한다.

### 6️⃣ 모델 레지스트리 연결
파일 하단의 `vggnet_11/13/16/19`는 각각 config 리스트를 만들고 `VGGNet`을 생성한다.

```python
@register_model
def vggnet_11(num_classes: int = 1000, **kwargs) -> VGGNet:
    ...
    return VGGNet(config, num_classes, **kwargs)
```

논문과의 대응은 다음처럼 읽는 게 자연스럽다.

- `vggnet_11` → 구성 A(11 layers)
- `vggnet_13` → 구성 B(13 layers)
- `vggnet_16` → 구성 D(16 layers)
- `vggnet_19` → 구성 E(19 layers)

---

## ✅ 정리

VGG 논문은 **더 깊게 쌓으면 좋다**는 직관을, 가장 단순하고 일관된 설계(3×3 conv 고정)로 밀어붙여 ILSVRC에서 SOTA에 도달한 작업이다. 이 논문을 자세히 읽으면, 깊이 자체의 이점뿐 아니라 스케일을 어떻게 다룰 것인가(학습 지터링, 테스트 멀티스케일), 평가에서 어떤 방식(dense/multi-crop)이 어디서 이득을 주는가, 앙상블이 어떤 수준의 추가 이득을 주는가까지 실제 성능을 결정하는 공학적 선택들이 한 묶음으로 연결되어 있음을 이해하게 된다.

Lucid의 `vgg.py` 구현은 그중에서도 VGG의 가장 핵심적인 설계 원칙(3×3 conv + ReLU 반복, 2×2 maxpool로 stage를 나누는 구조, 7×7 정렬 후 FC-4096-4096 head)을 `conv_config` 리스트와 `_make_layers` 빌더로 깔끔하게 코드화한 형태다. 논문을 먼저 따라가고 이 구현을 읽으면, 표(Table 1)의 아키텍처 나열이 실제 코드에서는 어떤 생성 규칙으로 나타나는지를 매우 직접적으로 연결해서 이해할 수 있다.

#### 📄 출처
Simonyan, Karen, and Andrew Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition." *International Conference on Learning Representations (ICLR)*, 2015. arXiv:1409.1556.
