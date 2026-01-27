# [MobileNet] MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
MobileNet 논문은 이미지 분류를 포함한 다양한 비전 과제를 모바일·임베디드 환경에서 수행하려면, 높은 정확도뿐 아니라 **모델 크기와 지연 시간(latency)**까지 함께 최적화해야 한다는 문제의식에서 출발한다. 저자들은 네트워크를 단순히 얕게 만들거나 파라미터를 줄이는 수준을 넘어, convolution 연산을 **Depthwise Separable Convolution**으로 치환해 계산량을 구조적으로 줄이고, 여기에 **Width Multiplier**와 **Resolution Multiplier**라는 두 하이퍼파라미터를 도입해 지연 시간–정확도 트레이드오프를 손쉽게 조절할 수 있는 아키텍처 계열인 MobileNets를 제안한다. 또한 ImageNet 분류뿐 아니라, 객체 검출·얼굴 속성·지오로컬라이제이션·지식 증류 등 폭넓은 응용에서 MobileNet이 실용적인 백본으로 기능함을 수치로 보여준다.

(Fig. 1: Standard Convolution을 Depthwise Convolution과 Pointwise Convolution으로 분해하는 도식)  

## 1️⃣ Introduction

### 🔹 모바일·임베디드 비전의 제약 조건과 목표 함수
논문은 AlexNet 이후 CNN이 고정밀을 향해 깊이와 복잡도를 늘려 왔지만, 그 과정이 모델을 반드시 효율적으로 만들지는 않았다고 지적한다. 로보틱스, 자율주행, 증강현실과 같은 환경에서는 추론이 제한된 연산 자원 위에서 **시간 제약을 만족**해야 하므로, 단순 정확도 경쟁이 아닌 **속도·크기·전력**까지 포함한 설계 목표가 필요하다.

MobileNet은 이 목표를 다음처럼 구체화한다.

1. 연산량을 줄이되, 정확도의 손실은 완만하도록 설계한다.  
2. 서로 다른 디바이스/서비스 요구사항에 맞춰 모델 크기를 쉽게 조절할 수 있어야 한다.  
3. 단일 과제에만 맞춘 특수 설계가 아니라, 다양한 응용으로 전이 가능한 백본이어야 한다.  

#### 효율 지표의 분리: Mult-Adds, Parameter Count, Latency
논문은 효율을 논할 때 하나의 숫자만으로 정리하기 어렵다고 전제한다. MobileNet 실험은 기본적으로 Mult-Adds와 parameter count를 함께 제시하고, 이후 실제 배포에서 더 중요한 latency까지 고려하는 방향으로 설계를 정당화한다.

1. **Mult-Adds**는 연산량의 근사치로서, 모델이 수행하는 산술 연산 규모를 비교하는 데 유용하다.  
2. **Parameter Count**는 저장 크기와 메모리 대역폭 요구를 반영하며, 모델 다운로드·캐시·상주 메모리까지 포함한 비용과 연결된다.  
3. **Latency**는 하드웨어와 커널 구현에 강하게 의존한다. 같은 Mult-Adds라도 연산이 어떤 형태로 배열되는지(예: 1×1 GEMM 위주인지)에 따라 실제 실행 시간은 달라질 수 있다.  

### 🔸 핵심 아이디어의 개요: Operator Factorization과 두 하이퍼파라미터
MobileNet이 택한 핵심 전략은, 표준 2D convolution이 한 번에 수행하던 연산을 **공간 방향의 필터링**과 **채널 혼합**으로 분해하는 것이다. 이를 통해 연산의 대부분을 1×1 convolution(GEMM으로 효율적으로 구현 가능)에 몰아 넣고, 3×3 공간 convolution은 채널별로 독립적인 depthwise 단계로 최소화한다.

또한 논문은 모델을 한 번 설계해 두고, 이후의 리소스 제약에 맞게 조정할 수 있는 두 하이퍼파라미터를 제시한다.

1. **Width Multiplier** $\alpha$: 채널 수를 층 전반에 걸쳐 균일하게 줄인다.  
2. **Resolution Multiplier** $\rho$: 입력 해상도와 내부 feature map 해상도를 함께 줄인다.  

이 두 값은 계산량과 파라미터를 구조적으로 줄이는 동시에, 정확도의 하락을 비교적 예측 가능한 형태로 만들도록 의도된다.

#### 하이퍼파라미터의 역할 분담
$\alpha$와 $\rho$는 모두 계산량을 줄이지만, 감소 메커니즘이 다르다.

1. $\alpha$는 채널 폭을 줄이므로, 1×1 convolution의 입력/출력 채널이 함께 줄어들어 계산이 대략 $\alpha^2$에 비례해 줄어든다. 파라미터 역시 채널 기반 항이 지배적이므로 비슷한 스케일로 감소한다.  
2. $\rho$는 공간 해상도를 줄이므로, feature map의 면적 항이 $(\rho D_F)^2$로 들어가 계산이 $\rho^2$에 비례해 줄어든다. 그러나 파라미터는 커널 텐서 크기로 결정되므로 $\rho$에 거의 영향을 받지 않는다.  

이 분담은 Table 6과 Table 7에서 각각 다른 형태의 트레이드오프 곡선이 나타나는 이유로 연결된다.

### 🔹 논문 전개의 로드맵
논문은 먼저 작은 네트워크 설계와 압축 계열의 선행 연구를 검토한 뒤, MobileNet을 정의하는 핵심 연산인 Depthwise Separable Convolution을 수식적으로 정리한다. 이후 표준 MobileNet 아키텍처(Table 1)를 제시하고, $\alpha$와 $\rho$가 연산량과 모델 크기를 어떻게 변화시키는지 식으로 기술한다. 마지막으로 ImageNet을 중심으로 한 ablation과 트레이드오프 실험(Table 4–9, Fig. 4–5), 그리고 응용 과제(Table 10–14)로 논증을 마무리한다.

#### 논문이 다루는 응용 과제의 의미
MobileNet 논문은 ImageNet 분류 성능만 보고 끝내지 않는다. MobileNet의 목적이 특정 벤치마크의 정확도를 소폭 올리는 것이 아니라, 모바일 환경에서 유용한 **범용 백본**을 제공하는 데 있기 때문이다. 따라서 논문이 다양한 과제를 포함하는 이유는 다음처럼 정리된다.

1. 분류 기반 표현이 detection과 embedding으로 전이되는지 확인한다.  
2. distillation과 결합할 때 작은 모델의 성능이 어떻게 바뀌는지 확인한다.  
3. 실제 서비스에서 자주 등장하는 과제(예: 얼굴 속성, 지오로컬라이제이션)에서 실용성을 점검한다.  

---

## 2️⃣ Prior Work

### 🔸 작은 네트워크 설계의 두 축: 압축과 직접 설계
논문은 효율 모델 연구를 크게 두 범주로 구분한다.

1. **Pretrained Network Compression**: 큰 모델을 먼저 학습한 뒤 가지치기, 양자화, 저랭크 근사, 해싱 등을 통해 모델을 압축하는 접근이다.  
2. **Small Network Training**: 처음부터 작은 모델을 설계해 직접 학습하는 접근이다.  

MobileNet은 두 번째 축에 속한다. 즉, 특정 대형 모델을 전제하지 않고도, 설계 자체로 지연 시간과 크기를 줄이는 아키텍처를 제공하는 것을 목표로 한다.

### 🔹 Latency 중심 최적화의 강조
논문은 기존의 많은 접근이 파라미터 수(모델 크기) 중심으로만 효율을 논하지만, 실제 서비스에서는 latency가 더 직접적인 제약인 경우가 많다고 말한다. 파라미터 수가 작더라도 구현이 비효율적이면 빠르지 않을 수 있기 때문이다. MobileNet은 설계 단계에서부터 계산을 1×1 convolution에 집중시키고, 이를 GEMM 기반으로 효율 구현 가능하게 만들어 **실제 지연 시간 측면의 이점**까지 겨냥한다.

#### 압축 접근의 한계와 MobileNet의 포지셔닝
압축 기반 접근은 큰 모델의 성능을 유지하면서 작은 모델을 얻을 수 있다는 장점이 있지만, 배포 관점에서는 다음 문제가 남을 수 있다.

1. 압축 형태가 커널 수준에서 구조적이지 않으면, 실제 구현에서 속도 이득이 제한적일 수 있다.  
2. 압축 과정 자체가 별도의 파이프라인을 요구해, 모델 제작 비용이 증가할 수 있다.  
3. 하드웨어나 런타임이 특정 압축 형태를 지원하지 않으면, 이득을 체감하기 어렵다.  

MobileNet은 이런 문제를 회피하기 위해, 처음부터 연산 패턴이 명확한 구조적 블록(depthwise + pointwise)을 반복하는 아키텍처를 제시한다. 이는 **작게 만들기**가 아니라 **작게 설계하기**로 문제를 전환하는 방식이다.

---

## 3️⃣ MobileNet Architecture

### 🔸 Depthwise Separable Convolution의 정의와 계산량
표준 convolution은 입력 채널 수 $M$, 출력 채널 수 $N$, 커널 공간 크기 $D_K\times D_K$, feature map의 공간 해상도 $D_F\times D_F$에서 대략 다음의 Mult-Adds를 요구한다.

$$
\text{Mult-Adds}_\text{standard}
= D_K^2\cdot M\cdot N\cdot D_F^2
$$

Depthwise Separable Convolution은 이를 두 단계로 분해한다.

1. **Depthwise Convolution**: 각 입력 채널에 대해 독립적인 $D_K\times D_K$ 공간 convolution을 적용한다.  
2. **Pointwise Convolution**: 1×1 convolution으로 채널 혼합을 수행해 $M\to N$ 사상을 만든다.  

각 단계의 Mult-Adds는 다음처럼 정리된다.

$$
\text{Mult-Adds}_\text{depthwise}
= D_K^2\cdot M\cdot D_F^2
$$

$$
\text{Mult-Adds}_\text{pointwise}
= M\cdot N\cdot D_F^2
$$

따라서 전체는

$$
\text{Mult-Adds}_\text{separable}
= D_K^2\cdot M\cdot D_F^2 + M\cdot N\cdot D_F^2
$$

이 되며, 표준 convolution 대비 비율은

$$
\frac{\text{Mult-Adds}_\text{separable}}{\text{Mult-Adds}_\text{standard}}
= \frac{1}{N} + \frac{1}{D_K^2}
$$

로 정리할 수 있다. 일반적으로 $D_K=3$이고 $N$이 충분히 크다면, 분해는 대략 8–9배 수준의 계산 감소를 만들 수 있다는 것이 논문의 직관적 메시지다.

(Fig. 2: Standard Convolution Filter를 Depthwise Filter와 Pointwise Filter로 대체하는 개념 도식)  

#### Depthwise Separable Convolution Forward Pseudocode
논문 정의를 연산 순서로 옮기면 다음 의사코드로 정리할 수 있다.

```text
Algorithm: Depthwise Separable Convolution
Inputs:
  - Input X (N x M x H x W)
  - Depthwise kernels Kd (M groups, spatial Dk x Dk)
  - Pointwise kernels Kp (1 x 1, M -> N)
1. Z = DepthwiseConv(X; groups=M)     # channel-wise spatial filtering
2. Y = PointwiseConv(Z; kernel=1x1)   # channel mixing
Output: Y (N x N x H' x W')
```

#### Parameter Count 관점의 비교
연산량뿐 아니라 파라미터 수도 같은 형태로 비교할 수 있다. 표준 convolution의 파라미터 수는

$$
\text{Params}_\text{standard} = D_K^2\cdot M\cdot N
$$

이며, depthwise separable의 파라미터 수는

$$
\text{Params}_\text{separable} = D_K^2\cdot M + M\cdot N
$$

이다. 여기서 $M\cdot N$ 항은 pointwise가 담당하며, 대부분의 경우 이 항이 지배적이 된다. 이 사실은 Table 2에서 1×1 convolution이 파라미터의 74.59%를 차지한다는 결과와도 일관된다.

### 🔹 MobileNet Body Architecture의 구성 원리
MobileNet의 본체는 첫 레이어를 제외하고는 거의 모든 공간 convolution을 depthwise separable로 구성한다. 논문은 모든 convolution과 separable convolution 뒤에 batch normalization과 ReLU를 둔다고 명시하며, 마지막 fully-connected는 softmax로 이어지는 분류기로서 비선형을 두지 않는다.

또한 downsampling은 다음 두 지점에서 이루어진다.

1. 첫 번째 레이어의 stride 2 convolution  
2. 특정 depthwise convolution 단계의 stride 2 설정  

이 선택은 공간 해상도를 줄이는 비용을 depthwise 단계로 보내는 구조로 볼 수 있다.

(Fig. 3: Standard ConvBNReLU와 Depthwise Separable ConvBNReLU의 비교 도식)  

(Table 1: MobileNet Body Architecture)  

| Stage | Type / Stride | Filter Shape | Input Size |
|---:|---|---|---|
| 1 | Conv / s2 | 3×3×3×32 | 224×224×3 |
| 2 | Conv dw / s1 | 3×3×32 dw | 112×112×32 |
| 3 | Conv / s1 | 1×1×32×64 | 112×112×32 |
| 4 | Conv dw / s2 | 3×3×64 dw | 112×112×64 |
| 5 | Conv / s1 | 1×1×64×128 | 56×56×64 |
| 6 | Conv dw / s1 | 3×3×128 dw | 56×56×128 |
| 7 | Conv / s1 | 1×1×128×128 | 56×56×128 |
| 8 | Conv dw / s2 | 3×3×128 dw | 56×56×128 |
| 9 | Conv / s1 | 1×1×128×256 | 28×28×128 |
| 10 | Conv dw / s1 | 3×3×256 dw | 28×28×256 |
| 11 | Conv / s1 | 1×1×256×256 | 28×28×256 |
| 12 | Conv dw / s2 | 3×3×256 dw | 28×28×256 |
| 13 | Conv / s1 | 1×1×256×512 | 14×14×256 |
| 14 | 5× (Conv dw / s1, Conv / s1) | 3×3×512 dw, 1×1×512×512 | 14×14×512 |
| 15 | Conv dw / s2 | 3×3×512 dw | 14×14×512 |
| 16 | Conv / s1 | 1×1×512×1024 | 7×7×512 |
| 17 | Conv dw / s2 | 3×3×1024 dw | 7×7×1024 |
| 18 | Conv / s1 | 1×1×1024×1024 | 7×7×1024 |
| 19 | Avg Pool / s1 | Pool 7×7 | 7×7×1024 |
| 20 | FC / s1 | 1024×1000 | 1×1×1024 |
| 21 | Softmax / s1 | Classifier | 1×1×1000 |

#### Table 1의 읽는 법과 핵심 패턴
Table 1은 MobileNet이 반복적으로 사용하는 단위를 보여준다. 핵심은 depthwise(3×3)에서 공간 변환을 수행하고, pointwise(1×1)에서 채널 혼합을 수행한다는 점이다. 또한 512 채널 구간에서 동일 블록을 여러 번 반복하는 구조는, MobileNet을 하나의 VGG-Style stack으로도 읽을 수 있게 만든다.

#### Downsampling 설계의 일관성
Table 1을 단계적으로 따라가면, 공간 해상도는 대략 224 → 112 → 56 → 28 → 14 → 7 순서로 감소한다. 이 감소는 대부분 depthwise 단계의 stride 2에 의해 일어나며, pointwise는 stride 1로 채널 혼합만 수행한다. 이 패턴은 다음 장점을 갖는다.

1. 해상도 감소 비용을 depthwise에 실어, 상대적으로 저렴한 연산에서 downsampling이 수행되도록 한다.  
2. pointwise는 해상도를 바꾸지 않으므로, 채널 혼합의 역할이 더 명확해진다.  
3. stage 경계가 stride 설정으로 표현되므로, 구현에서 구조가 단순해진다.  

이 설계는 이후 Lucid 구현 파트에서 `_Depthwise(..., stride=2)`가 stage 전환의 핵심 인자로 등장하는 이유로 연결된다.

### 🔸 연산 분포와 구현 효율: 1×1 Convolution의 지배
논문은 MobileNet에서 연산과 파라미터가 어디에 집중되는지를 Table 2로 보여준다. 결과는 직관적이다. depthwise 단계는 공간 convolution이지만 채널별 독립이므로 상대적으로 비중이 작고, 1×1 convolution이 연산의 대부분을 차지한다.

(Table 2: Resource Per Layer Type)  

| Layer Type | Mult-Adds | Parameters |
|---|---:|---:|
| Conv 1×1 | 94.86% | 74.59% |
| Conv DW 3×3 | 3.06% | 1.06% |
| Conv 3×3 | 1.19% | 0.02% |
| Fully Connected | 0.18% | 24.33% |

논문은 이 분포가 중요한 이유를 구현 관점에서 설명한다. 1×1 convolution은 im2col 같은 메모리 재배치를 덜 요구하며, GEMM으로 직접 매핑해 고효율 구현이 가능하다. 반면 unstructured sparsity는 높은 sparsity가 아니면 실제 속도 이득이 작을 수 있어, 구조적 분해가 더 직접적인 대안이라는 논리를 제공한다.

#### Fully Connected의 파라미터 비중 해석
Table 2에서 fully-connected는 Mult-Adds 관점에서는 0.18%로 매우 작지만, 파라미터 관점에서는 24.33%로 꽤 큰 비중을 차지한다. 이는 마지막 분류기가 1024×1000 크기의 가중치를 갖기 때문이다.

이 관찰은 두 방향의 설계 시사점을 준다.

1. 분류 과제에서는 마지막 classifier가 파라미터 병목이 될 수 있으므로, 클래스 수가 큰 과제에서는 head 설계가 중요해진다.  
2. 반대로 연산량 관점에서는 대부분이 1×1 convolution에 있으므로, 속도 최적화는 주로 pointwise 구현에 좌우될 가능성이 크다.  

논문이 다양한 응용 과제에서 마지막 레이어가 과제에 따라 크게 달라질 수 있음을 언급하는 것도, 이 파라미터 분포의 성격과 연결된다.

#### Table 2와 Table 3의 연결: 지배항의 이동
Table 3은 단일 레이어 예시에서 표준 convolution이 depthwise separable로 바뀌면 Mult-Adds와 파라미터가 동시에 급격히 줄어드는 것을 보여준다. 그런데 MobileNet 전체 관점(Table 2)에서는 1×1 convolution이 연산의 약 95%를 차지한다.

이 두 관찰을 함께 놓으면, MobileNet에서 효율을 논할 때 어떤 항이 지배항이 되는지 구분할 수 있다.

1. 표준 convolution 기반 네트워크에서는 $D_K^2MN$ 항이 지배적이어서, 3×3 공간 convolution 자체가 병목이 되기 쉽다.  
2. MobileNet에서는 공간 변환이 depthwise로 분리되면서, $MN$ 항을 갖는 pointwise가 연산의 지배항으로 이동한다.  

따라서 MobileNet 설계의 핵심은 단지 depthwise를 추가한 것이 아니라, 계산 병목을 **구조적으로 이동**시켜 효율 구현이 가능한 형태로 만드는 데 있다. 논문이 1×1 convolution을 GEMM으로 직접 구현 가능한 장점과 연결하는 것도 이 맥락에서 이해할 수 있다.

### 🔹 Width Multiplier $\alpha$: 채널 폭의 균일 스케일링
논문은 이미 작은 MobileNet을 더 축소할 수 있도록, 채널 폭을 전 층에 걸쳐 얇게 만드는 폭 스케일링 하이퍼파라미터 $\alpha$를 제안한다. 한 층에서 입력 채널 수 $M$과 출력 채널 수 $N$이 있을 때,

$$
M\to \alpha M,\quad N\to \alpha N
$$

으로 바꾸는 방식이며, depthwise separable convolution의 Mult-Adds는

$$
D_K^2\cdot \alpha M\cdot D_F^2 + \alpha M\cdot \alpha N\cdot D_F^2
$$

로 표현된다. 논문은 $\alpha$가 대략적으로 계산량과 파라미터 수를 $\alpha^2$에 비례해 감소시키는 효과를 갖는다고 설명한다. 이때 중요한 점은, $\alpha<1$인 모델은 구조가 달라지는 것이 아니라 채널 폭만 줄어들기 때문에, 설계자가 동일한 아키텍처를 기반으로 다양한 리소스 제약을 대응할 수 있다는 점이다.

#### $\alpha^2$ 스케일링이 성립하는 이유
왜 계산량이 대략 $\alpha^2$로 줄어드는지 직관을 더 분해해보면 다음과 같다.

1. depthwise 항은 $D_K^2\cdot \alpha M\cdot D_F^2$로, 채널이 선형으로 줄어든다.  
2. pointwise 항은 $\alpha M\cdot \alpha N\cdot D_F^2$로, 입력/출력 채널이 동시에 줄어들어 제곱으로 감소한다.  

그리고 MobileNet은 Table 2에서 보이듯 pointwise 항이 연산의 대부분을 차지하므로, 전체 계산량 감소가 $\alpha^2$에 가까워지는 경향이 나타난다. 즉, MobileNet에서 폭 스케일링은 단순한 채널 축소가 아니라, 계산 병목이 있는 1×1 convolution에 직접 작동하는 조절 노브다.

### 🔸 Resolution Multiplier $\rho$: 표현 해상도의 균일 스케일링
두 번째 하이퍼파라미터 $\rho$는 입력 해상도를 줄이는 방식으로, 내부 feature map의 공간 해상도도 함께 줄어들도록 한다. 이를 $D_F\to \rho D_F$로 보면, depthwise separable convolution의 Mult-Adds는

$$
D_K^2\cdot \alpha M\cdot (\rho D_F)^2 + \alpha M\cdot \alpha N\cdot (\rho D_F)^2
$$

가 되며, 결과적으로 계산량이 $\rho^2$에 비례해 감소한다. 논문은 $\rho$를 224, 192, 160, 128 같은 입력 크기 선택으로 암묵적으로 설정하는 형태로 사용한다.

#### 단일 레이어 예시를 통한 누적 효과의 직관
논문은 Table 3에서 특정 내부 레이어($D_K=3$, $M=512$, $N=512$, $D_F=14$)를 예로 들어, 표준 convolution → depthwise separable → $\alpha$ 적용 → $\rho$ 적용 순으로 비용이 누적 감소함을 보여준다.

(Table 3: Resource Usage for Modifications to Standard Convolution)  

| Modification | Million Mult-Adds | Million Parameters |
|---|---:|---:|
| Convolution | 462.0 | 2.36 |
| Depthwise Separable Conv | 52.3 | 0.27 |
| $\alpha=0.75$ 추가 적용 | 29.6 | 0.15 |
| $\rho=0.714$ 추가 적용 | 15.1 | 0.15 |

이 표는 두 메시지를 함께 준다. 첫째, depthwise separable 치환만으로도 가장 큰 폭의 계산 감소가 발생한다. 둘째, 이후의 $\alpha$, $\rho$는 설계자가 요구 조건에 맞춰 추가로 줄이는 조절 노브로 기능한다.

#### 해상도 축소의 성격: 파라미터 고정, 연산량 감소
해상도 축소는 입력 텐서의 공간 크기를 줄이는 것이므로, 커널 텐서의 파라미터에는 영향을 주지 않는다. 따라서 Table 7에서 파라미터가 모두 4.2M으로 고정되는 현상이 나타난다. 반면 연산량은 $D_F^2$ 항에 의해 줄어들므로, 특히 초기 stage처럼 feature map이 큰 구간에서 효과가 크게 나타난다.

이 관찰은 실무적으로도 의미가 있다. 같은 모델을 유지한 채 입력 해상도를 조절해 latency를 줄일 수 있다면, 모델 교체 없이 서비스 품질을 조정할 수 있는 간단한 전략이 된다. 다만 해상도 축소는 작은 객체나 세밀한 텍스처에 더 민감할 수 있으므로, 다운스트림 과제에 따라 정확도 하락 패턴이 달라질 수 있다는 점까지 함께 고려해야 한다.

---

## 4️⃣ Experiments

### 🔹 실험 질문의 구성: 연산 분해, 구조 축소, 응용 전이
논문은 실험을 세 층위로 구성한다.

1. depthwise separable 치환이 표준 convolution 대비 얼마나 손해 없이 효율을 만드는가  
2. 네트워크를 얕게 하는 것과 폭을 줄이는 것 중 무엇이 더 낫나  
3. $\alpha$와 $\rho$가 만드는 트레이드오프가 어떻게 나타나나  

그리고 이 기본 질문들을 ImageNet 기반의 비교(Table 4–9)로 정리한 뒤, 다양한 응용 과제에서 MobileNet의 전이 가능성을 Table 10–14로 제시한다.

#### Training Configuration의 핵심 포인트
논문은 MobileNet 학습이 Inception V3와 유사한 설정을 일부 차용하되, 작은 모델의 특성에 맞춰 정규화와 데이터 증강을 줄였다고 설명한다.

1. Optimizer는 RMSProp를 사용한다.  
2. 학습은 TensorFlow에서 비동기(asynchronous) gradient descent로 수행한다.  
3. 작은 모델은 overfitting이 상대적으로 덜하다고 보고, side head나 label smoothing을 사용하지 않는다.  
4. Inception 계열에서 사용하던 강한 image distortion 중 일부를 줄이며, 작은 crop의 크기를 제한한다.  
5. depthwise 필터는 파라미터 수가 매우 적기 때문에, 이 부분에 큰 weight decay를 주지 않는 것이 중요하다고 언급한다.  

이 설명은 MobileNet의 성능이 아키텍처뿐 아니라, 작은 모델에 맞춘 학습 레시피와도 결합되어 있음을 보여준다.

#### Depthwise Weight Decay에 대한 직관
논문이 depthwise 필터에 weight decay를 거의 주지 않는 것이 중요하다고 말하는 이유는, depthwise 단계의 파라미터 수가 매우 적기 때문이다. 파라미터 수가 적은 텐서에 강한 L2 정규화를 걸면

1. 표현이 과도하게 0으로 수축하며  
2. 채널별 공간 필터가 충분히 유연한 형태를 갖기 어려워지고  
3. 결과적으로 pointwise가 혼합할 수 있는 기반 특징 자체가 약해질 수 있다  

는 형태의 문제가 발생할 수 있다. MobileNet은 depthwise와 pointwise의 역할 분담이 명확하므로, depthwise 단계의 과도한 정규화가 전체 표현에 미치는 영향이 상대적으로 크게 나타날 수 있다.

### 🔸 Depthwise Separable vs Full Convolution 비교
논문은 MobileNet을 표준 convolution으로 구성한 Conv MobileNet과, 제안한 MobileNet을 직접 비교한다. 결과는 표준 convolution 기반이 약간 더 높은 정확도를 갖지만, 계산량과 파라미터에서 큰 격차가 난다는 점을 보여준다.

(Table 4: Depthwise Separable vs Full Convolution MobileNet)  

| Model | ImageNet Accuracy | Million Mult-Adds | Million Parameters |
|---|---:|---:|---:|
| Conv MobileNet | 71.7% | 4866 | 29.3 |
| MobileNet | 70.6% | 569 | 4.2 |

이 비교는 MobileNet의 설계 논지를 직접 뒷받침한다. 즉, 1% 수준의 정확도 감소와 맞바꿔 계산량과 파라미터를 크게 줄일 수 있다면, 모바일 환경에서는 설계 선택이 합리적일 수 있다.

#### 비용 절감 비율의 정량적 해석
Table 4의 숫자는 ratio로 보면 메시지가 더 선명해진다.

1. Mult-Adds는 4866M에서 569M으로 감소하므로, 대략 8배 이상의 연산 절감이 발생한다.  
2. Parameters는 29.3M에서 4.2M으로 감소하므로, 모델 크기도 수 배 감소한다.  

논문이 정확도 감소를 비교적 작게 서술한 배경은 이 ratio에 있다. 즉, 작은 정확도 하락을 대가로 얻는 리소스 절감이 매우 크다는 점을 수치로 보여준다.

### 🔹 Narrow vs Shallow 비교: 폭 축소의 우위
두 번째 실험은 네트워크를 **얕게 만드는 것**과 **얇게 만드는 것**을 비교한다. 논문은 Table 1에서 14×14×512 구간에 해당하는 5개의 separable 블록을 제거해 Shallow MobileNet을 구성하고, 이에 준하는 비용을 갖는 0.75 MobileNet과 비교한다.

(Table 5: Narrow vs Shallow MobileNet)  

| Model | ImageNet Accuracy | Million Mult-Adds | Million Parameters |
|---|---:|---:|---:|
| 0.75 MobileNet | 68.4% | 325 | 2.6 |
| Shallow MobileNet | 65.3% | 307 | 2.9 |

논문은 이 결과를 바탕으로, 동일한 연산 예산에서 깊이를 줄이는 것보다 폭을 줄이는 편이 정확도 측면에서 더 유리하다고 결론내린다. MobileNet의 축소 전략이 $\alpha$ 중심으로 제시되는 배경이 여기에서 정당화된다.

#### 얕게 만들기의 구조적 비용
Shallow MobileNet은 단순히 블록을 빼서 깊이를 줄인 형태이지만, 표현력 관점에서는 다음 손실이 발생할 수 있다.

1. 14×14 해상도 구간의 반복은 중간 수준의 receptive field 확장과 채널 혼합을 누적하는 구간이다. 이 반복이 빠지면 고수준 특징이 충분히 정리되기 어렵다.  
2. 폭 축소는 모든 단계에서 표현력을 조금씩 줄이는 형태이므로, 특정 stage만 크게 약화시키지 않는 장점이 있다.  

논문은 이 실험으로, 작은 모델 설계에서 깊이를 줄이는 방식이 직관적으로는 쉬워도 성능 비용이 더 클 수 있다는 점을 강조한다.

### 🔸 Width Multiplier 트레이드오프
Table 6은 $\alpha$ 변화에 따른 ImageNet 정확도와 비용 변화를 직접 제시한다. $\alpha$가 작아질수록 비용이 줄고 정확도가 떨어지며, 특히 $\alpha=0.25$에서 급격한 하락이 관찰된다고 논문은 해석한다.

(Table 6: MobileNet Width Multiplier)  

| Width Multiplier | ImageNet Accuracy | Million Mult-Adds | Million Parameters |
|---:|---:|---:|---:|
| 1.0 MobileNet-224 | 70.6% | 569 | 4.2 |
| 0.75 MobileNet-224 | 68.4% | 325 | 2.6 |
| 0.5 MobileNet-224 | 63.7% | 149 | 1.3 |
| 0.25 MobileNet-224 | 50.6% | 41 | 0.5 |

#### $\alpha$ 축소에서의 급격한 붕괴 구간
Table 6의 해석에서 중요한 점은, $\alpha$가 줄어들수록 정확도가 선형으로 떨어지지 않는다는 점이다. 1.0→0.75→0.5 구간은 비교적 완만하지만, 0.25에서는 급격한 하락이 나타난다. 이는 채널 폭이 너무 얇아지면

1. pointwise 층의 채널 혼합 용량이 급격히 줄고  
2. depthwise에서 얻은 채널별 공간 특징을 충분히 결합하지 못하며  
3. 결과적으로 전체 표현력이 임계점 아래로 떨어질 수 있기 때문으로 해석할 수 있다.  

논문은 이 현상을 통해, 폭 스케일링은 강력한 노브이지만 극단적으로 작은 모델 영역에서는 더 정교한 설계가 필요하다는 신호를 함께 제공한다.

### 🔹 Resolution Multiplier 트레이드오프
Table 7은 입력 해상도를 줄이는 방식으로 $\rho$를 변화시켰을 때의 결과다. 해상도 감소는 파라미터를 거의 바꾸지 않지만 계산량을 줄이며, 정확도는 완만하게 감소한다.

(Table 7: MobileNet Resolution)  

| Resolution | ImageNet Accuracy | Million Mult-Adds | Million Parameters |
|---:|---:|---:|---:|
| 224 | 70.6% | 569 | 4.2 |
| 192 | 69.1% | 418 | 4.2 |
| 160 | 67.2% | 290 | 4.2 |
| 128 | 64.4% | 186 | 4.2 |

#### 해상도 축소에서의 안정적 하락 패턴
Table 7은 해상도가 낮아질수록 정확도가 완만하게 하락하는 모습을 보여준다. 파라미터는 고정되어 있으므로, 이 결과는 표현 용량 자체가 줄었다기보다 입력 정보량과 공간 샘플링이 줄어드는 효과로 이해하는 편이 자연스럽다.

특히 분류 과제에서는 global average pooling 이전의 feature map이 충분히 요약 정보를 담을 수 있다면, 입력 해상도가 어느 정도 줄어도 성능이 유지될 수 있다. 반대로 작은 물체 중심의 과제에서는 같은 축소가 더 큰 성능 손실로 이어질 가능성이 있으므로, 해상도 노브는 과제 의존성이 큰 조절 수단이다.

논문은 Fig. 4에서 $\alpha\in\{1,0.75,0.5,0.25\}$와 해상도 집합의 교차 곱으로 16개 모델을 만들고, 정확도–계산량 관계가 로그-선형에 가깝게 나타난다고 설명한다. 또한 Fig. 5에서는 정확도–파라미터 관계를 같은 방식으로 제시하며, 해상도는 파라미터에 영향을 주지 않는다는 점을 시각적으로 확인한다.

(Fig. 4: Accuracy–Mult-Adds 트레이드오프 곡선)  
(Fig. 5: Accuracy–Parameter Count 트레이드오프 곡선)  

#### Fig. 4–5의 해석: 두 노브의 분리된 효과
Fig. 4는 계산량(Mult-Adds)과 정확도의 관계를, Fig. 5는 파라미터 수와 정확도의 관계를 시각화한다. 두 그림을 함께 보면 $\alpha$와 $\rho$가 서로 다른 방식으로 트레이드오프를 만든다는 점이 더 선명해진다.

1. $\alpha$를 줄이면 계산량과 파라미터가 함께 줄어든다. 따라서 Fig. 4와 Fig. 5 모두에서 이동이 나타난다.  
2. $\rho$를 줄이면 계산량은 줄지만 파라미터는 거의 변하지 않는다. 따라서 Fig. 4에서는 점이 이동하지만, Fig. 5에서는 같은 파라미터 수 위에서 색(해상도)만 달라지는 형태로 나타난다.  

논문은 정확도–비용 관계가 로그-선형처럼 보인다고 설명하며, 매우 작은 모델 영역(예: $\alpha=0.25$)에서 궤적이 꺾이는 현상을 별도로 강조한다. 이는 앞서 Table 6에서 관찰한 급격한 붕괴 구간과 같은 현상으로 이해할 수 있다.

### 🔸 인기 모델과의 비교: MobileNet의 위치
논문은 MobileNet을 GoogLeNet과 VGG16 같은 대표 모델과 비교해, 정확도에 크게 뒤지지 않으면서도 비용을 크게 줄일 수 있음을 강조한다.

(Table 8: MobileNet Comparison to Popular Models)  

| Model | ImageNet Accuracy | Million Mult-Adds | Million Parameters |
|---|---:|---:|---:|
| 1.0 MobileNet-224 | 70.6% | 569 | 4.2 |
| GoogleNet | 69.8% | 1550 | 6.8 |
| VGG 16 | 71.5% | 15300 | 138 |

또한 더 작은 MobileNet을 SqueezeNet, AlexNet과 비교해, 작은 모델 구간에서도 경쟁력이 있음을 보여준다.

(Table 9: Smaller MobileNet Comparison to Popular Models)  

| Model | ImageNet Accuracy | Million Mult-Adds | Million Parameters |
|---|---:|---:|---:|
| 0.50 MobileNet-160 | 60.2% | 76 | 1.32 |
| Squeezenet | 57.5% | 1700 | 1.25 |
| AlexNet | 57.2% | 720 | 60 |

#### 비교 표의 메시지: 정확도 유지와 리소스 절감의 동시 달성
Table 8은 MobileNet이 VGG16에 근접한 정확도를 보이면서도 계산량과 파라미터가 대폭 줄어든다는 점을 강조한다. Table 9는 더 작은 MobileNet이 AlexNet, SqueezeNet과 같은 효율 모델들과 비교해도 경쟁력이 있음을 보여준다.

이 비교는 MobileNet의 목적이 최고 정확도가 아니라, 특정 정확도 수준을 달성하는 데 필요한 리소스를 줄이는 데 있다는 점을 다시 확인시킨다. 특히 모바일 환경에서는 1–2%의 정확도 차이보다, 배포 가능성과 응답 시간 차이가 더 중요한 경우가 많기 때문이다.

### 🔹 응용 과제 확장: Fine-Grained, Geolocalization, Face, Detection, Embedding
논문은 MobileNet이 ImageNet 분류에만 맞춘 구조가 아니라, 다양한 응용으로 전이 가능한 백본임을 강조하기 위해 여러 과제를 보고한다.

#### Fine-Grained Recognition: Stanford Dogs
(Table 10: MobileNet for Stanford Dogs)  

| Model | Top-1 | Million Mult-Adds | Million Parameters |
|---|---:|---:|---:|
| Inception V3 | 84.0% | 5000 | 23.2 |
| 1.0 MobileNet-224 | 83.3% | 569 | 3.3 |
| 0.75 MobileNet-224 | 81.9% | 325 | 1.9 |
| 1.0 MobileNet-192 | 81.9% | 418 | 3.3 |
| 0.75 MobileNet-192 | 80.5% | 239 | 1.9 |

논문이 이 실험으로 보여주려는 메시지는, MobileNet이 작은 모델임에도 fine-grained 설정에서 강한 기반 표현을 제공할 수 있다는 점이다. 특히 Inception V3 수준의 정확도에 근접하면서 비용이 크게 줄어든다.

#### Fine-Grained 설정에서의 해석 포인트
fine-grained 분류는 클래스 간 차이가 미세하고, 배경보다 대상의 국소 디테일이 중요해질 수 있다. 그럼에도 MobileNet이 근접한 성능을 보이는 것은, depthwise separable이 단순히 계산을 줄이는 기법이 아니라 충분한 표현력을 유지할 수 있는 연산 분해라는 메시지를 강화한다.

또한 이 실험은 해상도와 폭의 축소가 무조건적으로 손해만을 의미하지 않음을 시사한다. 작은 모델이더라도 적절한 사전학습과 미세조정을 거치면, 대상 도메인에서는 높은 효율로 경쟁력 있는 결과를 만들 수 있다.

#### Large-Scale Geolocalization: PlaNet
PlaNet은 지리적 셀을 클래스처럼 두고 분류 문제로 지오로컬라이제이션을 수행한다. 논문은 PlaNet을 MobileNet 구조로 재학습한 결과를 함께 보고하며, 기존 PlaNet 대비 성능 손실이 크지 않으면서 모델이 훨씬 작아짐을 강조한다.

(Table 11: Performance of PlaNet Using the MobileNet Architecture)  

| Scale | PlaNet | MobileNet | Im2GPS |
|---|---:|---:|---:|
| Continent (2500 km) | 79.3% | 77.6% | 51.9% |
| Country (750 km) | 60.3% | 64.0% | 35.4% |
| Region (200 km) | 45.2% | 51.1% | 32.1% |
| City (25 km) | 31.7% | 31.7% | 21.9% |
| Street (1 km) | 11.4% | 11.0% | 2.5% |

Table 11은 MobileNet이 대규모 분류 기반 지오로컬라이제이션에서도 충분한 표현을 제공하며, 기존 Im2GPS 대비 크게 우수한 성능을 유지한다는 점을 보여준다.

#### PlaNet에서의 마지막 레이어 크기와 파라미터 분포
논문 본문은 PlaNet의 경우 마지막 레이어가 매우 커질 수 있음을 함께 언급한다. 즉, MobileNet 본체(body)는 수백만 파라미터 수준이지만, 지오로컬라이제이션처럼 클래스 수가 많으면 최종 분류기의 파라미터가 지배적일 수 있다.

이 점은 Table 2의 fully-connected 파라미터 비중 관찰과 자연스럽게 연결된다. 실무적으로는 backbone을 아무리 효율화해도, head 설계가 전체 모델 크기를 결정할 수 있다는 의미가 된다.

#### Face Attributes: Distillation과 모델 축소의 결합
얼굴 속성 분류에서는, 큰 모델의 출력을 모방하도록 학습하는 distillation과 MobileNet의 구조적 효율을 결합한다. 논문은 MobileNet 기반 분류기가 in-house baseline에 근접한 mean AP를 유지하면서도 Mult-Adds를 크게 줄일 수 있음을 Table 12로 제시한다.

(Table 12: Face Attribute Classification Using the MobileNet Architecture)  

| Setting | Mean AP | Million Mult-Adds | Million Parameters |
|---|---:|---:|---:|
| 1.0 MobileNet-224 | 88.7% | 568 | 3.2 |
| 0.5 MobileNet-224 | 88.1% | 149 | 0.8 |
| 0.25 MobileNet-224 | 87.2% | 45 | 0.2 |
| 1.0 MobileNet-128 | 88.1% | 185 | 3.2 |
| 0.5 MobileNet-128 | 87.7% | 48 | 0.8 |
| 0.25 MobileNet-128 | 86.4% | 15 | 0.2 |
| Baseline | 86.9% | 1600 | 7.5 |

#### Distillation과 MobileNet의 결합이 의미하는 것
논문이 강조하는 포인트는, distillation이 단순히 큰 모델의 성능을 작은 모델로 옮기는 기술이 아니라, 작은 모델이 상대적으로 약한 구간을 teacher의 출력 분포가 보완해 줄 수 있다는 점이다. MobileNet이 구조적으로 파라미터를 절약한다면, distillation은 학습 신호 측면에서 그 절약이 만든 표현력 공백을 줄여주는 역할을 할 수 있다.

표에서 Baseline이 86.9%이고 MobileNet 계열이 88% 수준을 보이는 것은, 이 setting에서 teacher 기반 학습이 강력한 정규화·지도 역할을 했을 가능성을 시사한다. 다만 baseline과 MobileNet의 학습 파이프라인이 완전히 동일하다고 가정하기 어렵기 때문에, 이 결과는 MobileNet 자체의 우위라기보다 작은 모델과 distillation 조합의 실용성을 보여주는 예로 읽는 편이 안전하다.

#### Object Detection: SSD와 Faster R-CNN의 백본으로서의 MobileNet
논문은 COCO에서 MobileNet을 SSD 300 및 Faster R-CNN의 백본으로 사용한 결과를 보고한다. 표는 다양한 프레임워크와 백본 조합에서 mAP와 연산량/파라미터를 함께 제시하며, MobileNet이 비교적 작은 비용으로도 경쟁력 있는 결과를 낼 수 있음을 보여준다.

(Table 13: COCO Object Detection Results Comparison)  
(Fig. 6: MobileNet SSD의 검출 결과 예시)  

#### Detection에서의 비용–정확도 해석
Table 13은 다양한 프레임워크 조합을 포함하므로, 단일 숫자만으로 결론을 내리기보다 비용 대비 성능을 함께 보는 것이 핵심이다. MobileNet 백본은 VGG 대비 Mult-Adds와 파라미터를 크게 줄이면서도, 특정 설정에서는 유사하거나 더 좋은 mAP를 보인다.

이 결과는 MobileNet이 분류 전용 백본이 아니라, RPN/SSD 같은 검출 구성요소와 결합해도 유의미한 특징 표현을 제공할 수 있음을 보여준다. 또한 모바일 환경에서 detection을 수행할 때, 백본을 바꿔서 latency를 낮추는 전략이 현실적임을 뒷받침한다.

#### Face Embeddings: FaceNet Distillation
마지막으로 논문은 FaceNet의 출력 임베딩을 모방하도록 MobileNet을 학습하는 distillation 기반 접근을 보고한다. Table 14는 작은 MobileNet 구성이 FaceNet 대비 어느 정도의 정확도를 달성하는지와 비용을 함께 보여준다.

(Table 14: MobileNet Distilled from FaceNet)  

| Model | Accuracy | Million Mult-Adds | Million Parameters |
|---|---:|---:|---:|
| FaceNet | 83.0% | 1600 | 7.5 |
| 1.0 MobileNet-160 | 79.4% | 286 | 4.9 |
| 1.0 MobileNet-128 | 78.3% | 185 | 5.5 |
| 0.75 MobileNet-128 | 75.2% | 166 | 3.4 |
| 0.75 MobileNet-128 (Smaller) | 72.5% | 108 | 3.8 |

#### Embedding 과제에서의 의미
embedding 학습은 분류와 달리, 출력 공간에서의 거리 구조를 직접 최적화하는 경향이 있다. 그럼에도 MobileNet이 distillation을 통해 FaceNet의 출력을 근사할 수 있다는 결과는, MobileNet의 표현이 충분히 풍부해 teacher가 만든 embedding 구조를 따라갈 수 있음을 시사한다.

표에서 작은 설정으로 갈수록 정확도가 감소하는 것은 자연스럽지만, Mult-Adds가 크게 줄어드는 구간에서도 완전히 붕괴하지 않는다는 점은 모바일 환경에서의 유용성을 강조하는 근거가 된다.

---

## 5️⃣ Conclusion

### 🔸 결론의 요지: 효율 연산과 조절 가능한 설계 공간
논문 결론은 다음 흐름으로 정리된다.

1. Depthwise Separable Convolution 기반의 MobileNet 아키텍처를 제안한다.  
2. $\alpha$와 $\rho$로 구성되는 두 하이퍼파라미터를 통해, 같은 설계 철학 위에서 다양한 지연 시간–정확도 조합을 쉽게 만들 수 있음을 보인다.  
3. ImageNet을 포함해 객체 검출, 얼굴 속성, 지오로컬라이제이션 등 다양한 과제에서 MobileNet이 실용적 백본이 될 수 있음을 수치로 확인한다.  

이 결론은 MobileNet이 단일 모델 제안이 아니라, 설계자가 제약 조건에 맞춰 탐색 가능한 **아키텍처 계열**을 제공한다는 점을 강조한다.

#### 후속 연구로의 연결
논문은 MobileNets를 TensorFlow에서 공개할 계획을 언급하며, 효율 모델이 연구 실험을 넘어 실제 채택을 목표로 한다는 태도를 드러낸다. 이후의 MobileNetV2/V3, 그리고 depthwise separable을 핵심으로 하는 여러 아키텍처가 등장한 흐름을 감안하면, MobileNet(v1)의 메시지는 다음처럼 요약할 수 있다.

1. 연산을 분해해 구조적으로 효율을 만든다.  
2. 설계 공간을 하이퍼파라미터로 정리해, 모델 선택을 체계화한다.  
3. 다운스트림 과제에서의 전이를 통해 효율 백본의 정당성을 강화한다.  

---

## 💡 해당 논문의 시사점과 한계 혹은 의의
MobileNet의 의의는 효율을 단순한 압축 기법으로만 다루지 않고, convolution 연산 자체를 분해해 **구조적으로 계산량을 줄이는 설계 원리**를 제시했다는 점이다. 이후의 Xception, MobileNetV2/V3를 포함한 여러 효율 모델이 depthwise separable 연산을 핵심 구성 요소로 채택한 흐름을 감안하면, 이 논문은 효율 아키텍처 설계의 기초적 레퍼런스로 기능한다.

또한 $\alpha$와 $\rho$는 연구자의 관점에서 설계 공간을 정리하는 도구로도 의미가 있다. 정확도–계산량 곡선이 완만하게 이동하는 패턴이 관측되면, 모델 선택을 정성적 감각이 아니라 서비스 제약에 기반한 선택 문제로 바꿀 수 있다.

한계로는, depthwise separable이 계산량을 줄인다고 해서 항상 동일 비율로 실제 latency가 줄어드는 것은 아니라는 점을 들 수 있다. 구현 커널과 하드웨어 특성에 따라 이득이 달라질 수 있으며, 논문도 구현 가능성과 GEMM 친화적 구조를 강조하는 방식으로 이 문제를 우회하려 한다. 또한 Table 13처럼 다운스트림 과제에서는 정확도 외에 프레임워크/하이퍼파라미터 영향이 함께 얽히므로, 백본 선택의 효과를 해석할 때 통제 조건을 명확히 보는 태도가 필요하다.

#### 실무적 의의: 모델 선택을 설계 노브로 전환
MobileNet의 실무적 가치는, 모델을 바꾼다는 행위를 아키텍처를 새로 연구하는 문제로 남겨두지 않고, $\alpha$와 $\rho$라는 노브를 통해 제약 조건에 맞는 모델을 선택하는 문제로 바꾸는 데 있다.

예를 들어 같은 서비스에서도

1. 고성능 디바이스에는 $\alpha=1.0$과 224 입력을 사용하고  
2. 저사양 디바이스에는 $\alpha=0.75$와 192 입력을 사용하며  
3. 더 극단적인 환경에서는 $\alpha=0.5$와 160 입력을 사용한다  

같은 형태로 라인업을 구성할 수 있다. 이때 중요한 점은 모델의 설계 철학과 코드 구조가 동일하므로, 운영과 유지보수 비용이 줄어든다는 점이다. 논문이 여러 조합을 교차 곱 형태로 탐색해 Fig. 4–5로 정리하는 것도, 실제로는 이런 라인업 설계를 염두에 둔 서술로 읽을 수 있다.

---

## 👨🏻‍💻 MobileNet Lucid 구현
파이썬 라이브러리 [`lucid`](https://github.com/ChanLumerico/lucid)로 구현한 MobileNet [`mobile.py`](https://github.com/ChanLumerico/lucid/blob/main/lucid/models/imgclf/mobile.py)를 살펴보자. 이 파일에는 MobileNetV2/V3/V4 구현도 함께 포함되어 있지만, 이 글은 MobileNet 논문(v1)과 직접 대응되는 `MobileNet`, `_Depthwise`, `mobilenet` 팩토리 함수 중심으로 설명한다.

#### Lucid 구현의 범위와 논문 대응의 주의점
`mobile.py`가 여러 버전을 한 파일에 담고 있기 때문에, 리뷰에서 다루는 논문(v1) 범위와 코드 범위를 명확히 분리할 필요가 있다. 이 글은

1. v1의 핵심 연산(depthwise + pointwise)  
2. v1의 폭 스케일링 인터페이스(width multiplier)  
3. v1의 단순 스택 형태의 body와 분류 head  

에 대응되는 부분만을 집중적으로 해설한다. v2/v3/v4는 별도 논문에 기반한 변형이므로, 본문에서는 구현이 존재한다는 사실만 짧게 인지하는 수준으로 제한한다.

### 0️⃣ 모델 대응 범위와 구성 요소 식별
MobileNet(v1)과 Lucid 구현 요소의 대응은 다음처럼 정리된다.

1. Depthwise Separable Convolution 단위: `_Depthwise`  
2. Table 1의 본체 스택: `MobileNet.__init__`에서의 `conv1`–`conv7` 조립  
3. 분류 head: `AdaptiveAvgPool2d((1,1))`와 `Linear`  
4. 모델 생성 인터페이스: `mobilenet(width_multiplier=..., num_classes=...)`  

### 1️⃣ `_Depthwise`: Depthwise + Pointwise의 결합 블록
`_Depthwise`는 논문이 말한 depthwise separable convolution을 Lucid에서 가장 직접적으로 구현한 단위다. 구조는 depthwise 단계와 pointwise 단계를 각각 `nn.Sequential`로 구성하고, depthwise 출력이 pointwise로 전달되는 형태다.

```python
class _Depthwise(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))
```

이 구현은 논문 정의와 다음 지점에서 1:1로 대응된다.

1. depthwise 단계는 `groups=in_channels`로 채널별 독립 공간 convolution을 만든다.  
2. pointwise 단계는 `kernel_size=1`로 채널 혼합을 수행한다.  
3. 두 단계 각각에 `BatchNorm2d`와 `ReLU`가 배치되어, 논문이 Fig. 3에서 설명한 ConvBNReLU 반복 패턴을 따른다.  

#### stride가 의미하는 stage 경계
`_Depthwise`는 `stride`를 depthwise convolution에만 적용한다. 이는 논문이 Table 1에서 downsampling을 depthwise 단계의 strided convolution으로 처리한다고 설명한 것과 대응된다. 따라서 Lucid 코드에서 `_Depthwise(..., stride=2)`가 등장하는 지점은, 곧 224→112, 112→56 같은 해상도 전환이 일어나는 stage 경계로 읽을 수 있다.

### 2️⃣ `MobileNet`: Table 1 스택의 코드 구성
`MobileNet` 클래스는 `width_multiplier`를 입력으로 받아, 채널 수를 균일하게 스케일링한다. 코드에서 `alpha = width_multiplier`로 두고, 채널 수를 `int(기본채널 * alpha)`로 계산한다.

#### `nn.ConvBNReLU2d`의 역할: 첫 레이어의 결합 연산
`MobileNet` 구현에서 첫 레이어는 `_Depthwise`가 아니라 `nn.ConvBNReLU2d`를 사용한다. 이는 논문이 첫 레이어만은 full convolution으로 둔다고 명시한 것과 대응된다. Lucid에서 `ConvBNReLU2d`는 `lucid/nn/fused.py`에서 정의되며, `Conv2d → BatchNorm2d → ReLU`의 결합을 하나의 모듈로 제공한다.

```python
class _ConvBNReLU(nn.Module):
    D: ClassVar[int | None] = None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        conv_bias: bool = True,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        bn_affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        if self.D is None:
            raise ValueError("Must specify 'D' value.")

        self.conv: nn.Module = _Conv[self.D - 1](
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            conv_bias,
        )
        self.bn: nn.Module = _BN[self.D - 1](
            out_channels, eps, momentum, bn_affine, track_running_stats
        )
        self.relu = nn.ReLU()

    def forward(self, input_: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(input_)))


class ConvBNReLU2d(_ConvBNReLU):
    D: ClassVar[int] = 2
```

이 모듈은 논문이 설명한 BN과 ReLU의 반복 패턴을 코드에서 간결하게 표현한다. 또한 `conv_bias=False`처럼 bias를 끄는 선택은, BN을 뒤에 두는 구조에서 bias 항의 필요성이 낮아지는 일반적인 구현 관행과도 일관된다.

```python
class MobileNet(nn.Module):
    def __init__(self, width_multiplier: float, num_classes: int = 1000) -> None:
        super().__init__()
        alpha = width_multiplier

        self.conv1 = nn.ConvBNReLU2d(
            3, int(32 * alpha), kernel_size=3, stride=2, padding=1
        )
        self.conv2 = _Depthwise(int(32 * alpha), int(64 * alpha))

        self.conv3 = nn.Sequential(
            _Depthwise(int(64 * alpha), int(128 * alpha), stride=2),
            _Depthwise(int(128 * alpha), int(128 * alpha), stride=1),
        )
        self.conv4 = nn.Sequential(
            _Depthwise(int(128 * alpha), int(256 * alpha), stride=2),
            _Depthwise(int(256 * alpha), int(256 * alpha), stride=1),
        )

        self.conv5 = nn.Sequential(
            _Depthwise(int(256 * alpha), int(512 * alpha), stride=2),
            *[
                _Depthwise(int(512 * alpha), int(512 * alpha), stride=1)
                for _ in range(5)
            ],
        )
        self.conv6 = _Depthwise(int(512 * alpha), int(1024 * alpha), stride=2)
        self.conv7 = _Depthwise(int(1024 * alpha), int(1024 * alpha), stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(1024 * alpha), num_classes)
```

이 구성은 논문 Table 1의 패턴과 다음 관점에서 대응된다.

1. `conv1`은 첫 레이어의 표준 convolution으로, 논문이 예외로 둔 full convolution에 해당한다.  
2. `conv2`부터는 `_Depthwise`가 반복되며, 3×3 depthwise와 1×1 pointwise가 결합된 단위가 본체를 구성한다.  
3. `conv5`의 `for _ in range(5)` 반복은, Table 1의 5× 반복 구간(14×14×512 구간)과 대응되는 반복 구조다.  
4. `avgpool`과 `fc`는 논문이 말한 global average pooling 이후 fully-connected 분류기로 이어지는 head를 구현한다.  

여기서 주의할 점은, Lucid 구현은 downsampling을 `_Depthwise(..., stride=2)`로 표현하는데, 논문 역시 downsampling을 depthwise 단계의 stride 설정으로 처리한다고 설명한다는 점이다. 따라서 stride 값이 곧 Table 1의 해상도 전이를 결정하는 핵심 인자 역할을 한다.

#### Lucid 구현과 논문 아키텍처의 미세한 차이
Lucid 구현은 `conv7 = _Depthwise(..., stride=2)`로 마지막에 downsampling을 한 번 더 적용한다. 논문 Table 1의 마지막 1024 채널 구간은 depthwise와 pointwise의 추가 적용을 포함하지만, 마지막 stage에서 해상도를 추가로 줄이는지 여부는 구현에 따라 변형될 수 있다.

따라서 Lucid 구현을 사용할 때는 다음을 분리해 보는 것이 안전하다.

1. 논문의 핵심 아이디어: depthwise separable과 $\alpha$, $\rho$ 기반 트레이드오프  
2. 구현의 선택: 마지막 stage에서 downsampling을 한 번 더 두는지 여부  

논문 수치(Table 6–7)와 동일한 설정을 재현하려면, stage 구성과 stride 설정을 별도로 점검해야 한다.

### 3️⃣ `MobileNet.forward`: feature extractor와 head의 직렬 전개
`forward`는 `conv1`부터 `conv7`까지를 순서대로 적용한 뒤, average pooling으로 1×1로 줄이고 flatten 후 `fc`로 분류 점수를 만든다.

```python
def forward(self, x: Tensor) -> Tensor:
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.conv6(x)
    x = self.conv7(x)

    x = self.avgpool(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fc(x)

    return x
```

이 전개는 MobileNet을 branch 없는 단순한 stack으로 이해하게 만든다. 논문이 Table 1을 통해 아키텍처를 리스트로 제시하는 방식과도 잘 맞는다. 또한 `reshape`를 통해 (N, C, 1, 1) 형태를 (N, C)로 바꾸는 부분은, pooling 이후 fully-connected에 넣기 위한 표준적인 분류 head 결합 방식이다.

### 4️⃣ `mobilenet`: 모델 등록 함수와 폭 스케일링 인터페이스
Lucid는 `@register_model` 데코레이터를 통해 모델 팩토리 함수를 등록한다. MobileNet(v1)은 `width_multiplier`를 외부에서 받을 수 있도록 노출한다.

```python
@register_model
def mobilenet(
    width_multiplier: float = 1.0, num_classes: int = 1000, **kwargs
) -> MobileNet:
    return MobileNet(width_multiplier, num_classes, **kwargs)
```

이 함수는 논문이 제안한 $\alpha$를 사용자가 손쉽게 지정하도록 해주는 인터페이스다. 리뷰 관점에서는, 논문 Table 6의 width multiplier 실험을 코드 레벨에서 재현 가능한 형태로 제공한다는 의미를 갖는다.

#### Resolution Multiplier의 부재와 입력 전처리의 위치
논문은 $\rho$를 별도의 하이퍼파라미터로 정의하지만, Lucid의 `mobilenet` 팩토리 함수는 $\rho$를 직접 인자로 받지 않는다. 이는 $\rho$가 모델 내부 레이어의 파라미터가 아니라 입력 해상도 선택으로 암묵적으로 반영되는 설정이기 때문이다.

따라서 Lucid로 논문 Table 7과 같은 비교를 수행하려면, 모델 코드가 아니라 데이터 파이프라인에서 입력 크기를 224/192/160/128로 맞추는 방식으로 $\rho$를 구현해야 한다. 즉, Lucid 구현에서

1. $\alpha$는 모델의 `width_multiplier`로 들어가고  
2. $\rho$는 입력 전처리 단계의 resize로 들어간다  

는 역할 분담이 형성된다.

---

## ✅ 정리
MobileNet은 convolution을 depthwise 단계와 pointwise 단계로 분해함으로써, 표준 convolution 기반 네트워크가 갖는 계산 병목을 구조적으로 완화한다. 논문은 이 분해가 단순히 이론적 계산량 감소로 끝나지 않도록, 연산을 1×1 convolution에 집중시키는 형태로 구현 효율까지 함께 고려한다. 또한 $\alpha$와 $\rho$라는 두 하이퍼파라미터를 통해, 동일한 설계 철학 아래에서 다양한 리소스 제약 조건을 만족하는 모델을 체계적으로 생성할 수 있는 방법을 제시한다. ImageNet에서의 ablation(Table 4–9)과 다양한 다운스트림 과제(Table 10–14) 결과는, MobileNet이 작은 모델이라는 이유만으로 표현력이 부족하다고 단정하기 어렵다는 점과, 백본 선택 문제를 정확도–지연 시간의 트레이드오프 문제로 재정식화할 수 있음을 뒷받침한다. Lucid 구현은 `_Depthwise`와 `MobileNet`의 단순한 스택 조립을 통해 논문의 핵심 연산 분해를 직접 반영하며, `mobilenet(width_multiplier=...)` 인터페이스로 폭 스케일링을 노출해 논문에서 제시한 설계 노브를 코드에서도 그대로 사용할 수 있게 한다.

#### 📄 출처
Howard, Andrew G., et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." arXiv, 2017, arXiv:1704.04861.
