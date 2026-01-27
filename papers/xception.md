# [Xception] Xception: Deep Learning With Depthwise Separable Convolutions
Xception은 Inception 계열의 모듈을 다시 해석하는 것에서 출발한다. 논문은 Inception 모듈이 **정규 convolution**과 **Depthwise Separable Convolution** 사이의 연속체(정확히는 분할 수로 매개되는 이산 스펙트럼) 위에 놓여 있으며, Depthwise Separable Convolution은 Inception 모듈을 극단으로 밀어붙인 형태로 이해할 수 있다고 주장한다. 이 관점을 기반으로, Inception 모듈을 **Depthwise Separable Convolution으로 전면 치환**한 아키텍처를 제안하며, 이를 *Extreme Inception*이라는 의미의 **Xception**으로 명명한다.

논문의 핵심 메시지는 성능 향상이 단순한 파라미터 증가가 아니라 **파라미터 사용 방식의 효율**에서 기인한다는 점이다. ImageNet에서는 Inception V3 대비 소폭 개선을, 훨씬 큰 내부 데이터셋(JFT)에서는 큰 개선을 보고한다. 또한 Residual Connection의 중요성, Depthwise–Pointwise 사이의 중간 activation 유무가 학습에 미치는 영향을 별도 실험으로 분해해 논증한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/e954d6b4-c057-4d23-8481-0355c0a23e88/image.png" width="40%">
</p>

---

## 1️⃣ Introduction

### 🔹 문제의식: Convolution이 동시에 맡는 두 종류의 상관관계
논문은 convolution layer가 사실상 3차원 공간에서 필터를 학습한다고 본다. 여기서 3차원은 공간 축(가로·세로)과 채널 축을 포함한다. 즉, 단일 convolution kernel은

- **Cross-Channel Correlation**(채널 간 상관관계)
- **Spatial Correlation**(공간적 상관관계)

을 동시에 모델링하는 역할을 수행한다. 이 동시 모델링이 과연 가장 효율적인가가 Inception 및 Xception의 문제의식으로 연결된다.

#### Convolution Kernel Parameterization
이 문제의식을 텐서 관점으로 더 구체화하면 다음과 같다. 일반적인 2D convolution에서 하나의 커널은 공간 크기 $k\times k$와 채널 축을 함께 갖는다. 입력 채널 수를 $C_{in}$, 출력 채널 수를 $C_{out}$이라 하면, 파라미터 수는 대략

$$
k^2\cdot C_{in}\cdot C_{out}
$$

으로 증가한다. 이 구조는 한 번의 convolution이

- 채널 축에서 어떤 입력 채널 조합을 만들지  
- 공간 축에서 어떤 패턴을 검출할지  

를 함께 결정해야 함을 의미한다. 논문이 말하는 분해 가능성은 결국 이 결합된 파라미터화가 항상 최선은 아닐 수 있다는 주장과 연결된다.

#### 설계 관점 요약: 결합 학습과 분해 학습의 대비
정규 convolution은 채널 혼합과 공간 변환을 한 번에 학습하는 방식이고, Inception 및 Xception은 이를 분해해 학습 부담을 줄이는 방향을 택한다. 이후 섹션에서 논문은 Inception이 수행한 분해가 어느 정도 수준인지, 그리고 그 분해를 극단으로 밀어붙이면 무엇이 되는지를 Fig. 2–4를 통해 전개한다.

### 🔸 Inception Hypothesis: 채널 상관관계와 공간 상관관계의 분해
논문은 Inception 모듈의 기본 가설을 다음과 같이 정리한다. Cross-channel correlation과 spatial correlation은 충분히 분리 가능하며, 따라서 둘을 하나의 convolution에서 함께 처리하기보다 **분해된 연산의 연쇄**로 처리하는 편이 더 효율적일 수 있다는 것이다.

논문이 제시하는 전형적 흐름은 다음과 같은 형태로 이해할 수 있다.

1. 1×1 convolution으로 채널 방향의 혼합을 수행하여 입력을 더 작은 채널 공간들로 사상한다.  
2. 각 채널 공간에서 3×3 또는 5×5 같은 공간 convolution을 수행한다.  

즉, 채널 혼합을 먼저 수행하고 그 결과 위에서 공간 변환을 수행한다는 점에서, Inception은 단일 convolution이 수행하던 일을 부분적으로 분해한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/c787ed28-3b94-443f-8390-1a04571625d6/image.png" width="60%">
</p>  

#### Canonical Inception Module과 Simplified Inception Module의 역할 분리
논문은 Fig. 1에서 Inception V3의 canonical module을 보여주지만, 설계 해석을 위해 Fig. 2에서 단순화된 Inception module을 사용한다. 이 단순화는 핵심 논점을 분리하기 위한 장치로 이해할 수 있다.

1. canonical module의 구현 디테일을 제거하고, 채널 혼합과 공간 변환의 분해라는 핵심만 남긴다.  
2. 분해를 채널 분할의 관점으로 재표현하기 위해, 동일 크기의 공간 convolution만 남긴다.  

이렇게 단순화된 구조에서, 채널을 몇 개의 segment로 나누어 공간 convolution을 수행하는지의 관점이 등장하고, 이는 depthwise separable convolution으로 자연스럽게 이어진다.

### 🔹 질문의 확장: 분해를 더 극단으로 밀어붙이면 무엇이 되는가
논문은 Fig. 2의 단순화된 Inception 모듈을 Fig. 3처럼 다시 쓸 수 있음을 지적한다. 여기서 핵심은 1×1 convolution 뒤의 공간 convolution이 **출력 채널을 몇 개의 segment로 나누어** 각 segment에 대해 독립적으로 적용되는 것으로 재해석될 수 있다는 점이다.

이때 자연스럽게 다음 질문이 생긴다.

1. segment의 수를 늘리면 어떤 일이 생기는가  
2. segment의 크기를 줄여 1개 채널 단위까지 분해하면 무엇이 되는가  

논문은 이 질문을 통해 **정규 convolution과 depthwise separable convolution이 하나의 연속선 위에 놓인다**는 관점을 제시한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/fc42bb87-7530-4b86-b296-0a5fd3cdff3d/image.png" width="40%">
</p>

#### Segmented Spatial Convolution과 Grouped Convolution의 대응
Fig. 3의 재표현은 구현 관점에서 보면 grouped convolution과 매우 유사한 형태로 해석될 수 있다. 채널을 여러 segment로 나누어 각 segment에 대해 독립적인 공간 convolution을 적용한다는 것은, convolution의 `groups`를 증가시키는 방향과 같은 형태의 분해를 의미한다.

- segment 수가 작으면, 각 segment가 담당하는 채널 폭이 넓어져 정규 convolution과 유사해진다.  
- segment 수가 커지면, 각 segment가 담당하는 채널 폭이 좁아져 depthwise에 가까워진다.  

논문은 이 구조가 아직 충분히 탐구되지 않았다고 말하며, Xception의 다음 단계로서 중간 지점 탐색을 Future Directions로 남긴다.

#### Depthwise Separable Convolution의 연산 비용 직관
정규 convolution과 depthwise separable convolution을 파라미터 수 관점에서 비교하면 다음과 같은 직관을 얻을 수 있다. 일반적으로 depthwise separable은

1. depthwise 단계: $k^2\cdot C_{in}$  
2. pointwise 단계: $C_{in}\cdot C_{out}$  

의 파라미터를 갖는다. 따라서 합은 $k^2\cdot C_{in} + C_{in}\cdot C_{out}$가 된다. $k^2\cdot C_{in}\cdot C_{out}$에 비해 훨씬 작아질 수 있으며, 논문이 말하는 효율적 파라미터 사용이라는 표현을 구조적으로 뒷받침한다.

#### Depthwise Separable Convolution Forward Pseudocode
논문 정의를 구현 절차로 정리하면 다음 의사코드로 표현할 수 있다.

```text
Algorithm: Depthwise Separable Convolution Forward
Inputs:
  - Input feature map X (N x Cin x H x W)
  - Depthwise kernels Kd (Cin groups, spatial k x k)
  - Pointwise kernels Kp (1 x 1, Cin -> Cout)
1. Z = DepthwiseConv(X; groups=Cin)          # channel-wise spatial conv
2. Y = PointwiseConv(Z; kernel_size=1)      # channel mixing
Output: Y (N x Cout x H' x W')
```

이 의사코드는 논문이 Fig. 4에서 제시하는 극단적 분해의 연산적 의미를 그대로 반영한다.

#### Parameter Count And FLOPs 관점의 정리
파라미터 수가 줄어드는 것과 실제 연산량(FLOPs)이 줄어드는 것은 서로 강하게 연관되지만, 완전히 같은 개념은 아니다. Xception 논문이 말하는 효율은 두 관점 모두에서 이해할 수 있다.

입력 feature map의 공간 크기를 $H\times W$라고 할 때, 정규 convolution의 대략적 MAC(Multiply-Accumulate) 수는

$$
H\cdot W\cdot k^2\cdot C_{in}\cdot C_{out}
$$

으로 볼 수 있다. 반면 depthwise separable convolution은

$$
H\cdot W\cdot \left(k^2\cdot C_{in} + C_{in}\cdot C_{out}\right)
$$

에 해당한다. $C_{out}$이 충분히 큰 일반적인 구간에서는 $k^2\cdot C_{in}\cdot C_{out}$와 $C_{in}\cdot C_{out}$의 차이가 크게 나타나므로, depthwise separable은 같은 $C_{out}$을 유지하면서도 계산을 줄일 수 있다.

다만 논문이 Table 3에서 보여주듯, 모델 전체 관점의 속도는 단순한 FLOPs만으로 결정되지 않는다. depthwise convolution은 연산 패턴이 정규 convolution과 달라, 당시의 라이브러리 및 커널 최적화 수준에 따라 실제 steps/second가 불리해질 수 있음을 논문이 직접 언급한다.

#### Group Count $g$로 보는 이산 스펙트럼의 수식화
Fig. 3의 segmented convolution을 구현 관점에서 정리하면, channel dimension을 $g$개 그룹으로 나누는 grouped convolution으로 해석할 수 있다. $g$를 그룹 수라고 할 때, grouped convolution의 파라미터 수는 대략

$$
\frac{k^2\cdot C_{in}\cdot C_{out}}{g}
$$

이 된다. 여기서

- $g=1$은 정규 convolution  
- $g$가 커질수록 채널 혼합이 제한된 형태  

로 이동한다. 극단에서 depthwise convolution은 $g=C_{in}$이고, 이때 각 그룹의 출력 채널이 1개로 고정되는 특수한 형태로 이해할 수 있다. 논문이 말하는 연속체는, 결국 이 $g$라는 설계 변수를 통해 정규 convolution에서 depthwise로 이동하는 경로가 존재한다는 주장으로 읽을 수 있다.

---

## 2️⃣ Prior Work

### 🔸 CNN 설계 계보: VGG, Inception, 그리고 설계 레시피의 진화
논문은 CNN 설계가 LeNet에서 AlexNet, VGG로 이어지며 깊이를 키우는 방향으로 발전해 왔음을 상기한다. 이후 Inception 계열(GoogLeNet/Inception V2/V3, Inception-ResNet)은 Network In Network의 영향을 받아, 단순한 stack 구조를 넘어 모듈 단위로 연산을 구성하는 스타일을 정착시켰다.

이 맥락은 Xception이 단지 depthwise separable convolution을 채택한 모델이 아니라, **Inception 계열의 다음 단계**로 제안된다는 논문 흐름을 이해하는 데 필요하다.

#### VGG-Style Stack과의 구조적 유사성
논문은 Xception이 몇 가지 측면에서 VGG-16과 도식적으로 유사하다고 언급한다. 이 말은 Xception이 Inception처럼 복잡한 분기 구조를 유지하는 것이 아니라, 전체적으로는 다시 **선형 stack** 형태로 정리된다는 점을 강조한다.

1. stage 단위로 해상도를 줄이며 채널을 늘린다.  
2. 특정 채널 폭에서 동일한 형태의 블록을 반복한다.  
3. 최종적으로 global pooling과 분류기로 연결한다.  

차이는 각 stage를 구성하는 레이어가 정규 convolution이 아니라 depthwise separable convolution이라는 점이며, 이 차이로 인해 파라미터 사용 방식이 달라진다.

### 🔹 Depthwise Separable Convolution의 등장과 확산
논문은 depthwise separable convolution이 2014년 무렵부터 신경망 설계에서 사용되기 시작했으며, TensorFlow 등 프레임워크에 효율적 구현이 제공되면서 더 넓게 쓰이기 시작했다고 정리한다. 또한 Inception V1/V2의 첫 레이어에 depthwise separable convolution이 사용되었다는 맥락을 언급하며, Xception이 완전히 낯선 연산을 들고온 것이 아니라 **기존 흐름의 일반화**라는 점을 강조한다.

여기서 중요한 구분이 하나 있다. 논문은 deep learning 프레임워크에서 separable convolution이라 부르는 것이 **depthwise separable**을 가리키는 경우가 많으며, 영상처리 문맥에서 separable convolution이라 부르는 **spatially separable(예: 7×1, 1×7)**과는 다르다는 점을 명시한다. Xception은 전자의 의미를 사용한다.

#### Spatial Factorization And Depthwise Factorization의 구분
Inception V3를 읽어본 독자는 7×7 convolution을 7×1과 1×7로 나누는 식의 factorization을 떠올릴 수 있다. 이 경우의 분해는 공간 축($k\times k$)을 두 단계로 쪼개는 것으로, 채널 혼합($C_{in}\to C_{out}$)은 각 단계에서 여전히 일어난다.

반면 depthwise separable convolution에서의 분해는 공간 변환과 채널 혼합을 분리한다. 따라서 두 분해는 방향이 다르다.

- spatial factorization: 공간 convolution을 둘로 쪼개는 방식  
- depthwise factorization: 공간 변환과 채널 혼합을 분리하는 방식  

Xception은 후자를 기반으로 Inception 모듈을 재해석하며, 이 구분을 명확히 해두는 것이 이후 실험에서 논의되는 중간 activation 쟁점을 이해하는 데도 도움이 된다.

#### Depthwise Separable Convolution의 선행 사례 정리
논문은 depthwise separable convolution과 관련된 선행 사례를 비교적 구체적으로 열거한다. 요지는 다음과 같다.

1. 2013년 무렵 Google Brain에서의 연구를 통해 depthwise separable convolution이 AlexNet 변형에 적용되었고, 정확도 및 수렴 속도에서 이득이 보고되었다.  
2. 이후 Inception V1/V2에서 첫 레이어에 depthwise separable convolution이 사용되었다.  
3. MobileNet 같은 효율 모델이 depthwise separable convolution을 핵심 구성 요소로 삼는다.  
4. 다른 연구들 역시 separable convolution을 통해 모델 크기와 연산 비용을 줄이려는 흐름을 이어 왔다.  

이 나열은 Xception이 특정 트릭을 새로 도입한 모델이 아니라, 연산을 분해하는 흐름이 충분히 성숙했다는 전제 하에서 그 분해를 아키텍처 전체로 확장한 모델이라는 논문 포지셔닝을 강화한다.

### 🔸 Residual Connection의 위치: 깊은 Stack을 가능하게 하는 학습 장치
Xception은 전체 구조가 linear stack에 가까운 형태이기 때문에, 학습 안정성이 매우 중요하다. 논문은 Residual Connection을 기존 연구(ResNet)에서 온 핵심 구성요소로 위치시키며, 이후 실험에서 residual이 수렴과 성능에 미치는 영향을 별도로 확인한다.

#### Residual Connection의 필요 조건: 깊이와 단순 반복 구조
Xception의 middle flow는 동일 형태 모듈을 반복하는 구조이며, 논문은 이를 8회 반복한다고 명시한다. 반복이 깊어질수록 최적화는 어려워지기 쉬운데, 이때 residual은

1. gradient가 전파되는 경로를 짧게 만들고  
2. 모듈이 학습해야 하는 변화를 residual 형태로 제한하며  
3. 깊은 구조에서의 수렴 속도를 개선하는  

방향으로 작동한다. 논문이 residual의 효과를 별도 실험으로 분리해 확인하는 것도, Xception 구조가 단순 반복을 핵심으로 삼기 때문이다.

---

## 3️⃣ Xception Architecture

### 🔹 연속체 관점 정식화: Segmented Convolution의 이산 스펙트럼
논문은 정규 convolution과 depthwise separable convolution 사이에 **이산 스펙트럼**이 존재한다고 주장한다. 이 스펙트럼은 공간 convolution을 수행할 때 채널 공간을 몇 개의 독립 segment로 분할하느냐에 의해 매개된다.

- segment가 1개라면, 채널을 섞어서 공간 convolution을 수행하는 정규 convolution에 해당한다.  
- segment가 채널 수만큼(채널당 1개)이라면, depthwise convolution이 된다.  
- Inception 모듈은 그 중간 지점으로서, 몇 백 채널을 3~4개 segment로 나누는 것과 유사한 해석이 가능하다.  

이 관점이 논문 전체의 설계 동기이며, Xception은 이 스펙트럼의 극단을 직접 설계로 채택한다.

### 🔸 Depthwise Separable Convolution 정의: Depthwise 이후 Pointwise
논문에서 depthwise separable convolution은 다음 두 연산의 합성으로 정리된다.

1. **Depthwise Convolution**: 입력의 각 채널에 대해 독립적으로 공간 convolution을 수행한다.  
2. **Pointwise Convolution(1×1)**: depthwise 출력 채널들을 다시 선형 결합해 새로운 채널 공간으로 사상한다.  

이를 논문 용어대로 정리하면 depthwise convolution은 spatial correlation을, pointwise convolution은 cross-channel correlation을 주로 담당하도록 분해한 것이다.

#### Depthwise–Pointwise 순서와 중간 Non-Linearity의 쟁점
논문은 Inception의 연산 순서(1×1 이후 spatial)와, 일반적 depthwise separable 구현의 순서(depthwise 이후 1×1)가 다르다는 점을 지적한다. 다만 스택 형태로 반복 사용되는 상황에서는 순서 차이가 본질적이지 않을 수 있다고 주장한다.

더 중요한 쟁점은 **중간 activation의 유무**다. Inception에서는 각 단계 뒤에 ReLU가 들어가지만, depthwise separable convolution은 종종 중간 non-linearity 없이 구현된다. 논문은 이 차이가 학습에 영향을 줄 수 있음을 지적하고, 이후 실험(Fig. 10)으로 이를 검증한다.

#### Intermediate Activation의 위치: Depthwise 이후 Pointwise 사이
논문이 실험으로 검증하는 중간 activation은, depthwise와 pointwise 사이에 들어가는 non-linearity를 의미한다. 이 지점은 다음 이유로 민감하다.

1. depthwise 단계는 채널별 공간 변환이므로, 채널 간 정보 교환이 없다.  
2. pointwise 단계는 채널 혼합을 수행하므로, 채널 간 정보 결합이 이 단계에서 집중된다.  
3. 두 단계 사이에서 activation을 적용하면, 채널 혼합 이전에 정보가 잘릴 수 있다.  

논문은 Fig. 10에서 ReLU/ELU 등을 넣는 것이 오히려 해롭다는 결과를 통해, Xception 계열에서는 중간 activation이 필수적이지 않음을 주장한다.

### 🔹 Xception의 핵심 가설: 채널 상관관계와 공간 상관관계의 완전 분리 가능성
논문은 Xception의 가설을 다음처럼 선언한다. Feature map에서의 cross-channel correlation과 spatial correlation은 **완전히 분리해 매핑할 수 있다**. 즉, 분해를 중간 정도로만 수행하는 Inception보다, 분해를 극단으로 밀어붙인 depthwise separable convolution이 더 적합할 수 있다는 것이다.

이 가설은 Xception이라는 이름과 바로 연결된다. Inception 가설을 더 강하게 만든 형태이므로, Extreme Inception이라는 의미에서 Xception이라고 부른다.

### 🔸 전체 구조: Entry Flow, Middle Flow, Exit Flow
논문은 Xception을 3개의 흐름으로 설명한다.

1. **Entry Flow**: 입력에서 초기 특징을 추출하고 해상도를 줄이며 채널 수를 키운다.  
2. **Middle Flow**: 동일한 형태의 모듈을 여러 번 반복해 표현력을 확장한다.  
3. **Exit Flow**: 최종 특징을 정리하고 분류 head로 연결한다.  

논문은 Fig. 5에서 Xception 아키텍처를 제시하며, 데이터가 entry → middle(8회 반복) → exit 순으로 흐른다고 명시한다. 또한 다이어그램에서는 생략되지만, 모든 Convolution 및 Separable Convolution 뒤에 Batch Normalization이 존재한다고 설명한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/ab95f75c-1b1c-424e-9028-5059bc5b12bb/image.png" width="70%">
</p>

#### Entry–Middle–Exit 분해의 설계 효과
Entry, Middle, Exit로 구조를 분해하는 방식은 Inception 계열에서 이미 자주 등장하지만, Xception에서는 분기 구조가 사라지고 선형 스택 형태로 정리되었기 때문에 이 분해가 더 직접적으로 작동한다.

- Entry Flow는 해상도를 빠르게 줄이며 저수준 패턴을 채널 축으로 확장해, 이후 depthwise separable stack이 다룰 표현 공간을 만든다.  
- Middle Flow는 동일한 채널 폭에서 반복되는 변환을 통해 표현력을 누적한다. 반복 구조가 강하므로, residual의 역할이 두드러진다.  
- Exit Flow는 마지막 downsampling 이후 채널 확장을 통해 분류 head로 전달할 고수준 특징을 정리한다.  

논문이 전체 base를 36 convolutional layer로 설명하고 이를 14 modules로 묶어 말하는 것도, 실제로는 이런 반복 가능한 단위가 명확하다는 점을 강조하기 위함으로 읽을 수 있다.

### 🔹 모듈 구성: Residual Connection을 두른 Depthwise Separable Stack
논문은 Xception의 feature extraction base가 36개의 convolutional layer로 구성되어 있고, 이를 14개의 모듈로 묶는다고 설명한다. 그리고 첫/마지막 모듈을 제외한 모든 모듈에 linear residual connection을 둔다고 명시한다.

이 구성이 주는 설계적 효과는 다음처럼 해석할 수 있다.

1. 모듈 단위의 반복은 구조적 단순성을 만든다.  
2. residual은 깊은 반복에서의 최적화를 안정화한다.  
3. depthwise separable은 각 레이어의 채널/공간 처리 역할을 분해한다.  

#### BN 표기 생략이 의미하는 것: 연산 단위의 반복 패턴
논문은 Fig. 5에서 BN을 다이어그램에서 생략한다고 명시한다. 이 문장은 단순한 편의가 아니라, Xception의 구현 단위가 사실상

- Separable Convolution  
- Batch Normalization  
- Activation  

의 반복 패턴으로 구성된다는 사실을 강조한다. 즉, Xception을 이해할 때는 블록 내부의 세부 연산 나열보다, depthwise separable stack이라는 큰 형태와 그 위에 residual이 감싸는 구조를 중심으로 읽는 것이 적절하다.

#### Feature Extraction Base와 Classifier Head의 결합 방식
논문은 36개의 convolutional layer가 feature extraction base를 형성하고, 그 뒤에 logistic regression layer를 둔다고 설명한다. 또한 선택적으로 logistic regression 이전에 fully-connected layer를 삽입할 수 있음을 언급하며, 이는 JFT 실험에서 분기 비교의 형태로 등장한다.

이 구조는 다음처럼 해석할 수 있다.

1. base는 공간 해상도를 줄이며 고수준 특징을 만든다.  
2. head는 이 특징을 클래스 점수로 선형 분류한다.  
3. FC 삽입 여부는 head의 용량을 바꾸며, 데이터 규모에 따라 효과가 달라질 수 있다.  

---

## 4️⃣ Experimental Evaluation

### 🔸 평가 프로토콜: Single Crop, Single Model 기준 비교
논문은 ImageNet 결과를 single crop, single model로 평가했음을 명시한다. 또한 Inception V3에서 선택적으로 사용할 수 있는 auxiliary tower는 비교의 단순성을 위해 사용하지 않았다고 언급한다.

#### Controlled Comparison의 의미
이 평가는 단순한 관행적 보고가 아니라, 논문이 세우려는 논증 구조와 직접적으로 연결된다. 논문은 Xception과 Inception V3의 성능 차이를 모델 용량의 차이가 아니라 구조적 차이로 해석하려고 한다. 따라서

- 동일한 single crop 조건으로 inference time augmentation 효과를 배제하고  
- single model로 ensemble 효과를 배제하고  
- auxiliary tower를 제거해 Inception 쪽의 추가 정규화 기제를 제거한다  

는 통제 조건을 명시한 것으로 이해할 수 있다. 이 통제는 Table 1–3의 비교가 해석 가능하도록 만드는 최소한의 장치다.

### 🔹 데이터셋: ImageNet과 JFT
논문은 두 데이터셋에서 성능을 비교한다.

- **ImageNet**: 공개 분류 데이터셋, 논문은 validation set 결과를 보고한다.  
- **JFT**: 약 3.5억 이미지와 1.7만 클래스의 대규모 내부 데이터셋. 평가 지표는 FastEval14k의 MAP@100이며, 클래스별 빈도 기반 가중을 포함한다고 설명한다.  

Xception이 ImageNet보다 JFT에서 더 큰 개선을 보인다는 점은, 구조가 더 일반적으로 좋은 것인지 혹은 특정 레시피에 덜 맞춰진 것인지에 대한 해석 포인트가 된다.

#### FastEval14k And MAP@100의 정의적 관점
논문은 FastEval14k가 약 14,000장의 이미지로 구성되며, 클래스가 약 6,000개이고 이미지당 평균 라벨 수가 36.5개라고 명시한다. 이는 단일 라벨 분류(ImageNet)와 달리, 한 이미지가 여러 클래스를 동시에 갖는 multi-label setting임을 의미한다.

MAP@100은 각 이미지에서 상위 100개 예측을 사용해 Average Precision을 계산하고, 이를 평균내는 방식이다. 표준적인 정의를 기준으로 쓰면 다음처럼 정리할 수 있다.

한 이미지 $x$에 대해 모델이 예측한 클래스 순위 리스트를 $c_1,\dots,c_{100}$이라 하고, 정답 라벨 집합을 $Y$라 하자. 그러면 precision@k는

$$
P(k)=\frac{\left|\{c_i\mid i\le k,\ c_i\in Y\}\right|}{k}
$$

로 정의할 수 있다. 이를 사용해 Average Precision@100을

$$
AP@100 = \frac{1}{|Y|}\sum_{k=1}^{100} P(k)\cdot \mathbb{1}[c_k\in Y]
$$

로 둘 수 있고, MAP@100은 데이터셋 전체에 대해 $AP@100$을 평균낸 값으로 이해할 수 있다. 논문은 여기에 더해, 클래스별로 자주 등장하는 라벨에 더 큰 가중을 두는 가중 MAP@100을 사용한다고 설명한다.

### 🔸 최적화 및 정규화 구성: ImageNet 대비 JFT 설정
논문은 ImageNet과 JFT에서 서로 다른 최적화 구성을 사용했다고 명시한다.

#### ImageNet 최적화
- Optimizer: SGD  
- Momentum: 0.9  
- Initial Learning Rate: 0.045  
- Learning Rate Decay: 2 epoch마다 0.94 비율로 감소  

#### JFT 최적화
- Optimizer: RMSProp  
- Momentum: 0.9  
- Initial Learning Rate: 0.001  
- Learning Rate Decay: 3,000,000 samples마다 0.9 비율로 감소  

또한 weight decay는 Inception V3의 4e-5가 Xception에 부적절했고 1e-5를 사용했다고 밝힌다. Dropout은 ImageNet에서는 0.5로 사용했지만, JFT에서는 데이터가 매우 커 과적합 가능성이 낮아 사용하지 않았다고 서술한다.

#### 동일 최적화 설정 사용의 해석
논문은 중요한 단서를 하나 더 둔다. ImageNet과 JFT 각각에 대해, Xception과 Inception V3는 동일한 최적화 설정을 사용했으며, 이 설정은 Inception V3에 맞춰 튜닝된 것이라고 명시한다. 이는

- Xception이 Inception V3의 최적화 레시피 하에서도 경쟁력이 있음을 보여주면서  
- 동시에 Fig. 6에서 관측되는 training profile 차이가, Xception에 대해 최적이 아닐 수 있음을 인정하는  

방향의 서술이다. 따라서 ImageNet에서의 소폭 개선은 구조적 차이가 충분히 큰 개선을 만들지 못했다는 결론이라기보다, 최적화 측면의 여지가 남아 있다는 형태로 읽을 수 있다.

#### Polyak Averaging At Inference의 의미
논문은 두 데이터셋 모두에서 inference 시 Polyak averaging을 사용했다고 명시한다. Polyak averaging은 학습 과정에서 얻은 파라미터들을 단순히 마지막 step의 파라미터로 고정하지 않고, 여러 step에 걸쳐 평균낸 파라미터로 평가하는 방식이다. 직관적으로는

- SGD 계열 최적화가 만드는 파라미터 진동을 평균화해  
- 더 평평한 지역의 해를 선택하도록 유도해  
- 일반화 성능을 소폭 안정화하는  

효과를 기대할 수 있다. 논문이 이를 명시한 것은, Table 1–2의 수치가 단순히 마지막 체크포인트가 아니라 특정한 평가 레시피에 기반한다는 점을 독자가 추적할 수 있게 하기 위함이다.

#### Training Infrastructure And Convergence Status
논문은 실험 인프라와 학습 방식까지 비교적 구체적으로 공개한다. 모든 네트워크는 TensorFlow로 구현했으며, 각 실험은 60개의 NVIDIA K80 GPU에서 학습했다. 다만 데이터셋에 따라 병렬화 방식이 달랐다.

1. ImageNet: data parallelism + synchronous gradient descent  
2. JFT: asynchronous gradient descent  

논문은 ImageNet에서는 최고 성능을 위해 synchronous를 선택했고, JFT에서는 학습 속도를 위해 asynchronous를 사용했다고 설명한다. 또한 실험 소요 시간도 함께 제시한다.

1. ImageNet: 실험당 약 3일  
2. JFT: 실험당 1개월 이상  

그리고 JFT는 완전 수렴까지 학습하지 않았으며, 완전 수렴에는 실험당 3개월 이상이 걸릴 것이라고 언급한다. 이 단서는 Table 2의 절대값 해석에 영향을 준다. JFT 결과는 최종 수렴값이라기보다, 주어진 시간 예산 아래에서의 비교로 이해하는 편이 더 안전하다.

### 🔹 ImageNet 비교: Inception V3 대비 소폭 개선
논문은 Table 1에서 ImageNet single crop 성능을 비교한다. VGG-16 및 ResNet-152는 맥락 제공용으로 함께 제시되며, 비교의 핵심은 Inception V3와 Xception이다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/c2292a29-9aec-4e26-a8a9-b4c70634c207/image.png" width="40%">
</p>

이 표가 전달하는 논지는 두 가지다.

1. Xception은 Inception V3와 파라미터 규모가 유사한 조건에서 Top-1/Top-5 모두 개선을 보인다.  
2. 개선 폭은 ImageNet에서는 크지 않지만, 구조적 대체(Inception→Depthwise Separable)가 유효함을 보여준다.  

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/e9b38e54-29f7-415b-8cd4-c957ad1c37e4/image.png" width="40%">
</p>

#### Table 1 해석: 절대 성능과 설계 논증의 연결
Table 1의 핵심 비교는 Inception V3와 Xception의 근접한 스케일에서의 성능 차이다. Top-1이 0.782에서 0.790으로, Top-5가 0.941에서 0.945로 이동하는 것은 큰 폭의 도약은 아니지만, 다음 사실을 동시에 만족한다.

1. 파라미터 수가 비슷한 상태에서의 개선이므로, 단순히 더 큰 모델을 쓴 결과로 환원하기 어렵다.  
2. Inception 모듈의 분기 구조를 제거하고도 성능이 유지되며 오히려 좋아질 수 있음을 보여준다.  

이 두 조건이 충족되면, 논문이 주장하는 Extreme Inception이라는 관점이 단지 비유가 아니라 실제 설계 가설로서 기능한다는 근거가 된다.

### 🔸 JFT 비교: 큰 데이터에서의 더 큰 이득
논문은 JFT에서 Xception이 Inception V3 대비 4.3%의 상대 개선을 보인다고 정리한다. Table 2는 fully-connected layer 포함 여부에 따라 네 가지 조합을 비교한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/28927a52-3bf7-4278-8b18-1e73b64df505/image.png" width="40%">
</p>

논문은 JFT에서의 개선이 ImageNet보다 훨씬 큰 이유를, Inception V3가 ImageNet을 목표로 설계되어 **ImageNet에 더 맞춰져 있을 가능성**으로 해석한다. 반대로 JFT에는 두 모델 모두 특별히 튜닝된 구조가 아니므로, 구조 자체의 효율 차이가 더 드러날 수 있다는 논리다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/b1be1944-ee63-40c2-8072-604da94e2a66/image.png" width="60%">
</p>

#### Table 2 해석: Head 용량과 Backbone 구조의 상호작용
Table 2는 backbone 구조(Inception V3 vs Xception)뿐 아니라, head에 fully-connected layer를 추가할지 여부까지 함께 비교한다. 이 조합 비교는 다음과 같은 해석을 가능하게 한다.

1. No FC 조건에서도 Xception이 개선을 보이므로, 개선이 head 용량 때문이 아니라 backbone 변환 자체와 관련됨을 시사한다.  
2. With FC 조건에서도 Xception이 개선을 유지하므로, head를 키워도 backbone 구조 차이가 사라지지 않음을 보여준다.  
3. FC 추가 자체는 Inception V3에서 6.36→6.50, Xception에서 6.70→6.78로 이득이 있으나, 그 이득의 크기는 Xception 구조의 이득보다 작다.  

즉, 논문이 말하는 구조적 효율은 head 조정으로 흡수되는 종류의 효과가 아니라, backbone의 연산 분해 방식에서 비롯된 효과로 읽을 수 있다.

### 🔹 모델 크기와 속도: 파라미터는 유사, 속도는 근소하게 느림
논문은 성능 비교가 단지 수치 향상에 그치지 않도록, Table 3에서 크기/속도를 함께 보고한다. 파라미터 수는 ImageNet(1000 classes, FC 없음) 기준이며, 속도는 60개의 K80 GPU에서 synchronous SGD로 측정한 steps/second다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/dd6f850c-320d-4675-b5e8-269426e04ea2/image.png" width="40%">
</p>

논문은 두 모델의 파라미터 수가 매우 유사하므로, ImageNet 및 JFT에서 관측된 개선은 **모델 용량 증가가 아니라 파라미터 사용의 효율**에서 비롯된다는 결론을 강화한다. 또한 depthwise convolution 구현 최적화가 진행되면 Xception이 더 빨라질 수 있다는 전망을 제시한다.

#### Table 3 해석: 속도 수치의 읽는 법
Table 3에서 steps/second가 31에서 28로 감소한 것은, Xception이 더 적은 파라미터를 가지면서도 반드시 더 빠르지 않을 수 있음을 보여준다. 이는 depthwise convolution이 이론적으로는 계산량을 줄이는 방향이지만, 실제 GPU 커널에서는

- 작은 연산이 많이 쪼개져 메모리 접근 비중이 커지거나  
- 연산 병렬화가 정규 convolution만큼 효율적으로 되지 않거나  
- 프레임워크가 depthwise kernel을 충분히 최적화하지 못하는  

상황이 발생할 수 있기 때문이다. 논문이 이 점을 한계로도 언급하는 것은, 설계 아이디어의 가치가 단순히 속도 개선에만 있지 않으며, 성능과 단순화라는 다른 축의 이득도 함께 본다는 태도를 드러낸다.

### 🔸 Residual Connection 효과: 수렴 속도 및 최종 성능에 핵심적 역할
논문은 Xception에서 residual connection을 제거한 변형을 ImageNet에서 비교해, residual이 수렴 속도와 최종 성능 모두에 중요하다는 결론을 제시한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/bdc6922d-ac60-41d6-a97c-ac4814d31e82/image.png" width="40%">
</p>

이 실험은 단순한 결론을 말해준다. Xception은 depthwise separable convolution을 쌓는 설계를 취하지만, 그 설계를 깊게 만들기 위해서는 residual이 사실상 필수에 가깝다.

#### Residual Ablation의 논증 구조
Fig. 9의 비교는 단순히 성능이 좋아진다는 수준을 넘어서, Xception이라는 설계가 무엇에 의존하는지를 분해해 보여준다. depthwise separable convolution 자체는 표현을 분해하는 연산이지만, 그 연산을 수십 층 쌓았을 때 최적화가 가능한지는 별개의 문제다.

따라서 이 ablation은 다음을 확인하는 역할을 한다.

1. Xception의 성능이 depthwise separable이라는 연산 단일 요소만으로 설명되지 않음을 확인한다.  
2. 깊은 반복 구조를 가능하게 하는 최적화 장치로서 residual이 필수적임을 보여준다.  

논문이 Xception을 VGG-Style stack에 비유한 맥락과 연결하면, VGG가 깊이를 확보하는 방식이 단순 레이어 누적이었다면, Xception은 그 단순 누적을 residual로 뒷받침해 학습 가능한 형태로 만든다고 정리할 수 있다.

### 🔹 중간 Activation 효과: Depthwise–Pointwise 사이 ReLU/ELU의 부정적 영향
논문은 depthwise separable convolution과 Inception 모듈의 유사성이 중간 activation 포함을 암시할 수 있음을 언급한다. 그러나 실제로는 ReLU 또는 ELU를 중간에 넣는 것이 오히려 성능을 해친다고 보고한다. 즉, **중간 activation이 없는 설정이 더 빠른 수렴과 더 좋은 최종 성능**을 만든다는 결과를 제시한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/bc5ae5ee-8e15-4b8e-8cb5-7ed80a8ed6c5/image.png" width="40%">
</p>

논문은 이 결과를 중간 feature space의 깊이가 얕을수록 non-linearity가 정보 손실을 유발할 수 있다는 직관으로 연결한다.

#### Intermediate Activation 실험의 직관적 해석
Depthwise–Pointwise 사이의 activation은 채널 혼합 이전에 적용된다는 점에서 특별하다. depthwise 단계는 채널별로 독립적인 공간 변환이기 때문에, 이 단계의 출력은 채널 간 상호작용 없이 만들어진다. 이때 ReLU 같은 비선형이 먼저 적용되면

1. 채널별 공간 특징에서 음수 성분이 소거되고  
2. 이후 pointwise에서 채널들을 섞어 복합적인 조합을 만들 기회가 줄어들며  
3. 결과적으로 표현 공간이 불필요하게 제한될 수 있다  

는 방향의 설명이 가능하다. 반대로 activation을 pointwise 이후로 미루면, 채널 혼합이 먼저 일어난 뒤 비선형이 적용되므로 채널 조합의 표현력이 더 넓게 유지될 수 있다.

이 실험은 Xception이 단지 연산을 분해한다는 주장에 그치지 않고, 분해된 연산들 사이에 비선형을 어디에 둘지까지 포함해 설계가 완성된다는 점을 강조하는 역할을 한다.

---

## 5️⃣ Future Directions

### 🔸 연속체의 중간 지점 탐색: Depthwise Separable이 최선이라는 보장은 없음
논문은 정규 convolution과 depthwise separable convolution 사이의 이산 스펙트럼을 다시 상기시키며, Xception이 택한 극단이 최적이라는 보장은 없다고 말한다. 오히려 Inception과 depthwise separable 사이의 중간 지점에 추가 이점이 있을 수 있으며, 이는 후속 연구 과제로 남긴다.

이 섹션은 Xception이 단지 특정 아키텍처를 제시하는 것에서 끝나지 않고, **설계 공간 자체를 재정의**하려는 시도를 포함한다는 점에서 중요하다.

#### Intermediate Group Setting의 탐색 관점
논문이 말하는 중간 지점은 결국 group count $g$를 어떻게 두느냐의 문제로 구체화할 수 있다. $g$가 증가하면 파라미터 수는 줄어들지만, 채널 혼합이 제한된다. 따라서 중간 지점 탐색은

- 채널 혼합을 얼마나 자주, 어떤 위치에서 수행할지  
- spatial 변환이 채널 간 상호작용에 얼마나 의존하는지  
- 분해로 인한 효율 이득과 표현력 손실이 어디서 균형을 이루는지  

를 체계적으로 살펴보는 문제로 바뀐다. Xception의 주장처럼 Inception이 연속체의 중간 지점 중 하나라면, Inception의 특정 분기 설계가 우연히 좋은 것이 아니라, 더 일반적인 $g$-스펙트럼 상의 한 선택으로 이해될 수 있다.

#### Hardware Efficiency 관점의 후속 과제
Table 3의 속도 결과는 설계 공간 탐색이 정확도만으로 끝나지 않음을 시사한다. 같은 FLOPs라도 실제 throughput은 커널 최적화 수준에 크게 의존할 수 있으므로, $g$를 조정하는 탐색은

- 수학적 효율(FLOPs, params)  
- 시스템 효율(커널 효율, 메모리 접근 패턴)  

을 함께 고려해야 한다. 논문이 미래 과제를 남긴 시점에는 depthwise kernel 최적화가 충분히 성숙하지 않았음을 감안하면, 중간 group setting이 오히려 더 높은 실제 속도를 낼 가능성도 논리적으로 배제할 수 없다.

---

## 6️⃣ Conclusions

### 🔹 결론 요약: Inception 대체로서의 Depthwise Separable Stack
논문 결론은 다음 흐름으로 정리된다.

1. 정규 convolution과 depthwise separable convolution은 스펙트럼의 양 끝이며, Inception은 그 중간 지점이다.  
2. 이 관찰로부터 Inception 모듈을 depthwise separable convolution으로 치환한 Xception을 제안한다.  
3. Xception은 Inception V3와 유사한 파라미터 수에서 ImageNet에서는 소폭, JFT에서는 큰 성능 향상을 보인다.  
4. depthwise separable convolution은 Inception이 가진 성질을 유지하면서도 사용이 단순하므로, 향후 아키텍처 설계의 중요한 구성 요소가 될 수 있다.  

#### 결론의 논증 방식: 가설, 설계, 실험, 분해
Xception 결론이 인상적인 이유는, 단순히 새로운 블록을 제안하고 끝나는 구조가 아니라 논증 흐름이 비교적 명확하기 때문이다.

1. 가설: 채널 상관관계와 공간 상관관계는 분리 가능하다는 Inception 가설의 확장  
2. 설계: 그 분리를 극단으로 밀어 depthwise separable stack을 구성  
3. 실험: ImageNet과 JFT에서 스케일을 맞춘 비교로 구조적 차이를 관찰  
4. 분해: residual 및 intermediate activation을 분리 실험으로 검증  

특히 4단계의 분해 실험은, 설계가 단일 아이디어의 결과가 아니라 여러 선택의 결합임을 보여주며, 독자가 후속 설계를 할 때 무엇을 고정하고 무엇을 바꿔야 하는지에 대한 힌트를 제공한다.

---

## 💡 해당 논문의 시사점과 한계
Xception의 의의는 Inception을 단지 잘 설계된 모듈로 보지 않고, **연산 분해의 관점**에서 재정의했다는 데 있다. Inception이 채널·공간 상관관계의 부분적 분해를 수행한다면, Xception은 이를 극단으로 밀어붙여 완전 분해에 가까운 형태를 취한다. 이 덕분에 아키텍처는 오히려 단순해지고, 깊은 반복과 residual을 결합한 선형 스택 형태로 정리된다.

실험적으로는 파라미터 수가 유사한 조건에서 개선을 보이며, 특히 더 큰 데이터에서 개선 폭이 커지는 점이 구조적 효율 주장에 힘을 싣는다. 또한 중간 activation의 유무, residual의 유무를 별도 실험으로 분해해, 설계 선택의 필요성을 논증하려고 한다.

한계로는 depthwise convolution의 구현 효율이 하드웨어/프레임워크 최적화에 민감할 수 있고, 실제로 논문에서도 Xception이 당시 기준으로 약간 느리다고 보고한다. 또한 논문 스스로도 depthwise separable이 스펙트럼의 최적점이라는 보장은 없다고 인정하며, 중간 지점 탐색을 미래 과제로 남긴다.

#### 설계 해석의 의의: Module Engineering에서 Operator Factorization으로
Inception 계열은 종종 모듈 설계가 경험적 레시피처럼 보이기 쉽다. Xception은 이를 채널/공간 상관관계 분해라는 관점으로 재해석함으로써, 모듈 설계가 임의적 선택이 아니라 연산 구조의 한 점으로 설명될 수 있음을 보여준다. 이는 후속 설계에서

- 분기 수나 tower 구성 같은 표면적 선택보다  
- 채널 혼합과 공간 변환의 결합 정도를 먼저 결정하고  
- 그 결정을 구현 가능한 블록 형태로 내리는  

방향의 사고를 유도한다는 점에서 의미가 있다.

#### 실험 설계의 의의: 데이터 규모에 따른 구조의 일반성 점검
논문이 JFT를 함께 사용하는 것은, 모델이 ImageNet에 튜닝된 레시피의 산물인지 여부를 확인하기 위함이다. JFT에서의 더 큰 개선은 Xception 구조가 특정 데이터셋의 특수성에 덜 의존하고, 더 일반적인 구조적 효율을 제공할 가능성을 시사한다.

다만 JFT 실험이 완전 수렴까지 학습된 것이 아니라는 점은, 결과 해석에서 항상 염두에 두어야 한다. 이 경우 학습 곡선(Fig. 7–8)의 형태가 중요해지며, 초기 구간의 수렴 속도 차이가 보고된 MAP@100 차이로 이어졌을 가능성도 충분히 있다.

---

## 👨🏻‍💻 Xception Lucid 구현
파이썬 라이브러리 [`lucid`](https://github.com/ChanLumerico/lucid) 속 구현된 Xception [`xception.py`](https://github.com/ChanLumerico/lucid/blob/main/lucid/models/imgclf/xception.py)을 살펴보자. 구현 해설은 다음 순서로 진행한다.

1. `ConvBNReLU2d`가 Stem에서 수행하는 결합 연산
2. Depthwise Separable Convolution이 Lucid에서 어떤 연산으로 구현되는지
3. Xception의 핵심 블록인 `_Block`의 구성과 residual 경로  
4. `Xception` 클래스의 Entry/Middle/Exit 대응 구조  
5. 모델 등록 함수 `xception`  

### 0️⃣ `nn.ConvBNReLU2d`: Stem의 Convolution + Batch Normalization + ReLU 결합
`Xception` 클래스의 stem은 `ConvBNReLU2d`를 사용해 초기 3×3 convolution 두 개를 구성한다.

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

이 모듈은 `Conv2d → BN → ReLU` 패턴을 하나의 클래스로 묶는다. 논문 Fig. 5에서도 convolution 뒤에 batch normalization이 붙는 패턴이 반복되며, Lucid 구현은 이 반복을 `ConvBNReLU2d`와 `_Block` 내부의 BN 배치로 구현한다.

### 1️⃣ `nn.DepthSeparableConv2d`: Depthwise 이후 Pointwise
Xception의 핵심 연산은 `nn.DepthSeparableConv2d`다. Lucid의 `_DepthSeparableConv`를 통해 depthwise와 pointwise를 다음처럼 구현한다.

```python
class _DepthSeparableConv(nn.Module):
    D: ClassVar[int | None] = None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if self.D is None:
            raise ValueError("Must specify 'D' value.")

        self.depthwise = _Conv[self.D - 1](
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = _Conv[self.D - 1](
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        self.reversed = reversed

    def forward(self, input_: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(input_))
```

여기서 논문 정의와 1:1로 대응되는 포인트는 다음과 같다.

- depthwise 단계는 `groups=in_channels`로 설정된 convolution이며, **채널별 독립 공간 convolution**을 구현한다.  
- pointwise 단계는 `kernel_size=1`인 convolution이며, **채널 공간 사상(혼합)**을 구현한다.  
- forward는 `depthwise → pointwise` 순서이며, 논문이 말한 일반적 구현 순서와 동일하다.  

논문에서 논의한 중간 activation의 부재는 Lucid 구현에서도 그대로다. `DepthSeparableConv2d` 내부에는 ReLU/ELU가 존재하지 않으며, activation은 블록 단위에서 외부적으로 배치된다.

#### `self.reversed` 필드의 성격
`_DepthSeparableConv`에 있는 `self.reversed = reversed`는 이 클래스의 forward 경로에서 직접 사용되지는 않는다. 따라서 Xception 구현을 이해하는 관점에서는 핵심 요소가 아니며, 연산 정의와도 무관하다. 다만 코드가 실제로 포함하고 있는 필드이므로, Lucid 구현과 논문 정의가 만나는 지점을 확인할 때는 depthwise/pointwise 두 conv와 그 순서를 중심으로 보면 충분하다.

### 2️⃣ `_Block`: Depthwise Separable Stack과 Residual Skip
`_Block`은 Xception의 모듈 단위를 구현한다. 입력/출력 채널과 반복 횟수(`reps`), 다운샘플링(`stride`) 및 블록 내부의 배치 규칙을 인자로 받는다.

#### Skip 경로 정의: 채널/해상도 불일치 시 1×1 Projection
`_Block.__init__`은 `out_channels != in_channels` 또는 `stride != 1`이면 skip 경로에 1×1 convolution과 BN을 둔다.

```python
if out_channels != in_channels or stride != 1:
    self.skip = nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False
    )
    self.skipbn = nn.BatchNorm2d(out_channels)
else:
    self.skip = None
```

이는 ResNet의 projection shortcut과 동일한 역할을 한다. 다운샘플링이 있거나 채널 수가 바뀌면, skip 경로를 통해 shape을 맞춘 뒤 더한다.

#### Main 경로 구성: `reps`에 따른 반복 DepthSeparableConv2d
`rep` 리스트를 구성하는 핵심은 두 개의 플래그다.

- `start_with_relu`: 블록 첫 연산에서 ReLU를 둘지 여부  
- `grow_first`: 채널 확장을 블록 초반에 둘지 여부  

`grow_first=True`이면, 첫 번째 separable conv에서 `in_channels → out_channels`로 채널을 먼저 확장하고(`channels = out_channels`로 갱신), 이후 `reps-1`번은 `channels → channels` 반복을 수행한다.

```python
if grow_first:
    rep.append(nn.ReLU())
    rep.append(
        nn.DepthSeparableConv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
    )
    rep.append(nn.BatchNorm2d(out_channels))
    channels = out_channels

for i in range(reps - 1):
    rep.append(nn.ReLU())
    rep.append(
        nn.DepthSeparableConv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
    )
    rep.append(nn.BatchNorm2d(channels))
```

반대로 `grow_first=False`이면, 반복을 수행한 뒤 블록 마지막에서 `in_channels → out_channels`로 채널을 맞추는 형태로 구성된다.

```python
if not grow_first:
    rep.append(nn.ReLU())
    rep.append(
        nn.DepthSeparableConv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
    )
    rep.append(nn.BatchNorm2d(out_channels))
```

#### ReLU 시작 조건과 Downsampling
`start_with_relu`에 따라 첫 ReLU를 제거하거나 유지한다.

```python
if not start_with_relu:
    rep = rep[1:]
else:
    rep[0] = nn.ReLU()
```

또한 `stride != 1`이면 블록 끝에 MaxPool을 추가한다.

```python
if stride != 1:
    rep.append(nn.MaxPool2d(kernel_size=3, stride=stride, padding=1))
```

이 구성은 논문 Fig. 5에서 entry/exit flow에서 다운샘플링이 들어가는 모듈들의 역할과 대응된다.

#### Forward: Main 경로와 Skip 경로의 합
`_Block.forward`는 main 경로를 통과한 뒤, skip을 더한다.

```python
def forward(self, x: Tensor) -> Tensor:
    out = self.rep(x)

    if self.skip is not None:
        skip = self.skip(x)
        skip = self.skipbn(skip)
    else:
        skip = x

    out += skip
    return out
```

즉 Lucid의 `_Block`은 논문이 강조한 residual connection을 모듈 단위로 강제하는 구현이다.

### 3️⃣ `Xception`: Entry Flow, Middle Flow×8, Exit Flow 대응
`Xception` 클래스는 논문 Fig. 5의 큰 흐름을 코드로 조립한다.

#### Stem: 두 개의 초기 Convolution
초기 부분은 `ConvBNReLU2d`를 사용해 3×3 convolution 두 번으로 채널을 32, 64로 늘린다.

```python
self.conv1 = nn.ConvBNReLU2d(
    3, 32, kernel_size=3, stride=2, padding=0, conv_bias=False
)
self.conv2 = nn.ConvBNReLU2d(32, 64, kernel_size=3, conv_bias=False)
```

여기서 `conv1`은 stride 2로 해상도를 줄이고, `conv2`는 stride 1로 추가 변환을 수행한다.

#### Entry Flow: 채널 확장과 다운샘플링을 포함한 3개 블록
entry flow는 `_Block` 3개로 구성된다.

```python
self.block1 = _Block(64, 128, reps=2, stride=2, start_with_relu=False)
self.block2 = _Block(128, 256, reps=2, stride=2)
self.block3 = _Block(256, 728, reps=2, stride=2)
```

각 블록은 stride 2로 다운샘플링을 수행하면서 채널을 128, 256, 728로 확장한다. `block1`이 `start_with_relu=False`인 것은 블록 경계에서 activation 배치를 조정하기 위한 설정으로 읽을 수 있다.

#### Middle Flow: 동일 채널(728)에서 반복되는 8개 블록
middle flow는 동일한 `_Block(728, 728, reps=3)`를 8회 반복한 `nn.Sequential`이다.

```python
self.mid_blocks = nn.Sequential(*[_Block(728, 728, reps=3) for _ in range(8)])
```

논문이 middle flow 반복을 강조하는 지점과 동일하게, 구현도 반복 구조가 매우 명확하다.

#### Exit Flow: 채널 확장(1024→1536→2048)과 분류 Head
exit flow의 첫 단계는 `end_block`이다.

```python
self.end_block = _Block(728, 1024, reps=2, stride=2, grow_first=False)
```

이후 두 개의 depthwise separable conv와 BN, ReLU를 적용해 채널을 1536, 2048로 확장한다.

```python
self.conv3 = nn.DepthSeparableConv2d(1024, 1536, kernel_size=3, padding=1)
self.bn3 = nn.BatchNorm2d(1536)

self.conv4 = nn.DepthSeparableConv2d(1536, 2048, kernel_size=3, padding=1)
self.bn4 = nn.BatchNorm2d(2048)
```

마지막은 global average pooling과 linear classifier다.

```python
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
self.fc = nn.Linear(2048, num_classes)
```

#### Forward 전체 흐름
forward는 논문 도식의 순서를 그대로 따른다.

```python
def forward(self, x: Tensor) -> Tensor:
    x = self.conv1(x)
    x = self.conv2(x)

    x = self.block3(self.block2(self.block1(x)))
    x = self.mid_blocks(x)
    x = self.end_block(x)

    x = self.relu(self.bn3(self.conv3(x)))
    x = self.relu(self.bn4(self.conv4(x)))

    x = self.avgpool(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fc(x)
    return x
```

여기서 중요한 구현 포인트는 다음이다.

1. Entry flow는 `block1 → block2 → block3`로 직렬 적용된다.
2. Middle flow는 `self.mid_blocks`로 반복이 추상화된다.
3. Exit flow는 `end_block` 이후 separable conv + BN + ReLU 2회로 마무리된다.
4. 분류 head는 `AdaptiveAvgPool2d((1,1)) → flatten → Linear`이다.

즉, 논문이 강조한 Xception의 장점 중 하나인 구조적 단순성이 코드 구조에서도 그대로 나타난다.

### 4️⃣ 모델 등록 함수 `xception`
Lucid는 `@register_model` 데코레이터로 모델을 등록한다.

```python
@register_model
def xception(num_classes: int = 1000, **kwargs) -> Xception:
    return Xception(num_classes, **kwargs)
```

이 함수는 `Xception` 인스턴스를 생성해 반환하며, 외부에서는 registry를 통해 `xception` 이름으로 모델을 생성할 수 있다.

---

## ✅ 정리
Xception은 Inception 모듈을 연산 분해의 관점에서 재해석하고, 그 분해를 극단으로 밀어붙인 depthwise separable convolution을 Inception 모듈의 대체물로 제안한다. 아키텍처는 entry/middle/exit flow의 선형 스택 형태로 정리되며, 모듈 단위 residual connection을 광범위하게 사용해 학습을 안정화한다. 실험에서는 Inception V3와 유사한 파라미터 수에서 ImageNet에서 소폭의 개선을, JFT에서 더 큰 개선을 보고해 파라미터 효율 주장을 강화한다. 또한 residual connection과 중간 activation의 효과를 별도 실험으로 분해해 설계 선택의 필요성을 논증한다.

- 핵심 관점: Inception–Depthwise Separable 연속체(이산 스펙트럼)  
- 핵심 연산: Depthwise Separable Convolution(Depthwise + Pointwise)  
- 핵심 결과: ImageNet 소폭 개선, JFT 큰 개선(Table 1, Table 2)  
- 구현 대응: `DepthSeparableConv2d` + `_Block` residual 모듈 + `Xception`의 Entry/Middle/Exit 조립  

#### 📄 출처
Chollet, François. "Xception: Deep Learning With Depthwise Separable Convolutions." *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, arXiv:1610.02357.
