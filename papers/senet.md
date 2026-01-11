# [SENet] Squeeze-and-Excitation Networks
SENet은 ResNet이나 Inception처럼 하나의 새로운 backbone을 제안한다기보다, 기존 CNN의 블록에 **채널 단위 주의(attention)** 에 가까운 _부착형 모듈_ 을 더해 표현력을 끌어올리는 방법론으로 보는 편이 정확하다. 논문의 출발점은 간단하다. convolution은 공간 방향으로는 local receptive field를 통해 특징을 학습하지만, **채널 간 의존성(channel-wise dependency)** 을 충분히 명시적으로 모델링하지 못한다. 결과적으로 같은 spatial 위치라도 어떤 채널을 더 믿고 덜 믿을지(또는 클래스/입력에 따라 어떤 특징을 강화/억제할지)가 암묵적으로만 학습된다.

저자들은 이를 **Squeeze-and-Excitation(SE) block**으로 해결한다. SE block은 feature map의 공간 정보를 채널별로 요약해(global pooling) 전역적인 채널 descriptor를 만들고(squeeze), 이 descriptor를 작은 게이트 네트워크로 통과시켜 채널별 중요도 벡터를 만든 뒤(excitation), 원 feature map의 채널을 그 중요도로 스케일링해 재보정(recalibration)한다. 핵심은 이 중요도 벡터가 **입력마다 동적으로** 변한다는 점이다. 따라서 동일한 backbone이라도 입력/클래스에 따라 채널 사용 패턴이 달라지고, 결과적으로 더 강한 표현을 만들 수 있다는 것이 논문 주장이다.

이 리뷰는 논문 전개를 따라 SE block의 정의와 수식을 정리하고, 다양한 backbone(ResNet/ResNeXt/Inception/VGG 및 모바일 네트워크)과 데이터셋(ImageNet/CIFAR/Places365/COCO detection)에서 성능이 어떻게 개선되는지 다양한 표를 기반으로 해석한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/5b4de56b-55f5-481b-b353-f58c609b4706/image.png" width="70%">
</p>

SENet이 흥미로운 이유는, 아이디어 자체가 엄청 복잡해서라기보다 **작고 단순한 연산을 블록 수준의 표준 구성 요소로 격상**시켜 버렸다는 데 있다. global average pooling은 이미 분류기 앞에서 자주 쓰이던 연산이고, FC 두 개도 특별한 기술이 아니다. 그런데 이 둘을 조합해 **채널별 중요도 벡터를 입력마다 만들어내고**, 그걸로 feature map을 다시 스케일링하는 형태로 만들면, 네트워크가 스스로 어떤 채널을 강조해야 하는지 훨씬 직접적으로 학습할 수 있게 된다.

또 하나의 포인트는 SE가 흔히 말하는 _attention_ 과 닮아 있지만, 전형적인 spatial attention(어떤 위치를 볼지)과는 축이 다르다는 점이다. SE는 위치 $(H,W)$가 아니라 **채널 $C$에 대한 self-attention**에 가깝다. 그리고 이 attention 신호가 local receptive field에 갇히지 않도록, squeeze에서 전역 정보를 요약해 넣는다. 이 구조 때문에 논문은 SE를 단순한 추가 레이어가 아니라 **feature recalibration의 메커니즘**으로 해석한다.

마지막으로, 이 방법론은 하나의 특정 backbone만을 전제로 하지 않는다. residual branch 출력이든, Inception 모듈 출력이든, 혹은 VGG의 conv 블록 출력이든 간에, 형태만 맞으면 SE를 끼워 넣어 같은 방식으로 재보정을 수행할 수 있다. 논문이 여러 backbone과 여러 데이터셋을 넓게 훑으며 실험을 구성한 이유도, 이 attachment 관점의 일반성을 보여주기 위해서라고 이해할 수 있다.

---

## 1️⃣ 논문 배경

### 🔹 채널 의존성의 문제
CNN의 feature map은 $(C, H, W)$ 형태의 3차원 텐서로 생각할 수 있고, 여기서 채널 축 $C$는 서로 다른 필터(혹은 feature detector)들이 만든 응답을 의미한다. 표준 convolution은 채널을 선형 결합해 다음 층의 채널을 만들기 때문에, 채널 간 상호작용 자체가 전혀 없다고 말할 수는 없다. 하지만 그 상호작용은 주로 **고정된 가중치**에 의해 이루어지고, 입력 내용에 따라 어떤 채널을 더 강조해야 하는지(채널 선택의 동적 변화)를 명시적으로 분리해 모델링하기는 어렵다.

논문은 이 점을 다음의 직관으로 연결한다. 같은 이미지라도 클래스/문맥에 따라 유용한 특징 채널이 달라질 수 있고, 심지어 같은 클래스 내부에서도 개체 인스턴스(각도, 배경, 조명)에 따라 어떤 채널이 중요해지는지 달라질 수 있다. 그렇다면 모델이 입력에 따라 채널 단위 스케일을 조절(이 입력에서는 이 채널을 더 크게, 저 채널은 더 작게)할 수 있으면 표현력이 올라갈 가능성이 크다. SENet은 이 조절을 매우 작은 모듈로 구현한다.

#### 특징 조합과 경쟁의 관점
채널을 단순히 필터들이 만든 응답의 나열로 보면, 채널 간 관계는 다음 conv가 알아서 섞는다고 생각하기 쉽다. 하지만 실제로는 같은 층의 여러 채널이 서로 **보완적인 조합 특징**을 만들기도 하고, 반대로 특정 입력에서는 서로 **경쟁하는 특징**이 되기도 한다. 예를 들어 물체의 질감(texture) 채널과 형태(shape) 채널이 동시에 활성화될 수도 있지만, 배경이 복잡한 경우에는 특정 질감 채널만 과도하게 켜져 오히려 분류에 방해가 될 수 있다. 이때 채널 중요도를 입력마다 조정할 수 있으면, 동일한 convolution 가중치로도 더 다양한 상황을 안정적으로 처리할 수 있다.

#### 정적 스케일(학습된 상수)과 동적 스케일(입력 조건부)의 차이
채널별 스케일링 자체는 새로운 개념이 아니다. 예컨대 BatchNorm의 $\gamma$처럼 채널마다 곱해지는 계수는 이미 널리 쓰인다. 다만 $\gamma$는 입력과 무관한 **정적 파라미터**이고, SE가 만드는 $s_c$는 입력에 따라 달라지는 **동적 게이트**다. 같은 네트워크라도 샘플마다 $s$가 달라질 수 있으므로, SE는 일종의 조건부 computation처럼 동작한다. 논문이 dynamic이라는 단어를 반복해서 강조하는 이유가 여기에 있다.

#### 논문이 강조하는 Dynamic의 의미
SE block이 만드는 중요도 벡터는 고정 파라미터가 아니라 입력에서 계산되는 값이다. 즉, 동일한 네트워크라도 입력 샘플마다 채널별 스케일이 달라질 수 있고, 이는 feature recalibration이 정적 re-weighting이 아니라 **동적 게이팅**에 가깝다는 것을 의미한다. 이후 Section 7의 분석에서 저자들은 실제로 깊이에 따라 excitation이 더 class-specific해지는 경향과, stage 마지막에서 saturation되는 경향 등을 관찰하며 이 동적 성질을 구체적으로 보여준다.

#### SE가 학습에 도움이 되는 이유
SE는 최종적으로 $\widetilde{\mathbf{u}}_c = s_c\,\mathbf{u}_c$ 형태의 곱을 수행한다. 이때 $s_c$는 입력에서 계산되는 값이므로, 네트워크는 **어떤 채널이 유용한지를 직접 반영하는 방향**으로 $s$를 만들도록 학습될 수 있고, 동시에 그 $s$를 만들어내는 게이트 네트워크가 더 좋은 채널 의존성 표현을 학습하도록 역전파된다. 특히 residual 구조에서는 identity shortcut이 항상 남아 있기 때문에, 채널이 강하게 억제되는 상황에서도 학습이 완전히 막히지 않는 경로가 존재한다는 점이 실전적으로는 중요하다.

### 🔸 다양한 Backbone에 부착되기 쉬운 이유
SE block은 어떤 변환 $F_{tr}$의 출력에 후처리로 붙는다. 여기서 $F_{tr}$은 residual branch일 수도 있고(Inception 모듈 전체일 수도 있다), 일반적인 convolution block일 수도 있다. 즉, SE는 backbone을 새로 설계하는 대신, 기존 네트워크가 만들어낸 intermediate representation을 **채널 단위로 재보정**하는 방식이다.

이 설계가 잘 붙는다는 의미는 두 가지다.

1. SE는 입력/출력 텐서의 공간 크기 $H, W$를 바꾸지 않고 채널별 스케일만 곱한다. 따라서 기존 블록의 I/O 인터페이스를 거의 바꾸지 않는다.
2. 추가 연산은 global pooling과 작은 FC(또는 1×1 conv)로 이루어져 FLOPs 증가가 작다. 즉, 성능 향상 대비 비용이 낮아 다양한 backbone에 부착하기 좋다.

논문은 이런 특성을 바탕으로 ResNet/ResNeXt/Inception-ResNet-v2/VGG/BN-Inception뿐 아니라 MobileNet/ShuffleNet, 그리고 다운스트림(Places365, COCO detection)까지 광범위한 실험을 통해 일반성을 보이려 한다.

#### 모듈 부착이 쉬운 이유 1: 텐서 형태를 보존하는 연산
SE의 출력은 입력과 같은 $C\times H\times W$ 형태를 유지한다. 관점에 따라 SE는 feature map에 **대각 행렬(diagonal matrix)** 형태의 채널별 gain을 곱하는 연산으로 볼 수 있다. 중요한 것은 이 gain이 입력에서 계산된다는 점이고, 그렇기 때문에 모듈을 끼워 넣더라도 앞뒤 블록의 인터페이스가 깨지지 않는다. 구조를 바꾸지 않고도 representation의 성질만 바꾸는 방식이므로, 기존 backbone 설계 철학과 충돌이 적다.

#### 모듈 부착이 쉬운 이유 2: Residual과의 궁합
Residual block에 SE를 붙이면, residual branch의 출력이 add로 합쳐지기 전에 한 번 재보정된다. 이는 residual branch에서 어떤 채널이 중요했는지 반영해 shortcut과 합쳐지는 비율을 조정하는 효과로도 해석할 수 있고, add 이후에 비선형이 붙는 구조에서 채널 간 스케일이 과도하게 커지거나 작아지는 현상을 완화하는 형태로도 볼 수 있다. 논문이 Fig. 3에서 SE-ResNet 모듈을 도식화하고, 이후 ablation에서 삽입 위치를 비교하는 흐름은 이 직관을 실험으로 확인하려는 시도다.

#### 이 논문이 의도적으로 강조하는 실무적 메시지
SE는 개별 데이터셋이나 특정 모델에서만 통하는 트릭이라기보다, 기존 backbone에 들어가던 **블록을 대체하는 형태**로 쉽게 적용할 수 있다는 점이 큰 장점이다. 그래서 논문은 classification 성능뿐 아니라 mobile setting, 장면 분류, detection까지 실험을 넓힌다. 결과를 단순히 숫자 비교로 끝내기보다는, 어떤 비교가 공정한지(re-implementation), 어떤 변수가 핵심인지(reduction ratio, non-linearity, 삽입 위치)를 단계적으로 분해해 보여준다.

---

## 2️⃣ Related Work

### 🔹 Channel Attention과 Feature Recalibration의 배경
SE의 핵심은 **channel-wise gating**이지만, 논문은 이를 독립적인 새로운 발명이라기보다 여러 흐름과 연결해서 위치시킨다. 대표적으로

- global average pooling을 통한 채널 요약(분류기 앞의 GAP)  
- gating/attention 계열 아이디어(특징 선택)  
- 네트워크 내부의 컨텍스트 활용(전역 정보로 local feature를 보정)

같은 요소들이 이전부터 부분적으로 존재했다. 하지만 SENet의 차이는 이런 요소들을 **블록 단위로 표준화**해 어디에나 붙일 수 있는 모듈로 만든 점이다. 즉, 어떤 특정 task에만 쓰이는 attention이 아니라, backbone 설계 단위의 primitive로 제안한다.

#### 설계 단위로서의 SE: 관련 연구를 바라보는 관점
논문은 SE를 완전히 새로운 계열의 attention으로 포장하기보다는, 기존에 흩어져 있던 요소들을 **하나의 표준 블록 패턴**으로 정리해 보여준다. global average pooling은 전역 정보를 뽑는 가장 단순한 방법이고, gating은 특징 선택을 구현하는 가장 기본 형태다. 그런데 이 둘을 조합해 채널별 게이트를 만들면, 채널 의존성이 블록 내부에서 $\mathbf{z},\mathbf{s}$ 같은 중간 변수로 명시되고, 그 결과를 중심으로 설계 선택을 체계적으로 비교할 수 있게 된다.

또한 채널 축 attention의 성격은 공간 위치를 고르는 attention과 다르다. SE는 위치 $(H,W)$가 아니라 **채널 $C$에 대한 self-attention**에 가깝고, 그래서 여러 채널이 동시에 강조되는 상황(조합 특징)을 자연스럽게 처리해야 한다. 실제로 논문은 excitation이 **non-mutually-exclusive** 관계를 학습해야 한다고 말하며, 이 관점은 softmax보다 sigmoid 게이팅을 선택하는 이유로 이어진다.

#### 채널 의존성을 명시적으로 분리하는 시도
일반적으로 convolution은 공간과 채널을 동시에 다루는 연산이고, 채널을 통한 상호작용은 고정 가중치의 선형 결합으로 처리된다. SENet은 이 상호작용을 채널 중요도 벡터라는 형태로 분리해 계산하고, 그 벡터를 통해 feature map을 재스케일한다. 이렇게 하면 채널 관계를 학습하는 부분(FC 게이트)이 블록 내부에서 명시적으로 드러나고, 그 결과를 해석하거나 ablation하기가 쉬워진다.

### 🔸 다양한 Backbone에서 공통적으로 재사용되는 이유
ResNet 이후 많은 연구가 보여주듯, backbone의 구조적 변화는 classification뿐 아니라 detection/segmentation 등으로 **전이되는 경우가 많다**. SE block은 이 전이 흐름에 특히 적합하다. 왜냐하면 SE는 backbone의 공간적 구조를 건드리지 않고, 채널 recalibration이라는 표현 학습의 성질을 바꾸기 때문이다. 따라서 ResNet에 붙이면 SE-ResNet, ResNeXt에 붙이면 _SE-ResNeXt_, Inception 모듈에 붙이면 _SE-Inception_ 형태로 동일한 원리가 재사용된다.

#### 변환 $F_{tr}$의 출력 위에 SE 얹기
Fig. 2/3의 핵심은 SE가 특정 레이어 하나(conv 하나)에만 붙는 게 아니라, 설계자가 선택한 변환 단위 $F_{tr}$ 전체를 하나의 블록으로 보고 그 출력에 **채널 게이팅**을 적용할 수 있다는 점이다.

- Inception에서는 여러 branch를 concat해서 나온 모듈 출력 전체가 $\mathbf{U}$가 되고, 그 $\mathbf{U}$에 SE가 붙는다.
- Residual에서는 residual branch의 출력이 $\mathbf{U}$가 되고, SE로 재보정된 뒤 shortcut과 더해진다.

이 관점이 유지되면, architecture가 달라져도 SE의 수식과 직관(전역 요약 → 채널 게이트 → 채널 스케일링)은 거의 그대로 재사용된다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/b8a45afa-8c88-4daf-ac89-e425ef580edb/image.png" width="70%">
</p>

---

## 3️⃣ Squeeze-and-Excitation Blocks

### 🔹 기본 표기: Feature Map과 변환 $F_{tr}$
논문은 어떤 변환 $F_{tr}$이 입력 feature map을 출력 feature map으로 사상한다고 두고, 그 출력이 SE block의 입력이 된다고 설명한다. 일반적으로 CNN의 중간 feature map은

$$
\mathbf{U} \in \mathbb{R}^{C \times H \times W}
$$

로 표기된다. 여기서 $C$는 채널 수, $H,W$는 공간 해상도다. 각 채널의 feature map을 $\mathbf{u}_c \in \mathbb{R}^{H\times W}$로 두면, SE는 $\mathbf{U}$로부터 채널별 중요도 벡터 $\mathbf{s}\in\mathbb{R}^{C}$를 만들고, 이를 통해 $\mathbf{U}$를 재보정한 출력 $\widetilde{\mathbf{U}}$를 만든다.

SE의 목표는 $\mathbf{U}$에 담긴 채널 응답을 그대로 다음 블록에 넘기는 것이 아니라, 채널 간 관계를 고려해 어떤 채널은 키우고 어떤 채널은 줄여서 **더 유리한 representation**을 만들어내는 것이다.

#### $F_{tr}$를 도입하는 이유
논문이 $F_{tr}$이라는 추상화를 쓰는 이유는, SE를 단지 conv-activation 같은 미시적 레이어에만 붙이는 것이 아니라 Inception 모듈이나 residual unit 같은 **설계 단위** 위에도 자연스럽게 붙일 수 있게 만들기 위해서다. 입력을 $\mathbf{X}$라고 두면,

$$
\mathbf{U} = F_{tr}(\mathbf{X})
$$

처럼 변환 출력 $\mathbf{U}$가 만들어지고, SE는 $\mathbf{U}\mapsto \widetilde{\mathbf{U}}$라는 후처리를 수행한다. 여기서 중요한 제약은 shape 보존이다. $\widetilde{\mathbf{U}}$는 $\mathbf{U}$와 같은 $C\times H\times W$ 형태를 유지하므로, SE는 $F_{tr}$의 의미(어떤 변환을 했는지)를 바꾸지 않고도 채널 관점에서 **표현을 재보정**할 수 있다.

### 🔸 Squeeze: 전역 정보를 채널 Descriptor로 압축하기
squeeze 단계는 공간 축 $(H,W)$를 채널별로 요약해, 채널 descriptor $\mathbf{z}\in\mathbb{R}^{C}$를 만든다. 논문이 기본으로 쓰는 연산은 global average pooling이다. 채널 $c$에 대해

$$
z_c = \frac{1}{H\cdot W}\sum_{i=1}^{H}\sum_{j=1}^{W} u_c(i,j) \tag{2}
$$

로 정의할 수 있다. 이 연산은 각 채널이 이미지 전체에 대해 얼마나 활성화되는지(평균 응답)를 요약한다. 결과적으로 $\mathbf{z}$는 입력 이미지의 전역 문맥을 반영한 채널 요약이며, **excitation 단계의 입력**이 된다.

#### Squeeze가 실제로 하는 일: 공간 정보를 버리고 전역 문맥을 남기기
Global average pooling은 각 채널의 공간적 활성화 패턴을 하나의 스칼라로 요약한다. 이는 정보 손실을 동반한다. 하지만 논문이 원하는 것은 위치별로 다른 가중치가 아니라, **이 채널이 현재 입력에서 전반적으로 유용한가**라는 전역적 신호다. SE는 채널별로 동일한 스케일을 $(H,W)$ 전 위치에 곱하므로, 채널 게이트를 만들 때도 공간 좌표마다 다른 값을 만들 필요가 없고, 전역 요약이 설계 목적과 잘 맞는다.

또한 squeeze는 단순히 평균을 내는 것처럼 보이지만, 결과적으로 각 채널이 입력 전체에서 얼마나 강하게 반응했는지에 대한 통계를 제공한다. 이 통계가 다음 단계에서 채널 간 관계를 학습하는 데 들어가므로, squeeze는 채널 attention이 **local receptive field를 넘어선 문맥을 반영하도록** 만드는 핵심 연결 고리다.

#### Global Average Pooling을 사용하는 이유
논문은 squeeze 단계에서 더 복잡한 집계 연산도 가능하지만, 글로벌 평균 풀링이 간단하고 효과적이어서 기본 선택으로 삼는다. 이후 ablation에서 global max pooling과 비교해도 큰 차이는 없지만 average가 약간 더 낫다는 결과를 보고한다. 중요한 것은, squeeze가 채널 간 관계를 계산하는 데 필요한 전역 정보를 제공한다는 점이며, 단순한 pooling만으로도 의미 있는 성능 차이가 나타난다는 것이 논문의 주장이다.

### 🔹 Excitation: 채널별 게이트 학습하기
Excitation 단계는 $\mathbf{z}$로부터 채널별 중요도 벡터 $\mathbf{s}$를 만든다. 논문은 이를 게이팅 메커니즘으로 설명하고, 두 개의 FC 레이어와 비선형을 사용한다. 가장 기본적인 형태는 다음과 같다.

$$
\mathbf{s} = \sigma\big(W_2\,\delta(W_1\mathbf{z})\big) \tag{3}
$$

여기서

- $W_1 \in \mathbb{R}^{\frac{C}{r}\times C}$는 채널 차원을 줄이는 FC(감소 비율 $r$)  
- $W_2 \in \mathbb{R}^{C\times \frac{C}{r}}$는 채널 차원을 복원하는 FC  
- $\delta$는 ReLU, $\sigma$는 sigmoid

로 해석할 수 있다. 감소 비율 $r$는 계산량과 파라미터를 줄이기 위한 hyperparameter이며, 기본값으로 $r=16$을 사용한다.

#### 논문이 Excitation에 요구하는 두 조건: 유연함 + 비(非)배타성
논문 텍스트는 excitation을 설계할 때 두 가지 조건을 명시한다. 채널 간 상호작용을 학습할 수 있을 정도로 충분히 유연해야 하고(특히 비선형 상호작용), 여러 채널이 동시에 강조될 수 있도록 **non-mutually-exclusive** 관계를 학습해야 한다는 것이다. 이 관점에서 sigmoid 게이팅은 자연스럽다. softmax처럼 한 채널을 올리면 다른 채널을 반드시 내리는 경쟁 구조가 아니라, 필요한 채널들을 동시에 올려 줄 수 있기 때문이다.

#### Bottleneck이 들어가는 이유
만약 $\mathbf{z}\in\mathbb{R}^{C}$에 대해 단일 FC로 $\mathbf{s}\in\mathbb{R}^{C}$를 만들면, 파라미터가 $C^2$로 커진다. SE는 중간 차원을 $C/r$로 줄였다가 다시 늘리는 bottleneck을 넣어 파라미터를 대략 $2C^2/r$로 낮춘다. 이때 $r$은 단순히 비용만 줄이는 값이 아니라, 채널 의존성을 얼마나 압축해서 표현할지(표현력 vs 효율)를 정하는 손잡이이기도 하다. 그래서 논문은 Table 10에서 $r$을 바꿔가며 정확도/파라미터 균형이 어떻게 달라지는지 확인한다.

#### Excitation이 채널 의존성을 모델링하는 방식
중요한 점은, $\mathbf{s}$의 각 원소 $s_c$는 입력 $\mathbf{z}$의 모든 채널 정보를 보고 결정된다는 것이다(FC가 모든 채널을 섞기 때문). 즉, 특정 채널의 중요도가 다른 채널들의 활성화 패턴에 의존할 수 있게 된다. 이 점이 SE가 단순한 채널별 스케일 파라미터(고정 스케일)와 다른 이유다.

또한 sigmoid를 통해 $s_c\in(0,1)$ 범위로 제한되므로, excitation은 본질적으로 채널을 억제하거나(0에 가까움) 유지/강화하는(1에 가까움) 게이트 역할을 하게 된다. ablation에서 sigmoid를 ReLU나 tanh로 바꾸면 성능이 나빠지는 결과는, 이 게이트의 형태가 성능에 중요하다는 것을 보여준다.

#### 입력 조건부라는 말의 의미
Excitation을 통해 만들어지는 $\mathbf{s}$는 입력 $\mathbf{U}$에서 유도된 $\mathbf{z}$의 함수다. 즉, 같은 네트워크 파라미터 $W_1,W_2$를 갖고 있어도 입력이 달라지면 $\mathbf{s}$가 달라진다. 논문이 이 구조를 self-attention으로 해석하는 이유도 여기에 있다. 핵심은 attention 신호가 local receptive field에만 갇히지 않도록 squeeze에서 전역 요약을 넣었다는 점이고, 그래서 결과적으로 채널 간 관계가 **더 긴 범위의 문맥을 반영**할 수 있게 된다는 주장이다.

### 🔸 Scale: Feature Map을 채널별로 재보정
마지막으로 SE는 원 feature map을 채널별로 스케일링한다. 채널 $c$에 대해

$$
\widetilde{\mathbf{u}}_c = s_c \cdot \mathbf{u}_c \tag{4}
$$

로 정의되며, 전체 텐서로 보면 _channel-wise broadcasting_ 을 통한 곱셈이다. 이 단계가 **feature recalibration**의 핵심이다. squeeze/excitation이 계산한 채널 중요도 벡터가 실제로 feature map을 바꾸는 유일한 지점이기 때문이다.

#### Scale 단계의 해석: 입력 조건부 채널 Gain
Scale은 계산 자체는 단순한 곱이지만, 의미는 크다. $s_c$는 (0,1) 범위의 스칼라이므로 각 채널은 입력에 따라 억제되거나 유지된다. 이를 채널별 gain으로 보면, SE는 블록 출력 $\mathbf{U}$의 각 채널을 입력마다 다시 정규화해 주는 역할을 한다. 그리고 이 gain은 conv 가중치처럼 고정된 것이 아니라, squeeze에서 만든 전역 문맥 요약을 바탕으로 **동적으로 계산**된다.

또 다른 관점은, $\mathbf{s}$가 채널 축에 대한 대각 행렬을 정의한다는 것이다. 즉, 같은 $\mathbf{U}$라도 입력에 따라 다른 대각 스케일링이 적용된다. 이 대각 스케일링은 채널 간 선형 결합을 새로 만드는 것이 아니라, 이미 만들어진 채널들을 **선택적으로 통과/억제**하는 형태이기 때문에, 구조 변경 없이도 표현을 안정적으로 개선할 수 있다는 직관으로 이어진다.

#### SE를 실제 네트워크에 끼우는 방식
논문은 SE block이 표준 conv 블록뿐 아니라 여러 변환 단위 위에 올릴 수 있는 모듈임을 강조한다.

- **VGG 계열**: 각 convolution 뒤의 비선형을 통과한 출력에 SE를 붙여, conv 블록마다 채널 재보정을 수행한다.
- **Inception 계열**: 여러 branch를 합친 Inception 모듈 전체를 $F_{tr}$로 보고, 모듈 출력에 SE를 붙인다(Fig. 2).
- **Residual 계열**: residual branch의 출력에 SE를 붙인 뒤 shortcut과 더한다(Fig. 3).

여기서 공통점은, 어디에 붙이든 SE의 수식은 변하지 않고, 단지 무엇을 $\mathbf{U}$로 볼지만 바뀐다는 점이다. 그래서 논문은 이를 instantiation이라고 부르고, 다양한 backbone을 빠르게 SE-variant로 바꿀 수 있음을 장점으로 내세운다.

#### SE Block을 이해하기 위한 최소 체크리스트
SE를 구현/해석할 때 헷갈리기 쉬운 포인트를 논문 흐름대로 체크리스트로 정리하면 다음과 같다.

- 입력/출력 텐서 형태는 동일해야 한다: $\mathbf{U},\widetilde{\mathbf{U}}\in\mathbb{R}^{C\times H\times W}$
- squeeze는 공간 축을 제거해 채널 descriptor를 만든다: $\mathbf{z}\in\mathbb{R}^{C}$
- excitation은 $\mathbf{z}$의 모든 채널 정보를 섞어 $\mathbf{s}\in\mathbb{R}^{C}$를 만든다
- 게이트는 non-mutually-exclusive해야 한다: 여러 채널을 동시에 강조할 수 있어야 한다
- reduction ratio $r$은 비용과 표현력을 동시에 조절한다: 너무 작으면 비싸고, 너무 크면 약해진다(Table 10)
- sigmoid는 단순한 선택이 아니라 게이트의 의미를 보장한다(Table 12)
- scale은 채널별 곱으로 구현된다: 공간 위치마다 다른 가중치가 아니라 채널별 동일 스케일이다
- residual에 통합할 때는 add 이전이 자연스럽다(Table 14)
- stage별 적용은 누적될 수 있고, 모든 stage가 반드시 동일할 필요는 없다(Table 13, Fig. 6/7)

#### SE Block Pseudocode
SE의 forward 흐름을 논문 정의에 맞춰 의사코드로 정리하면 다음과 같다.

```text
Algorithm: Squeeze-and-Excitation (SE) block
Input: U in R^{C x H x W}, reduction ratio r
Parameters: W1 in R^{C/r x C}, W2 in R^{C x C/r}

z = GlobalAvgPool(U)                 # z in R^{C}
s = Sigmoid(W2(ReLU(W1(z))))         # s in R^{C}
U_tilde = U * s                      # channel-wise scaling (broadcast over H,W)
return U_tilde
```

이 알고리즘은 공간 정보를 요약해 채널에 대한 전역 문맥을 만든 뒤, 그 문맥으로 채널을 재보정한다는 SE의 핵심 메시지를 그대로 반영한다.

---

## 4️⃣ 계산 복잡도

### 🔹 FLOPs 증가가 작은 이유
SE block이 추가하는 연산은 크게 두 가지다.

1. global average pooling: $C$개의 채널에 대해 $H\cdot W$를 평균내는 연산  
2. 두 개의 FC: $C\rightarrow C/r\rightarrow C$

논문은 ResNet-50에 SE를 붙일 때 GFLOPs 증가가 거의 없음을 강조한다. Table 2 기준으로 ResNet-50의 GFLOPs가 3.86이고, SENet(=SE-ResNet-50)의 GFLOPs가 $3.87$로 거의 같다. 즉, 채널 재보정의 효과를 매우 작은 비용으로 얻는 구조다.

#### FLOPs가 작게 느껴지는 이유
SE의 추가 연산을 conv와 대비해 보면 직관이 분명해진다. conv는 일반적으로 $H\times W$ 공간 전 위치에서 채널 혼합을 수행한다. 특히 3×3 conv는 커널 면적까지 포함돼 연산량이 커지기 쉽다. 반면 SE에서 무거운 부분은 두 개의 FC인데, 이 FC는 $H,W$에 의존하지 않고 **채널 축에서만** 계산된다. squeeze가 $H\times W$를 평균내기는 하지만, 이는 곱-합 구조의 conv에 비해 상대적으로 단순한 집계 연산이며, 전체 네트워크 FLOPs에서 차지하는 비중이 작다.

또한 SE는 블록 출력의 채널을 다시 섞는 것이 아니라 채널별 스케일을 곱하기만 한다. 즉, 복잡한 feature 생성 연산을 추가하는 방식이 아니라, 이미 생성된 feature의 사용 비율을 조절하는 방식이므로, 비용 대비 성능 개선이라는 관점에서 설계 의도가 명확하다.

#### 대략적인 SE Block의 계산 비용
SE block의 각 단계는 다음처럼 매우 단순한 형태로 비용을 생각할 수 있다.

- **Squeeze(GAP)**: 채널마다 $H\cdot W$를 집계하므로 대략 $O(C\cdot H\cdot W)$  
- **Excitation(FC)**: 두 FC의 곱-합이 대략 $O(C\cdot (C/r) + (C/r)\cdot C)=O(2C^2/r)$  
- **Scale**: 입력 feature map에 채널별 스칼라를 곱하므로 대략 $O(C\cdot H\cdot W)$  

여기서 conv의 비용이 대체로 $O(k^2\cdot C_{in}\cdot C_{out}\cdot H\cdot W)$처럼 커널 면적과 채널 곱까지 포함되는 것과 비교하면, SE의 추가 비용이 왜 상대적으로 작게 나타나는지 직관적으로 이해할 수 있다. 특히 excitation의 비용은 $H,W$에 직접 비례하지 않기 때문에, 공간 해상도가 큰 구간에서 conv의 비용이 급격히 커지는 것과 대비된다.

### 🔸 추가 파라미터: reduction ratio $r$로 제어되는 FC 비용
SE의 추가 파라미터는 주로 excitation의 FC 레이어에서 생긴다. $W_1$과 $W_2$의 파라미터 수는 대략

$$
C\cdot \frac{C}{r} + \frac{C}{r}\cdot C = \frac{2C^2}{r}
$$

로 볼 수 있다(bias 포함 여부에 따라 약간 달라질 수 있음). 채널 수가 큰 stage일수록 이 항이 커지므로, 논문은 실제로 파라미터 증가가 주로 네트워크 마지막 stage에서 발생한다고 설명한다.

#### 마지막 Stage가 특히 민감한 이유
채널 수가 커질수록 $C^2$ 항이 빠르게 커지기 때문에, 동일한 $r$을 쓰더라도 마지막 stage의 SE가 비용을 지배한다. 예를 들어 ResNet 계열에서는 깊은 stage로 갈수록 채널이 커지고, 결국 매우 큰 채널 폭을 갖는 구간이 생긴다. 이 구간에서 FC 기반 게이팅의 파라미터가 커지므로, 논문이 마지막 stage SE 제거를 통해 파라미터를 줄이는 실험을 따로 언급하는 것도 자연스럽다.

이 관찰은 단순히 파라미터를 줄이는 팁이 아니라, SE가 어디에서 가장 큰 이득을 주는지(채널 의존성이 강하게 필요해지는 구간)와 어디에서는 포화되어 이득이 줄어드는지(Section 8, Fig. 6/7)의 분석과도 연결된다.

또한 논문은 마지막 stage의 SE block을 제거하면 파라미터 증가를 크게 줄이면서 성능 손실을 0.1% top-5 이내로 제한할 수 있었다고 언급한다. 이는 Section 7의 분석(Fig. 6/7)에서 **마지막 stage excitation이 saturation되는 경향**과도 연결된다. 즉, SE의 비용-효율을 더 올리기 위해 stage별로 SE 사용을 조절할 수 있다는 실무적 힌트가 된다.

---

## 5️⃣ 구현 세부사항

### 🔹 ImageNet 학습 설정
논문은 ImageNet-2012에서 baseline과 SE variant를 동일한 최적화 스킴으로 학습시키는 것을 원칙으로 한다. 주요 설정은 다음과 같다.

- 데이터 증강: random crop(스케일/종횡비), horizontal flip  
- 입력 정규화: RGB 채널 평균 subtraction  
- 최적화: synchronous SGD, momentum 0.9, minibatch 1024  
- 초기 learning rate: 0.6, 30 epoch마다 10배 감소  
- 총 100 epochs 학습  
- 기본 reduction ratio: $r=16$

평가에서는 center crop을 사용한다($224×224$, Inception 계열은 $299×299$). Table 2는 단일 crop 기준의 top-1/top-5 error와 GFLOPs를 함께 제공해, 성능 향상과 비용 증가를 동시에 비교할 수 있게 한다.

#### 단일 Crop의 이유
ImageNet에서는 multi-crop이나 multi-scale 같은 test-time 기법을 쓰면 성능이 올라간다. 하지만 그런 기법은 구조 변화(SE 유무) 외의 요소를 추가로 섞는 효과가 있고, 특히 서로 다른 논문/모델 간에 적용 방식이 다르면 비교가 어려워진다. 그래서 논문은 Table 2에서 단일 crop 결과를 중심으로 제시해, 가능한 한 구조 변화의 효과를 깔끔하게 읽을 수 있도록 한다.

또한 crop 크기를 명시하는 것도 중요하다. 입력 해상도는 성능과 연산량에 직접 영향을 주기 때문에, crop 크기를 바꾸면 SE의 효과가 아니라 입력 해상도 변화가 성능 차이로 나타날 수 있다. Table 2처럼 GFLOPs까지 함께 제시하는 구성은, 이런 혼선을 줄이면서 성능-비용을 동시에 읽게 만들려는 의도로 이해할 수 있다.

#### Table 2의 Re-Implementation과 공정 비교
논문은 기존 논문에서 보고된 **original** 숫자만을 그대로 쓰지 않고, 공정 비교를 위해 baseline을 재학습(re-implementation)한 수치와 SENet 수치를 나란히 제공한다. 이는 학습 레시피나 구현 차이로 인한 변동을 줄이기 위한 장치다. 따라서 SE의 효과를 논문 안에서 읽을 때는 원칙적으로 re-implementation 대비 SENet의 개선 폭(괄호 안)을 중심으로 보는 것이 정확하다.

#### Table 2를 공정하게 읽기 위한 체크리스트
논문에서 baseline 대비 SENet 개선을 해석할 때, 실수하기 쉬운 포인트를 체크리스트로 정리하면 다음과 같다.

- original 숫자와 re-implementation 숫자를 구분해서 본다
- 가능한 한 re-implementation 대비 SENet 변화량을 중심으로 본다
- crop 크기(224 vs 299) 같은 입력 조건이 같은지 확인한다
- 단일 crop인지, multi-crop/multi-scale인지 확인한다
- 성능만 보지 말고 GFLOPs를 같이 본다(특히 backbone이 다르면 필수)
- 표가 말해주는 것은 평균적 경향이며, 실제 설정에서는 $r$이나 삽입 위치 튜닝 여지가 남는다(Table 10, 14)

---

## 6️⃣ 모델 실험

### 🔹 ImageNet Classification: 다양한 backbone에 대한 일관된 개선
ImageNet-1K에서의 핵심 결과는 Table 2다. Table 2는 `ResNet/ResNeXt/VGG/BN-Inception/Inception-ResNet-v2`에 SE를 붙였을 때의 단일 crop error와 GFLOPs를 제공한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/6f558dbf-1b8f-4acc-b2f7-5c0ced68b02d/image.png" width="70%">
</p>

이 표에서 눈에 띄는 패턴은 다음과 같다.

1. SE는 거의 모든 backbone에서 top-1/top-5를 동시에 개선한다.  
2. GFLOPs 증가는 매우 작다(예: ResNet-50 $3.86 → 3.87$ 수준).  
3. 상대적으로 얕은 모델에서도 큰 개선이 보이며, 깊이를 늘려서 얻는 이득이 줄어드는 구간에서도 SE는 여전히 추가 이득을 제공한다.

#### 개선 폭 관점에서의 재서술
아래는 Table 2의 re-implementation 대비 SENet의 개선 폭을, 절대 변화량 기준으로 다시 정리한 것이다(에러는 작아질수록 좋으므로, 음수는 개선을 의미한다).

| Model                    | Top-1 Δ | Top-5 Δ |
|--------------------------|---------|---------|
| `ResNet-50`              | $-1.51$ | $-0.86$ |
| `ResNet-101`             | $-0.79$ | $-0.45$ |
| `ResNet-152`             | $-0.85$ | $-0.61$ |
| `ResNeXt-50 (32×4d)`     | $-1.01$ | $-0.41$ |
| `ResNeXt-101`            | $-0.48$ | $-0.56$ |
| `VGG-16 (BN)`            | $-1.80$ | $-1.11$ |
| `BN-Inception`           | $-1.15$ | $-0.75$ |
| `Inception-ResNet-v2`    | $-0.57$ | $-0.42$ | 

이렇게 적어 보면, SE의 이득은 특정 계열에서만 튀는 것이 아니라 backbone이 바뀌어도 **꾸준히 존재한다는 점**이 더 분명해진다. 그리고 이 일관성이 곧 논문이 강조하는 generalizability 주장(Table 2~7)을 받쳐 준다.

#### VGG-16(BN) 사례
Table 2에서 VGG-16(BN)은 baseline top-5 error $8.81$이 SE 적용 후 $7.70$으로 내려간다. 절대값 기준으로도 변화 폭이 크고, 무엇보다 VGG는 residual connection 같은 안정화 장치가 없는 비교적 고전적인 구조다. 이 결과는 SE가 특정 구조(예: residual)만을 전제로 하지 않고, 단순한 conv 스택에서도 채널 재보정이 의미 있는 개선을 낼 수 있음을 보여준다.

다만 VGG는 GFLOPs 자체가 큰 모델이기 때문에, 이 비교는 성능만이 아니라 비용 관점에서도 함께 읽어야 한다. 논문이 GFLOPs를 같은 표에 함께 보고하는 이유가 여기서 다시 한 번 드러난다. SE는 VGG의 연산량을 근본적으로 줄이진 않지만, 같은 계산량 수준에서 오류를 줄여 주는 방향으로 작동한다.

#### ResNet/ResNeXt에서의 의미
ResNet/ResNeXt 비교에서는, SE가 깊이를 늘리는 방식과는 **다른 축의 개선을 제공**한다는 해석이 가능하다. ResNet-50에 SE를 붙인 모델이 ResNet-101의 성능에 근접한다는 점은, 단순히 파라미터를 늘려서가 아니라 representation을 더 효율적으로 쓰는 구조 변화가 성능에 반영될 수 있음을 보여준다. 물론 최종적으로는 depth와 SE를 같이 쓰는 것도 가능하며, Table 2에서 더 깊은 모델에서도 SE가 꾸준히 이득을 주는 것이 그 근거다.

#### 깊이 대비 효율: SE-ResNet-50 vs ResNet-101
논문이 강조하는 대표 비교는 `SE-ResNet-50`의 top-5 error $6.62$가 `ResNet-101`의 $6.52$에 근접하면서, GFLOPs는 절반 수준($3.87$ vs $7.58$)이라는 점이다. 즉, depth를 늘리는 것과 SE를 붙이는 것이 완전히 동일한 축은 아니며, depth가 제공하는 이득이 줄어드는 구간에서 SE는 다른 방식으로 추가 성능을 제공한다는 해석이 가능하다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/1c0a2ad1-5ee9-468a-bd67-de1cd6c53b35/image.png" width="70%">
</p>

### 🔹 CIFAR-10/100 실험
논문은 CIFAR-10/100에서도 SE가 일반적으로 유효한지 확인한다. Table 4/5는 ResNet-110/164, WideResNet, Shake-Shake(+Cutout) 등과 그 SE 버전을 비교한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/7550fddd-c427-448a-acf7-f59ddeafb45c/image.png" width="40%">
</p>

모든 비교에서 SENet이 baseline보다 낮은 에러를 보인다. 논문은 이를 통해 SE가 특정 backbone에만 특화된 트릭이 아니라, 다양한 데이터셋/태스크에서 재사용 가능한 일반적 모듈이라는 메시지를 강화한다.

#### CIFAR 결과의 해석
Table 4/5의 비교에는 이미 성능이 높은 구성(예: Cutout을 포함한 Shake 계열)도 포함된다. 즉, 학습 레시피 자체가 강해진 상황에서도 SE를 추가하면 에러가 더 내려간다. 이는 SE가 단순히 최적화 트릭을 대체하는 것이 아니라, 네트워크가 representation을 구성하는 방식 자체(_채널 재보정_)에 영향을 준다는 점을 간접적으로 보여준다.

또한 CIFAR는 이미지 해상도가 작고, 모델이 상대적으로 작은 설정에서도 많이 실험된다. 그럼에도 SE가 개선을 제공한다는 것은, 채널 의존성 모델링이 대규모 ImageNet 같은 환경에서만 중요해지는 것이 아니라, 작은 입력에서도 **일반적인 유용성**을 가질 수 있음을 시사한다. 물론 각 숫자의 크기는 설정마다 다르지만, 방향성(개선)이 유지되는 것이 논문이 전달하려는 핵심이다.

### 🔸 Places365: 장면 분류에서의 일반화
Places365는 장면 분류로, 객체 분류와 다른 형태의 일반화 능력을 요구한다. 논문은 `ResNet-152`를 baseline으로 SE를 붙인 `SE-ResNet-152`가 top-5 error를 $11.61 → 11.01$로 낮춘다고 보고한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/2aef1e9c-d2f3-4fad-a953-7e31a73f3483/image.png" width="40%">
</p>

논문은 이 결과를 scene classification에서도 SE가 도움이 된다는 근거로 사용한다.

#### 객체 분류와 다른 일반화 축
장면 분류는 특정 객체 하나의 존재보다 **배경 구성, 질감 패턴, 전역 레이아웃** 같은 신호가 더 중요해질 수 있다. 이 경우 채널들이 담당하는 의미 축도 객체 분류와는 다르게 구성될 가능성이 크다. SE는 전역 문맥을 squeeze로 요약해 채널 게이트를 만들기 때문에, 장면처럼 전역 정보가 중요한 과제에서 특히 잘 맞을 수 있다는 직관을 제공한다.

물론 Table 6의 수치 변화는 ImageNet의 대규모 개선 폭만큼 크진 않지만, 데이터셋이 바뀌어도 개선 방향이 유지된다는 점이 중요하다. 논문은 이를 통해 SE가 특정 데이터셋에만 맞춘 장치가 아니라, _표현 강화 모듈_ 로서 전이 가능하다는 메시지를 쌓아 간다.

### 🔸 ILSVRC 2017: SENet-154와 SOTA 비교
논문은 ILSVRC 2017 classification에서 1위였고, winning entry는 multi-scale/multi-crop ensemble로 top-5 error $2.251$을 얻었다고 언급한다. 또한 단일 모델로 `SENet-154`를 제안하고, Table 8에서 다른 SOTA와 비교한다.

#### SENet-154는 무엇인가: SE + (modified) ResNeXt
논문은 대회 제출 과정에서 추가 모델 `SENet-154`를 구성했다고 설명한다. 핵심은 SE block을 ResNeXt 계열의 backbone(논문 표현으로는 modified ResNeXt)에 통합했다는 점이다. 즉, `SENet-154`는 SE 자체의 원리를 바꾸는 것이 아니라, 경쟁력 있는 backbone 위에 SE를 결합해 더 높은 상한을 노린 설계로 볼 수 있다. 아키텍처 세부는 Appendix에 있다고 언급되는데, 본문 흐름에서는 SE가 다양한 backbone에 부착될 수 있다는 메시지를 대회 결과로 다시 한 번 강조하는 역할을 한다.

또한 논문이 ensemble과 single-model 결과를 구분해 보고하는 방식도 중요하다. ensemble은 강한 test-time 기법과 결합되어 있기 때문에, SE의 순수 구조 효과만을 말하긴 어렵다. 반면 Table 8의 single-crop 비교는 상대적으로 구조적 비교에 가깝고, SE가 SOTA 수준의 경쟁에서도 **유효한 구성 요소**가 될 수 있음을 보여준다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/febcb6ee-bfc7-4b2f-8632-0c93021a02a8/image.png" width="40%">
</p>

#### Table 8의 비교 포인트
Table 8은 두 가지 crop 크기 설정을 나눠 보고한다($224×224$, 그리고 더 큰 크기인 $320×320$ 또는 $299×299$). 입력 해상도가 커지면 일반적으로 성능이 좋아질 수 있지만, 동시에 연산량도 늘어난다. 그래서 이 표를 읽을 때는, 같은 해상도 구간 안에서 `SENet-154`가 어디에 위치하는지를 먼저 보고, 그 다음 해상도를 키웠을 때 모델 간 순위나 격차가 어떻게 변하는지를 보는 편이 깔끔하다.

또한 `SENet-154`는 SE를 단지 ResNet에 붙인 변형이 아니라 _modified ResNeXt_ 와 결합된 모델이다. 즉, Table 8의 순위는 SE만의 효과가 아니라 backbone 선택과 결합된 결과다. 그럼에도 이 표가 본문에서 차지하는 의미는, SE가 기존 backbone의 성능을 약간 올리는 수준을 넘어, 강력한 backbone과 결합됐을 때도 경쟁력 있는 상위권 성능을 유지할 수 있다는 메시지를 주는 데 있다.

Table 8에서 SENet-154의 단일 crop 성능은 다음과 같다.

| 모델 | $224×224$ top-1 | $224×224$ top-5 |
|---|---:|---:|
| `SENet-154` | $18.68$ | $4.47$ |

또한 Table 9는 더 큰 crop/추가 데이터/강한 증강 등을 사용한 이후의 강력한 결과들과 비교를 제공한다. 논문은 이런 요소들이 SE와 상호보완적일 수 있다고 설명한다. 즉, SE는 backbone 수준의 구조 변화이며, augmentation이나 pretraining 같은 시스템적 요소와 **결합할 여지**가 남는다는 메시지다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/cb6f5f9e-0f37-4114-a12b-8eeb0e83c599/image.png" width="40%">
</p>

---

## 7️⃣ Ablation 연구

### 🔹 Reduction Ratio $r$ – 성능과 파라미터의 균형
Table 10은 SE-ResNet-50에서 reduction ratio를 바꿨을 때의 성능과 파라미터 수를 비교한다. 논문은 성능이 꽤 넓은 범위의 $r$에서 안정적이지만, $r$이 너무 작으면 파라미터가 크게 늘고, 성능이 단조롭게 좋아지지는 않는다고 말한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/5f96da85-4f09-42f1-9cd7-6048d163384b/image.png" width="40%">
</p>

논문은 기본값으로 $r=16$이 정확도/복잡도 균형이 좋다고 결론 내린다. 또한 stage별로 최적의 $r$이 다를 수 있으므로, 더 세밀한 튜닝으로 추가 개선이 가능하다는 여지도 언급한다.

Table 10에서 $r$을 작게 만들면(예: $r=2$) bottleneck이 넓어져 게이트 네트워크의 파라미터가 크게 늘어난다. 그런데 성능이 그에 비례해 좋아지지는 않는다. 이는 SE의 효과가 단순히 게이트 네트워크의 용량을 키우는 것만으로 설명되지 않음을 시사한다. SE는 채널 의존성을 모델링하는 도구이지만, 그 도구가 너무 커지면 **오히려 학습/일반화 측면에서 이득이 줄어들 수 있다**.

반대로 $r$을 너무 크게 하면(예: $r=32$) bottleneck이 너무 좁아져 채널 의존성을 충분히 표현하지 못하고 성능이 떨어진다. 그래서 $r$은 단순한 비용 조절 파라미터가 아니라, 채널 관계를 얼마나 압축할지(표현력)와 얼마나 안정적으로 일반화할지(규제 효과)를 함께 결정하는 _하이퍼파라미터_ 로 보는 편이 자연스럽다.

### 🔸 Squeeze/Excitation 구성 요소 – Pooling과 비선형의 선택
Squeeze 연산에 대해 Table 11은 **global max pooling**과 **avg pooling**을 비교한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/25e57d06-3b07-48c8-85b0-2d503e345ee8/image.png" width="40%">
</p>

Avg pooling이 약간 더 낫지만, 큰 차이는 아니다. 논문은 이를 SE가 특정 집계 연산에 과도하게 민감하지 않다는 근거로도 사용한다.

#### 중요한 것은 집계 방식보다 전역 정보의 존재
Max vs. avg의 차이는 있다. **Max는 강한 활성만 남기고, avg는 전체 분포를 반영한다**. 그런데 Table 11에서는 둘 다 baseline 대비 좋은 성능을 보이고, 차이가 크지 않다. 이는 SE에서 핵심이 특정 pooling 연산의 미세한 선택이라기보다, **채널 게이트가 전역 정보를 받느냐**에 있다는 해석을 강화한다. 그래서 이후 Role of SE Blocks에서는 pooling 자체를 제거하는 NoSqueeze가 어떻게 달라지는지를 별도로 확인한다(Table 16).

excitation 비선형에 대해 Table 12는 sigmoid를 ReLU/tanh로 바꾼 결과를 보여준다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/dd769cc8-eb64-4b2b-85a7-340468b0a606/image.png" width="40%">
</p>

ReLU는 특히 나쁘며, baseline보다 성능이 떨어질 수 있다고 논문은 설명한다. 이는 SE가 단지 추가 파라미터 덕분에 좋아지는 것이 아니라, **게이트**의 형태(범위 제한, 포화 특성 등)가 중요한 설계 요소임을 보여준다.

#### Gating 범위 제한의 중요성
Sigmoid는 출력이 $(0,1)$로 제한되기 때문에, 채널별로 **통과/억제**라는 의미가 분명하다. 반면 ReLU는 $0 $아래를 자르고 위로는 제한이 없어서, 게이트라기보다 스케일을 무한히 키울 수 있는 형태가 된다. 이 경우 residual branch나 다음 층의 정규화/비선형과 상호작용하면서 학습이 불안정해질 수 있고, 실제로 성능이 떨어지는 결과가 나온다.

Tanh는 출력이 $(-1,1)$로 제한되지만 음수가 가능해지므로, 채널을 단순히 억제하는 것뿐 아니라 **부호를 뒤집는 효과**까지 생길 수 있다. 이런 동작은 원래 SE가 의도한 재보정(강조/억제)과는 결이 다르며, 결과적으로 _sigmoid_ 가 가장 자연스러운 선택이라는 결론으로 이어진다.

### 🔹 SE Block의 부착 위치
Table 13은 ResNet-50의 각 stage에 SE를 하나씩만 붙였을 때의 효과를 보여준다. 중요한 관찰은, stage 2/3/4 어디에 붙여도 개선이 있고, **전체에 붙이면 가장 좋다는 점**이다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/c1dfa167-3338-484c-a6c7-64a4af12a297/image.png" width="40%">
</p>

특히 stage 4에서 개선이 큰 편인데, 이 stage는 채널 수가 크고(표현의 추상도가 높고) 채널 의존성이 더 중요한 구간일 수 있다는 해석이 가능하다. 하지만 논문은 단순한 이론적 주장보다, stage별 효과가 누적된다는 점을 강조한다.

Stage 2/3/4 어디에 붙여도 개선이 있다는 사실은 SE가 **특정 위치에서만 동작하는 특수 모듈이 아니라는 점**을 보여준다. 동시에 stage 4의 개선 폭이 상대적으로 큰 것은, 깊은 층에서 채널들이 더 고수준 의미를 담당하고, 그 의미들 사이의 선택이 더 중요해질 수 있다는 직관과 맞는다. 논문은 이 지점에서 과도한 이론화를 하진 않지만, ablation 결과 자체가 실무적으로는 삽입 위치 선택의 자유도를 크게 높여준다.

삽입 위치에 대해 Fig. 5와 Table 14는 SE를 residual unit의 어느 위치에 둘지 비교한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/b4b537c3-dff9-4e1f-9e50-0b6490e713d1/image.png" width="70%">
  <img src="https://velog.velcdn.com/images/lumerico284/post/ebc83b32-3a8f-40c0-99df-3e4e00a5269b/image.png" width="40%">
</p>

#### 변형들의 정의를 논문 표현으로 정리
논문은 integration strategy ablation에서 세 가지 변형을 구분한다.

- **SE-PRE**: SE block을 residual unit 바깥(앞)으로 옮겨, residual unit 입력에 가까운 지점에서 재보정을 수행한다.
- **SE-POST**: SE block을 residual branch와 identity branch를 더한 이후, 그리고 ReLU 이후에 두는 변형이다.
- **SE-Identity**: SE block을 residual branch가 아니라 identity connection 쪽에 두어, residual unit과 병렬로 shortcut 경로에서도 게이팅이 일어나도록 만든 변형이다.

이 변형들은 결국 SE를 어디에 붙여야 가장 자연스럽고 안정적으로 성능이 나오는지를 확인하기 위한 실험이며, Fig. 5는 이 위치 차이를 구조적으로 보여주는 역할을 한다.

논문 결론은 aggregation(잔차 더하기) 이전에 적용되면 위치가 꽤 robust하다는 것이다. 반대로 SE-POST처럼 add 이후(ReLU 이후)에 두면 성능이 나빠진다. 즉, SE는 residual branch의 출력을 조절해 add에 들어가기 전에 균형을 맞추는 방식으로 동작하는 것이 더 자연스럽다는 해석이 가능하다.

#### Residual 관점에서의 Table 14
Residual unit은 `(residual branch) + (shortcut)`이라는 합으로 표현을 만들고, 그 뒤에 비선형을 거친다. 이 구조에서 SE가 residual branch에만 작동한다면, SE는 사실상 두 경로가 합쳐지기 전에 **residual 경로의 기여도를 조절하는 장치**가 된다. 그래서 aggregation 이전(SE, SE-PRE)의 결과가 안정적으로 좋고, add 이후(SE-POST)로 넘어가면 이미 섞인 표현을 다시 게이팅하는 형태가 되어 오히려 이점이 줄어들 수 있다는 해석이 가능하다.

또한 Table 15는 SE를 residual unit 내부의 3×3 conv 바로 뒤에 넣는 변형(SE 3×3)을 비교한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/32711a13-d2a5-458b-9c0d-90558e08a1a2/image.png" width="40%">
</p>

성능은 비슷하지만 파라미터는 줄어든다. 논문은 이를 근거로, 특정 architecture에 맞춰 SE 사용을 더 효율적으로 튜닝할 여지가 있음을 언급한다.

#### SE 부착 위치에 따라 달라지는 비용
SE를 **3×3 conv 뒤**에 두는 변형은, 채널 폭이 상대적으로 작은 지점에서 게이팅을 수행하게 만들 수 있다. 그 결과 파라미터가 줄어드는 방향으로 작동한다. 성능이 크게 떨어지지 않으면서 비용이 줄어든다는 점은, SE가 단일 고정 레시피라기보다 architecture에 맞춰 비용-성능 균형을 조절할 수 있는 모듈임을 보여준다.

---

## 8️⃣ SE Blocks의 역할

### 🔹 Squeeze의 역할: 전역 정보가 정말 필요한가
논문은 squeeze의 중요성을 보기 위해 _NoSqueeze_ 변형을 만든다. 이는 global average pooling을 없애고, excitation의 두 FC를 1×1 conv로 대체해 공간 차원을 유지하는 방식이다. 결과는 Table 16이다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/f35a49b8-c94f-42a5-a1d8-6833d6178381/image.png" width="40%">
</p>

NoSqueeze도 baseline보다 좋아지지만, SE가 더 좋고 GFLOPs도 훨씬 낮다. 논문은 이를 통해 **전역 정보(global embedding)를 compact하게 사용하는 것**이 중요하다는 결론을 내린다. 즉, SE의 핵심은 단지 채널 remapping을 하는 것이 아니라, 전역 문맥을 채널 게이트에 넣어주는 squeeze가 중요한 역할을 한다는 것이다.

#### Table 16의 해석
NoSqueeze는 전역 평균 풀링을 제거하고 공간 차원을 유지한 채로 게이팅을 수행하려는 시도다. 결과를 보면 NoSqueeze도 baseline보다 좋아지지만, SE만큼 좋아지지 않고 GFLOPs는 오히려 크게 늘어난다. 이는 두 가지 메시지를 동시에 준다.

1. 채널 게이팅에서 전역 정보는 실제로 유효하다. 단순한 채널 remapping만으로도 개선이 일부 나타나기 때문이다.  
2. 하지만 전역 정보를 compact하게 압축해 사용하는 방식이 효율과 성능 모두에서 중요하다. SE는 squeeze로 $H\times W$를 1×1로 줄여 게이트 네트워크에 넣기 때문에, 더 싸게 더 좋은 결과를 낸다.

즉, SE가 제안하는 설계는 단순히 pooling을 붙여서가 아니라, 전역 정보를 채널 게이팅에 주입하는 경로를 **최소 비용으로** 만드는 형태라고 볼 수 있다.

### 🔸 Excitation의 동작: 깊이에 따라 더 Class-Specific해지는 경향
Section 7.2에서는 excitation 값의 분포를 시각화해 SE의 동작을 경험적으로 분석한다. 논문은 4개 클래스(goldfish, pug, plane, cliff)를 샘플링하고, 각 클래스에서 50개 샘플을 뽑아 특정 SE 블록들에서 채널 excitation의 분포를 비교한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/ac927858-05da-4ae3-ad26-2406c340d4cc/image.png" width="70%">
</p>

논문이 정리하는 관찰은 크게 **세 가지**다.

1. 얕은 층에서는 클래스별 분포가 비슷하다(초기 층 특징은 class-agnostic).  
2. 깊은 층으로 갈수록 클래스별 선호 채널이 달라지고 분포가 더 class-specific해진다.  
3. 마지막 stage의 일부 블록에서는 excitation이 1에 가까운 값으로 포화되어, SE가 identity에 가까워지는 경향이 있다.

이 세 번째 관찰은 Section 4의 마지막 stage SE를 빼도 성능 손실이 작다는 실험적 관찰과도 일관된다. 즉, 마지막 stage에서는 이미 표현이 충분히 정리되어 있고, 채널 재보정이 덜 필요해질 수 있다는 경험적 해석이 가능하다.

#### Fig. 6/7을 논문 흐름으로 다시 연결하면
초기 층에서 **class-agnostic한 excitation 분포**가 나온다는 것은, 초기에 학습되는 특징(엣지, 기본 텍스처 등)이 클래스마다 크게 다르지 않다는 일반적인 표현 학습 직관과 맞닿아 있다. 반대로 깊은 층에서 class-specific해진다는 것은, 고수준 의미 특징의 조합이 클래스마다 달라지고, 그 조합을 만들 때 어떤 채널을 얼마나 통과시킬지가 중요해진다는 해석으로 이어진다.

또한 마지막 stage에서 excitation이 1 근처로 포화되는 경향은, _(a)_ 그 시점의 표현이 이미 분류에 충분히 정리되어 있어 추가 억제가 필요하지 않거나, _(b)_ 학습 과정에서 게이트가 점차 열리며 identity에 가까운 형태로 수렴할 수도 있음을 시사한다. 논문은 이를 근거로 마지막 stage SE 제거 실험(비용 절감)을 해석하고, practical하게는 stage별로 SE 사용을 조절할 여지가 있음을 남긴다.

또한 Fig. 7에서는 같은 클래스 내에서도 인스턴스별 excitation 평균/표준편차를 보여주며, SE가 클래스뿐 아니라 인스턴스별로도 동적으로 반응한다는 점을 강조한다.

#### 같은 클래스라도 채널 선택은 달라질 수 있음
같은 클래스라고 해도 입력 _이미지의 구도, 배경, 조명, 부분 가림 정도_ 에 따라 **유용한 특징 채널은 달라질 수 있다**. Fig. 7은 바로 이 점을 excitation 분포의 평균과 분산으로 보여준다. 분산이 크다는 것은, SE가 클래스 고정의 단일 템플릿처럼 동작하는 것이 아니라, 같은 클래스 안에서도 샘플마다 채널 게이트를 다르게 열고 닫을 수 있음을 의미한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/d5b25fb8-bba6-4297-8863-9dd6dc937f92/image.png" width="70%">
</p>

이 관찰은 SE가 단지 정적 채널 스케일 파라미터를 학습한 것이 아니라, 입력 조건부로 채널 중요도를 계산한다는 논문 전체 메시지를 뒷받침한다. 또한 이런 동적 성질은 모델이 hard case(배경이 복잡하거나 객체가 작게 나온 사례)에서 필요한 특징을 더 선택적으로 통과시키도록 학습될 수 있다는 직관과도 연결된다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/b26ee6a8-a693-4a72-ad9b-6e132e217a8a/image.png" width="40%">
</p>

---

## 💡 해당 논문의 시사점과 한계
SENet의 가장 큰 의의는 채널 축을 단순한 필터들의 나열로 두지 않고, 입력에 따라 **동적으로 채널 중요도를 조절**하는 구조를 블록 단위로 표준화했다는 점이다. SE block은 매우 단순한 연산(GAP + 작은 FC + sigmoid)으로 구현되지만, Table 2에서 보이듯 다양한 backbone(ResNet/ResNeXt/Inception/VGG/BN-Inception)과 다양한 데이터셋(ImageNet/CIFAR/Places365/COCO)에서 일관된 성능 개선을 제공한다

또한 ablation(Table 10~16)은 SE의 효과가 단지 파라미터 증가 때문이 아니라, **squeeze(전역 정보)와 excitation(게이트 형태) 설계가 함께 맞물려야 한다는 점**을 보여준다. 예컨대 sigmoid를 다른 비선형으로 바꾸면 성능이 크게 나빠지고(Table 12), global pooling을 제거하면 효율이 떨어진다(Table 16). 즉, 채널 attention이라는 넓은 아이디어를 성능으로 연결하기 위해 어떤 구성 요소가 본질적인지에 대한 실험적 근거를 제공한다는 점도 중요하다.

#### 왜 이 논문이 오래 살아남았는가: 단순하지만 조립 가능한 설계
SE의 구조는 수식으로 보면 매우 단순하다. 그런데 그 단순함이 오히려 강점이 된다. 어떤 backbone을 쓰든 $\mathbf{U}$만 정의할 수 있으면 동일한 방식으로 채널 게이트를 만들고 적용할 수 있기 때문이다. 연구 관점에서는 새로운 블록을 제안할 때 중요한 질문이 두 가지인데, (1) 다양한 구조/과제에서 일반적으로 통하느냐, (2) 비용 대비 이득이 분명하냐이다. SENet은 이 두 질문에 대해 Table 2~7(여러 backbone/여러 데이터셋)과 Model Complexity 분석으로 비교적 설득력 있는 답을 구성한다.

또한 논문은 단순히 성능이 올랐다는 주장으로 끝내지 않고, ablation을 통해 어떤 구성 요소가 본질인지(전역 pooling, sigmoid 게이트, 삽입 위치 등)를 분해해 보여준다. 이런 형태의 논증은 이후 다른 attention/재보정 모듈을 설계할 때도 기준점으로 작동하기 쉽다.

#### 한계와 실무적 고려
SE는 모듈 하나만 붙인다고 항상 최고 성능을 보장하는 **만능 레시피는 아니다**. 논문에서도 reduction ratio, 삽입 위치, stage별 적용 등 다양한 선택지가 존재하며, 최적 설정은 base architecture와 데이터셋에 따라 달라질 수 있음을 인정한다. 또한 SE의 성능 이득이 가장 크게 나타나는 구간과, 마지막 stage처럼 saturation이 나타나는 구간이 공존한다는 분석(Fig. 6/7)은 어디에 얼마나 붙일지를 더 정교하게 설계할 여지가 있음을 의미한다.

#### 한계의 구체화
SE는 채널 축에 대한 재보정을 제공하지만, 공간 축 $(H,W)$에 대한 선택을 직접적으로 제공하진 않는다. 즉, 어떤 채널을 통과시킬지에 대한 신호는 강하지만, 그 **채널이 공간적으로 어디에서 활성화되는지를 별도로 조절하는 구조는 아니다**. 따라서 SE만으로 모든 형태의 attention 문제가 해결된다고 보기는 어렵고, 어떤 과제에서는 다른 형태의 attention이나 구조 변화와의 결합이 필요할 수 있다.

또한 비용 측면에서도 SE는 항상 무시할 수 있는 수준으로만 증가하는 것은 아니다. 특히 채널이 큰 stage에서 파라미터 증가가 두드러질 수 있고, 이 때문에 논문도 마지막 stage 제거 같은 비용 절감 트릭을 언급한다. 실무적으로는 이런 비용-성능 절충을 고려해, stage별 적용이나 $r$ 값을 조절하는 튜닝이 필요할 수 있다.

그럼에도 SENet은 이후 수많은 backbone 설계에서 채널 주의 메커니즘의 표준 구성 요소로 자리잡았다. 무엇보다 attachment로서의 모듈이라는 형태가 실무/연구 모두에서 강력하다. backbone을 완전히 바꾸지 않고도, 기존 모델을 구조적으로 강화할 수 있기 때문이다.

---

## 👨🏻‍💻 SENet 구현하기
이 파트에서의 [`lucid`](https://github.com/ChanLumerico/lucid/tree/main) 라이브러리를 이용한 SENet 구현 설명은 [`nn.SEModule`](https://github.com/ChanLumerico/lucid/blob/main/lucid/nn/fused.py#L130)을 먼저 소개하고, 그 다음에 [`senet.py`](https://github.com/ChanLumerico/lucid/blob/main/lucid/models/imgclf/senet.py)가 SE를 다양한 backbone(ResNet/ResNeXt) 팩토리로 제공하는 방식을 설명한다. SENet은 단일 모델이라기보다 기존 블록에 SE를 붙이는 방식이므로, Lucid 코드도 backbone 빌더(ResNet)를 재사용하면서 SE만 옵션으로 끼워 넣는 형태로 구현되어 있다.

### 0️⃣ 전체 대응 관계 요약
- 논문 squeeze(GAP) + excitation(FC→ReLU→FC→sigmoid) + scale(채널 곱) → `lucid/nn/fused.py`의 `SEModule`
- SE-ResNet/SE-ResNeXt 계열 모델 엔트리 → `lucid/models/imgclf/senet.py`의 `SENet` 및 `se_*` 팩토리 함수
- ResNet bottleneck에 SE를 삽입하는 구현 → `lucid/models/imgclf/resnet.py`의 `_Bottleneck`이 `se=True`일 때 `nn.SEModule`을 호출
- ResNet basic block(18/34)에는 기본 SE 옵션이 없으므로 → `senet.py`가 `_SEResNetModule`을 별도로 제공

### 1️⃣ `nn.SEModule`: SENet의 핵심 Attachment
Lucid의 `SEModule`은 논문 SE block의 가장 표준적인 형태를 거의 그대로 구현한다. 코드 흐름은 `AdaptiveAvgPool2d((1,1))`로 squeeze, `Linear` 두 번과 `ReLU`, `Sigmoid`로 excitation, broadcast 곱으로 scale이다.

```python
class SEModule(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("Only supports 4D-tensors.")

        spatial_axes = (-1, -2)
        y = self.avgpool(x).squeeze(axis=spatial_axes)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))

        y = y.unsqueeze(axis=spatial_axes)
        out = x * y
        return out
```

논문 Eq.(2)~(4)와의 대응을 문장으로 정리하면 다음과 같다.

- `avgpool(x)`는 $(H,W)$를 1×1로 줄이는 global average pooling이며, `squeeze`가 채널 descriptor $\mathbf{z}$를 만든다.
- `fc1 → ReLU → fc2 → Sigmoid`는 excitation의 게이트 네트워크로, 채널 중요도 $\mathbf{s}$를 만든다.
- 마지막 `x * y`는 채널별 스케일링(브로드캐스팅)을 수행해 $\widetilde{\mathbf{U}}$를 만든다.

#### 입력 형태와 브로드캐스팅 관점에서 본 `SEModule.forward`
`SEModule`은 입력이 4D 텐서인지(`x.ndim != 4`)를 먼저 확인한다. 이는 SE가 기본적으로 $(N, C, H, W)$ 형태의 feature map을 대상으로 설계되었기 때문이다. 그 다음 `avgpool(x)`는 공간 축을 $1×1$로 줄여 $(N, C, 1, 1)$을 만들고, `squeeze(axis=spatial_axes)`로 $(N, C)$ 형태의 채널 descriptor로 바꾼다.

이후 게이트 네트워크를 통과한 `y`는 다시 `unsqueeze(axis=spatial_axes)`로 $(N, C, 1, 1)$ 형태가 되고, `out = x * y`에서 브로드캐스팅이 일어나 각 채널이 동일한 스케일로 곱해진다. 이 흐름은 논문이 말하는 channel-wise multiplication을 코드 수준에서 그대로 구현한 것이다.

또한 `reduction`이 곧 논문에서의 reduction ratio $r$에 해당한다. `in_channels // reduction`이 중간 차원 $C/r$을 만든다.

### 2️⃣ `SENet`: SE Block 부착 Wrapper
`senet.py`의 `SENet` 클래스는 ResNet 빌더를 그대로 사용하면서, 블록 생성 시 `se=True`와 `se_args`를 강제로 전달하는 래퍼다. 즉, SENet이 **부착 모듈**이라는 관점을 코드 구조가 그대로 반영한다.

```python
class SENet(ResNet):
    def __init__(
        self,
        block: nn.Module,
        layers: list[int],
        num_classes: int = 1000,
        reduction: int = 16,
        block_args: dict = {},
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            block_args={"se": True, "se_args": dict(reduction=reduction), **block_args},
        )
```

여기서 `block_args`가 핵심이다. `ResNet`은 `_make_layer()`에서 블록을 만들 때 `**block_args`를 넘기는데, `SENet`은 여기에 `se=True`를 기본으로 넣어 항상 SE가 켜지게 한다. 또한 `reduction`을 `se_args`로 전달해 `SEModule(reduction=...)`에 연결한다.

#### Attachment 관점이 코드 구조에 반영되는 지점
이 구조는 논문에서 말하는 통합(integration)을 가장 간단한 형태로 실현한다. 기존 ResNet 빌더는 그대로 두고, 블록 생성 인자만 주입해 SE를 켜기 때문이다. 즉, Lucid에서 SE는 별도의 거대한 모델 정의가 아니라, backbone 생성 과정에서 옵션으로 끼워 넣을 수 있는 기능으로 취급된다.

또한 `SENet`이 `block_args`를 통해 추가 인자를 전달한다는 것은, 같은 래퍼 구조를 유지한 채로 ResNeXt의 cardinality/base_width 같은 인자도 함께 주입할 수 있음을 의미한다. 실제로 아래 `se_resnext_*` 팩토리 함수들이 이 방식을 그대로 사용한다.

즉, 논문에서 말하는 SE block을 기존 모듈에 통합하는 작업이 Lucid에서는 블록 생성 인자에 se 옵션을 주입하는 방식으로 구현된다.

### 3️⃣ `_SEResNetModule`: ResNet-18/34용 Basic Block에 SE 붙이기
Lucid의 ResNet 기본 블록(`_BasicBlock`)은 SE 옵션을 갖지 않는다. 그래서 `senet.py`는 ResNet-18/34용으로 별도의 블록 `_SEResNetModule`을 제공한다. 이 블록은 두 개의 3×3 conv(ResNet basic block) 뒤에 `nn.SEModule`을 적용하고, shortcut을 더한 다음 ReLU를 한다.

```python
class _SEResNetModule(nn.Module):
    expansion: ClassVar[int] = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        reduction: int = 16,
        **kwargs,
    ) -> None:
        super().__init__()

        self.conv1 = nn.ConvBNReLU2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            conv_bias=False,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.se_module = nn.SEModule(out_channels, reduction)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.conv2(out)

        out = self.se_module(out)
        if self.downsample is not None:
            out += self.downsample(x)

        out = self.relu(out)
        return out
```

여기서 논문 관점에서 중요한 구현 포인트는 다음이다.

- **SE 적용 위치**: residual branch의 두 conv를 지난 뒤(out이 residual branch 출력일 때) `se_module(out)`을 수행한다. 이는 branch aggregation 이전에 재보정을 적용하는 형태로, 논문 Fig. 5의 SE/SE-PRE/SE-POST 비교에서 성능이 좋은 계열과 같은 방향이다.
- **Shortcut 처리**: downsample이 있으면 `x`를 downsample해서 더한다. downsample이 없으면 identity shortcut을 `out`에 더한다.
- **`expansion=1`**: basic block이므로 출력 채널이 out_channels로 유지된다.

### 4️⃣ 모델 팩토리: `se_resnet_*`
`se_resnet_18/34`는 `_SEResNetModule`을 블록으로 써서 SENet을 만든다. 반면 `se_resnet_50/101/152`는 ResNet bottleneck 블록(`_Bottleneck`)을 사용하고, `SENet`이 `se=True`를 강제로 주입하기 때문에 `_Bottleneck` 내부에서 SE가 활성화된다.

```python
@register_model
def se_resnet_18(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [2, 2, 2, 2]
    return SENet(_SEResNetModule, layers, num_classes, **kwargs)


@register_model
def se_resnet_34(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 4, 6, 3]
    return SENet(_SEResNetModule, layers, num_classes, **kwargs)


@register_model
def se_resnet_50(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 4, 6, 3]
    return SENet(_Bottleneck, layers, num_classes, **kwargs)
```

즉, Lucid에서는 ResNet-18/34는 별도 SE-basic-block을 쓰고, ResNet-50+는 기존 bottleneck에 SE를 켠다는 방식으로 논문 SE-ResNet 계열을 구현한다.

#### 팩토리 함수를 직접 호출하는 가장 단순한 사용 예
Lucid 구현을 실험적으로 확인하고 싶다면, 등록 시스템을 거치지 않고도 팩토리 함수를 직접 호출해 모델 인스턴스를 만들 수 있다.

```python
from lucid.models import se_resnet_50

model = se_resnet_50(num_classes=1000)
```

### 5️⃣ 모델 팩토리: `se_resnext_*`
SE-ResNeXt는 ResNeXt의 grouped conv(논문 ResNeXt의 cardinality/base_width)와 SENet의 SE를 동시에 사용하는 형태다. Lucid에서는 `_Bottleneck`이 이미 `cardinality/base_width` 인자를 받도록 일반화되어 있고, 동시에 `se=True`이면 `nn.SEModule`을 적용할 수 있으므로, `block_args`에 ResNeXt 인자만 추가로 넣어주면 된다.

```python
@register_model
def se_resnext_50_32x4d(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 4, 6, 3]
    block_args = {"cardinality": 32, "base_width": 4}
    return SENet(_Bottleneck, layers, num_classes, block_args=block_args, **kwargs)


@register_model
def se_resnext_101_64x4d(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 4, 23, 3]
    block_args = {"cardinality": 64, "base_width": 4}
    return SENet(_Bottleneck, layers, num_classes, block_args=block_args, **kwargs)
```

이렇게 보면 Lucid에서 SENet은 진짜로 attachment다. ResNeXt의 핵심(`groups=cardinality`)은 그대로 두고, SE를 켜는 인자만 추가해 동일한 `_Bottleneck` 블록이 ResNeXt + SE를 동시에 구현하게 만든다. 이는 논문이 강조하는 다양한 architecture에 SE를 쉽게 통합한다는 메시지를 코드 구조로 그대로 옮긴 형태라고 볼 수 있다.

---

## ✅ 정리
**SENet**은 채널을 고정된 표현의 축으로만 다루지 않고, 입력에 따라 채널 중요도를 동적으로 조절하는 SE block을 제안함으로써 CNN 표현력을 강화한다. global average pooling으로 전역 문맥을 채널 descriptor로 압축(**squeeze**)하고, 작은 게이트 네트워크로 채널 중요도 벡터를 만든 뒤(**excitation**), 이를 feature map에 채널별로 곱해 재보정(**scale**)하는 구조는 매우 단순하지만, ImageNet에서 ResNet/ResNeXt/Inception 계열을 포함한 다양한 backbone에서 일관된 성능 향상을 제공한다(Table 2).

정리하면 이 논문이 설득력 있게 전달하는 메시지는 **크게 두 가지**다. 첫째, 채널 의존성을 명시적으로 모델링하는 것은 대부분의 CNN backbone에서 재사용 가능한 일반적 개선 방향이다. 둘째, 그 개선은 복잡한 attention 설계가 아니라 전역 요약(squeeze)과 범위가 제한된 게이트(sigmoid excitation), 그리고 단순한 곱(scale)이라는 매우 작은 구성만으로도 달성될 수 있다. 그래서 SE는 이후 다양한 모델 설계에서 조립 가능한 구성 요소로 자리잡기 쉬웠다.

또한 ablation과 분석은 SE를 쓸 때 무엇을 고정해야 하는지에 대한 실무적 힌트도 준다. 예를 들어 전역 정보가 사라지면(NoSqueeze) 효율이 급격히 나빠지고(Table 16), 게이트 비선형이 바뀌면 성능이 크게 흔들리며(Table 12), 삽입 위치는 add 이전이 더 자연스럽다(Table 14). 즉, SE는 단순한 모듈이지만, 그 단순함이 성능으로 이어지려면 몇 가지 설계 원칙이 함께 지켜져야 한다.

- **핵심 아이디어**: squeeze(GAP) → excitation(FC-ReLU-FC-sigmoid) → scale(채널 곱)
- **주요 하이퍼파라미터**: reduction ratio `r`(Lucid에서는 `reduction`)
- **실험 결론**: 다양한 backbone/데이터셋에서 일관된 개선(Table 2~7)

#### 📄 출처
Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-Excitation Networks." *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, arXiv:1709.01507.
