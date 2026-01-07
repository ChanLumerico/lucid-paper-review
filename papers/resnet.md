# [ResNet] Deep Residual Learning for Image Recognition

ResNet 논문은 네트워크를 더 깊게 쌓으면 더 좋은 표현을 학습할 수 있다는 직관이 실제 학습에서는 쉽게 깨진다는 문제에서 출발한다. 깊은 plain network(단순 적층 구조)는 어느 시점부터 **훈련 오차가 오히려 올라가며** 성능이 악화되는 현상(degradation problem)을 보이는데, 저자들은 이 현상이 단순한 overfitting이 아니라 **최적화 자체의 어려움**이라고 주장한다.

이 논문이 제안하는 해법은 **residual learning**이다. 핵심 아이디어는 네트워크가 어떤 목표 함수 $H(x)$를 직접 학습하기보다, 입력 $x$에 대한 **변화량(residual; 잔차)** $F(x)$를 학습하도록 만들고 출력은 $y = x + F(x)$로 구성하는 것이다. 이 구조는 identity mapping에 가까운 최적해가 존재할 때 특히 학습이 쉬워지며, 실제로 매우 깊은 네트워크에서도 안정적으로 학습이 진행됨을 ImageNet과 CIFAR-10에서 실험적으로 보여준다. 또한 ResNet-101을 기반으로 Faster R-CNN을 개선해, COCO/PASCAL/ImageNet detection·localization에서 큰 성능 향상을 보고한다.

이 글에서는 논문이 가장 직접적으로 다루는 분류 모델(ResNet-18/34/50/101/152)과, residual block의 수식, 그리고 실험 섹션에서 제시하는 분류·검출·localization 결과를 원문 흐름대로 따라가며 정리한다.

ResNet이 딥러닝 업계에 끼친 영향이 큰 이유는, 성능 숫자 자체뿐 아니라 설계 패턴이 남겼기 때문이다. **skip connection**이라는 구조적 아이디어는 이후 분류 모델뿐 아니라 detection, segmentation, generation 등 다양한 영역에서 사실상 기본 레고 블록처럼 자리잡았고, backbone을 깊게 만들면서도 학습이 무너지지 않게 하는 실무 표준이 되었다. 이 논문은 그런 변화의 출발점에 해당한다는 점에서, 원문 전개를 최대한 촘촘히 따라가며 읽을 가치가 있다.

---

## 1️⃣ 배경 상황

### 🔹 깊어질수록 성능이 떨어지는 이유
딥러닝에서 **깊이(depth)** 는 표현력을 키우는 가장 직접적인 축이다. 레이어를 더 쌓으면 더 복잡한 함수를 근사할 수 있고, 이상적으로는 더 깊은 모델이 더 얕은 모델보다 최소한 **훈련 오차가 나빠지지 않아야** 한다. 왜냐하면 깊은 모델이 얕은 모델을 포함하는 방식으로 구성될 수 있기 때문이다. 예컨대 추가 레이어들이 identity mapping(입력을 그대로 통과시키는 변환)처럼 동작한다면, 깊은 모델은 얕은 모델과 같은 함수도 표현할 수 있다.

하지만 저자들은 경험적으로 이 **직관이 자주 깨진다**고 말한다. 깊은 plain network를 만들면, 어느 시점부터 훈련 오차가 **더 낮아지지 않고 오히려 올라가며**, 그 결과 validation 성능까지 악화된다. 이 현상을 논문은 _degradation problem_ 이라고 부르며, 핵심은 더 깊은 모델이 더 큰 해 공간을 가지므로 더 좋은 해를 찾을 수 있어야 한다는 관점에서 보면 매우 이상하다는 점이다.

이 주장은 단순한 말장난이 아니라, 네트워크를 함수 공간의 관점에서 보는 직관을 깔고 있다. 얕은 모델의 최적해를 $H_{shallow}$라 하고, 깊은 모델은 그보다 더 많은 파라미터와 더 큰 함수 공간을 가진다고 생각하면, 깊은 모델은 최소한 $H_{shallow}$를 표현할 수 있어야 한다. 그런데 실제로는 SGD가 그 해까지 도달하지 못하고, 깊은 plain 모델이 더 높은 training error에 머문다. ResNet은 이 지점에서 깊이를 늘리는 문제를 표현력 문제가 아니라 **solver가 다루기 쉬운 파라미터화** 문제로 바꿔 보려는 시도라고 이해할 수 있다.

#### Identity가 어려운 이유
논문이 제시하는 핵심 직관은 다음과 같다. 추가된 여러 비선형 층들이 identity를 근사할 수 있으면 깊은 모델은 얕은 모델보다 나빠질 이유가 없다. 그런데 실제로는 여러 층의 합성으로 identity를 만드는 것이, 사람이 생각하는 것처럼 **손쉽지 않다는 것**이다. 특히 ReLU 같은 비선형은 입력 분포의 일부를 $0$으로 눌러버릴 수 있고, BN이 있다고 해도 조합된 층들이 identity 근처로 모이는 것이 항상 자연스럽지는 않다. 결국 깊은 plain 모델에서는 optimizer가 identity를 구현하는 파라미터를 찾는 과정 자체가 어려워져, 훈련 오차가 얕은 모델보다 높게 유지될 수 있다.

### 🔸 Degradation의 근본적인 원인
Degradation의 원인이 overfitting이라면 훈련 오차는 계속 줄어들고, 일반화 오차만 나빠져야 한다. 그런데 실제로는 **훈련 오차 자체가** 깊은 모델에서 더 나쁘게 나온다. 즉, 문제는 데이터에 과적합하는 것이 아니라, SGD 기반의 학습이 깊은 plain network에서 좋은 해를 찾지 못하는 최적화 문제로 해석된다.

저자들은 Batch Normalization(BN)을 사용하면 vanishing gradient가 완화된다는 점을 강조하며, 그럼에도 degradation이 발생한다는 관찰을 근거로 든다. 따라서 단순히 gradient가 사라지는 현상만으로는 설명하기 어렵고, 깊은 비선형 층들이 **identity mapping을 근사하는 것 자체가 어렵다**는 가설을 제시한다. 이 지점에서 residual learning은 identity에 대한 작은 수정만 학습하자는 형태로 등장한다.

여기서 논문의 강조점은 BN이 만능이 아니라는 것이 아니라, **BN으로도 남는 최적화 난점**이 있다는 사실이다. 즉 forward 신호가 완전히 사라지거나, backward gradient가 $0$으로 붕괴하는 문제는 줄었는데도, 깊은 plain 네트워크는 여전히 좋은 해로 수렴하기 어렵다. 논문은 이를 보다 구조적인 관점에서 해석하며, 여러 비선형 층이 쌓인 모듈이 identity를 잘 근사하지 못하는 것이 핵심 병목일 수 있다고 본다. 따라서 residual block은 identity를 기본값으로 깔아두고, 그 위에 학습 가능한 변화량만 얹는 형태로 최적화 경로를 단순화한다.

#### 최적화 관점에서의 재해석
이 논문을 최적화 관점에서 다시 보면, residual connection은 네트워크가 탐색하는 함수 공간을 바꾸는 것이 아니라, 같은 함수 공간을 더 쉽게 탐색하도록 **파라미터화를 바꾸는 것**으로 읽힌다. plain 네트워크가 어떤 $H(x)$를 직접 만들도록 강제한다면, residual 네트워크는 $H(x)=x+F(x)$ 꼴로 표현하도록 강제한다. 이때 $F(x)=0$은 곧바로 identity이고, 초기화가 $0$ 근처라면 초기 상태가 identity 주변에 놓이게 된다. 학습은 그 주변에서 필요한 방향으로만 움직이면 되므로, 초기 단계의 수렴이 쉬워질 가능성이 높다.

---

## 2️⃣ 관련 연구

### 🔹 더 깊은 네트워크를 가능하게 만든 장치들
Related Work에서는 깊은 네트워크가 성능을 끌어올려 왔던 흐름을 정리한다. 대표적으로 **VGG 계열**은 작은 커널(3×3)을 반복해 깊이를 키우는 방식으로 좋은 성능을 달성했고, **GoogLeNet 계열**은 Inception 모듈을 통해 계산량을 통제하면서 깊은 구조를 만들었다. 이들 모델은 깊이가 효과적이라는 경험적 근거를 제공했지만, depth가 무한히 늘면 항상 좋아지는지는 별개의 문제로 남는다.

또한 최적화 측면에서 BN은 학습을 안정화시키는 중요한 도구로 자리잡았고, dropout 같은 정규화 기법들도 자주 사용되었다. 그러나 ResNet 논문은 BN이 있음에도 degradation이 발생할 수 있음을 보여주며, 깊은 네트워크 최적화의 난점이 여전히 남아 있음을 강조한다.

또한 논문은 ImageNet 규모에서의 대표적인 학습 관행들(예: SGD + momentum, weight decay, 10-crop testing, fully-convolutional multi-scale testing 등)을 공유된 베이스라인으로 삼는다. 이는 모델 구조의 효과를 보여줄 때, 학습 레시피의 차이가 아닌 구조 차이로 결과를 설명하기 위함이다. ResNet은 새로운 최적화 알고리즘을 제안하는 논문이 아니라, **구조적 파라미터화 변경이 최적화 난점을 어떻게 바꾸는지**를 보여주는 논문이라는 점이 드러난다.

#### VGG/GoogLeNet의 성공을 ResNet의 문제의식으로 다시 보기
**VGG**의 핵심은 작은 커널을 반복해 receptive field를 키우면서도, 매 층이 비교적 단순한 패턴(3×3 conv + 비선형 + pooling 등)으로 구성되도록 만드는 것이다. 여기서 중요한 메시지는 두 가지다.

1. 깊이가 증가하면 표현력이 강해져 분류 성능이 개선된다는 경험적 근거가 이미 축적되어 있었다.
2. 그 깊이는 단지 파라미터가 많다는 의미가 아니라, 여러 단계의 비선형 변환을 통해 점진적으로 feature를 추상화한다는 의미로 받아들여졌다.

**GoogLeNet(Inception)** 쪽은 조금 다른 각도에서 깊이를 확장한다. 단일 경로로 층을 길게 늘리기보다, 한 stage 안에서 여러 커널 크기(또는 pooling 경로)를 병렬로 두고 concat하는 방식으로 표현을 풍부하게 만들고, 1×1 conv로 채널 폭을 조절해 계산량을 통제한다. 즉, 깊이를 늘리면서도 비용을 유지할 수 있는 구조적 아이디어가 존재한다는 것을 보여준다.

**ResNet**은 이 두 흐름을 부정하지 않는다. 오히려 이들의 성과를 전제로, 더 깊게 가려고 하면 마주치는 벽이 무엇인지(훈련 오차가 더 높아지는 degradation)를 문제로 삼는다. 다시 말해 이 섹션에서 강조하는 것은, 깊이 자체가 효과적이라는 사실과, 그 깊이를 실제로 최적화 가능한 형태로 만드는 문제는 별개라는 구분이다.

#### BN이 있음에도 Degradation이 나타나는 이유
BN은 입력 분포를 안정화해 학습을 쉽게 만드는 도구로 널리 받아들여져 왔다. 하지만 ResNet 논문은 BN을 쓰는 구성에서도 깊이가 커질 때 훈련 오차가 오르는 사례를 보여주며, BN이 모든 최적화 난점을 해결하지는 못한다고 말한다. BN은 주로 각 층의 _activation scale_ 을 다루지만, 깊이가 증가하며 생기는 문제는 단지 scale의 문제가 아니라, optimizer가 좋은 해(특히 identity 근처의 해)로 도달하기 어려운 **파라미터화 방식 자체**일 수 있다는 주장으로 이어진다.

이 관점은 이후 본문에서 residual learning을 preconditioning에 가까운 방식으로 해석하는 전개와 연결된다. 즉, BN은 학습의 수치적 안정성을 돕지만, residual connection은 탐색해야 하는 함수 공간에서 출발점을 더 좋은 곳(초기엔 거의 identity가 되는 곳)으로 **이동시키는 역할**을 한다는 식이다.

#### 왜 Classification을 넘어 일반 표현 학습으로 이어지는가
Related Work를 읽을 때 중요한 포인트는, 당시의 분류 모델들이 이미 여러 downstream 과제의 backbone으로 쓰이고 있었다는 점이다. 즉 ImageNet 분류 정확도는 그 자체로 중요한 목표이면서 동시에, 표현 학습의 질을 측정하는 간접 지표로도 사용되었다. ResNet이 제시하는 잔차 연결은 이런 backbone 스케일업을 훨씬 안정적으로 만들었고, 이후 detection/segmentation 같은 과제에서 backbone 교체만으로 큰 성능 향상이 가능한 길을 열었다는 점이 논문 후반 실험과도 맞물린다.

### 🔸 Shortcut Connection 계열과 Residual의 차별점
논문은 Highway Network 등 이전의 shortcut connection 계열을 언급하면서도, ResNet이 추구하는 방향은 게이트로 경로를 조절하는 복잡한 구조가 아니라, **파라미터 없이 단순한 더하기(add)** 로 identity 경로를 제공한다는 점에 있다.

#### Highway Network와의 대비
Highway Network의 대표적인 형태는 게이트를 통해 변환 경로와 carry 경로를 섞는다. 표기만 단순화하면 다음과 같은 형태를 생각할 수 있다.

$$y = H(x) \odot T(x) + x \odot C(x)$$

여기서 $T(x)$는 transform gate, $C(x)$는 carry gate에 해당하고, $\odot$는 element-wise 곱이다. 게이트가 $0/1$에 가까운 값을 가지면 정보가 어느 경로로 흐를지 조절할 수 있고, 이 덕분에 매우 깊은 네트워크를 학습시키는 것이 가능하다는 것이 Highway의 메시지다.

ResNet은 이 접근과 달리, identity 경로를 항상 열어 두되(기본은 $x$를 그대로 더함), 게이트 파라미터를 별도로 학습하지 않는다. 즉,

$$y = F(x) + x$$

로 경로 결합을 강제한다. 여기서 중요한 차이는 **계산 비용과 비교 실험의 공정성**이다. 게이트를 추가하면 파라미터와 연산량이 늘고, 학습이 쉬워진 것이 residual 구조 때문인지 단지 모델이 커졌기 때문인지 분리하기가 어렵다. 반면 ResNet의 shortcut은 파라미터가 없으므로, 같은 depth/width 조건에서 shortcut의 효과를 더 **직접적으로 확인할 수 있다**.

이 대비는 논문이 반복해서 강조하는 메시지와 맞닿아 있다. ResNet이 기여한 핵심은, 더 복잡한 조절 메커니즘을 넣는 것이 아니라, 가장 단순한 형태의 identity 경로를 넣었을 때도 degradation이 사라지고 깊이의 이점이 실제 성능으로 이어진다는 점이다.

즉 ResNet의 shortcut은 계산량/파라미터를 거의 늘리지 않으면서도, 깊은 구조에서 최적화를 돕는다는 목표를 가진다. 이러한 단순성 덕분에, plain network와 residual network를 공정하게 비교할 수 있고(같은 depth/width/parameter/FLOPs), 실제로 그 비교에서 residual의 효과가 뚜렷하게 드러난다는 것이 논문 전개의 핵심이다.

이 공정 비교라는 포인트는 이후 실험에서 반복된다. 저자들은 18/34-layer plain과 ResNet을 같은 구성으로 맞춘 뒤 shortcut만 추가해 비교하고, 더 깊은 모델에서도 bottleneck 구조를 통해 계산량을 통제하면서 비교한다. 이때 shortcut은 구조적 편의가 아니라, 실험 논증을 구성하는 핵심 장치가 된다.

#### Residual의 본질
이 논문에서 residual은 게이트나 attention처럼 무엇을 선택적으로 통과시키는 장치라기보다, 입력을 무조건 통과시키는 경로를 하나 더 제공하는 장치다. 따라서 모델은 **항상 최소한 identity 경로**를 가지며, residual branch는 그 위에 추가되는 변화량만 책임진다. 이런 역할 분담이 깊은 모델에서 학습을 단순화한다는 것이 ResNet의 핵심 주장이다.

---

## 3️⃣ 심층 잔차 학습

### 🔹 Degradation Problem과 Residual 학습의 동기
논문은 degradation problem을 CIFAR-10과 ImageNet 등에서 관찰 가능한 일반적 현상으로 제시한다. 특히 깊은 plain network가 얕은 plain network보다 훈련 오차가 더 큰 사례를 제시하며, 이는 모델 용량 부족이나 overfitting이 아니라 solver(최적화)가 identity mapping을 잘 근사하지 못하는 문제일 수 있다고 주장한다.

여기서 저자들이 세우는 중요한 가설은 다음과 같다.

- 깊은 모델의 최적해가 identity mapping에 가까울 수 있다.
- 그렇다면 $H(x)$를 직접 학습하는 대신, $F(x)=H(x)-x$ 형태의 잔차를 학습하게 만들면 더 쉬울 수 있다.

이 가설은 직관적으로도 설득력이 있다. 어떤 입력 표현이 이미 유용하다면, 다음 블록은 완전히 새로운 표현을 만들기보다 **기존 표현을 조금 수정**하는 편이 자연스럽다. residual learning은 이 과정을 구조적으로 강제한다.

또한 논문은 Fig. 7의 분석을 통해 실제로 learned residual response가 작은 경향(즉 $F(x)$가 작은 변화량)을 보인다고 보고하며, identity mapping이 reasonable preconditioning이라는 관찰을 뒷받침한다.

이 해석은 residual을 단순히 gradient가 잘 흐르게 하는 트릭으로만 보지 않고, 함수 근사의 관점에서 재해석한다는 점에서 의미가 있다. 만약 최적 함수 $H(x)$가 입력에 가까운 형태라면, $H(x)$를 처음부터 학습하는 것보다 $F(x)=H(x)-x$를 학습하는 것이 더 쉽다는 주장이다. 이때 쉬움은 표현력이 아니라 **탐색해야 하는 파라미터 공간의 위치**가 달라진다는 뜻에 가깝다. 즉, 초기화가 $0$ 근처일 때 $F(x)$가 $0$이면 블록은 곧바로 identity가 되고, 학습은 identity 주변에서 필요한 변화만 찾아가면 된다.

#### Residual이 주는 구조적 편의
ResNet의 residual block은 네트워크의 각 stage를 일종의 반복 구조로 만든다. plain 네트워크가 깊어질수록 구조적 의미를 해석하기 어려워지는 반면, residual 네트워크는 동일한 형태의 블록이 반복되며 각 블록이 입력을 얼마나 수정하는지만 보면 된다. 이는 분석(예: Fig. 7의 layer response)에서도 직접적인 장점으로 이어진다. 즉, 깊은 모델을 학습시키는 것뿐 아니라, 학습된 모델을 해석하고 디버깅하는 관점에서도 residual 연결은 구조를 단순하게 만든다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/3b9a2e43-e941-4777-bf6a-c1bc25acc2ee/image.png" width="70%">
</p>

### 🔸 수식으로 보는 Residual Block
ResNet의 기본 building block은 Fig. 2로 제시된다. 블록은 입력 $x$에 대해 여러 층을 통과한 출력 $F(x,\{W_i\})$를 만들고, 이를 입력과 더해 다음을 만든다.

$$
y = F(x,\{W_i\}) + x \tag{1}
$$

여기서 $x$와 $y$는 각각 블록의 입력/출력 벡터(혹은 feature map)이고, $F$는 **학습할 잔차(residual mapping)** 이다. Fig. 2의 2-layer 예시라면

$$
F(x,\{W_i\}) = W_2 \sigma(W_1 x)
$$

처럼 쓸 수 있고, 논문은 더하기 이후에 비선형성을 적용하는 형태(즉 $\sigma(y)$)를 사용한다고 밝힌다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/60a22421-c8f1-46ca-afdd-c987c8466d14/image.png" width="35%">
</p>

이 구조에서 가장 중요한 장점은 shortcut이 **추가 파라미터를 도입하지 않는다**는 점이다. 곧, $(1)$의 shortcut은 모델 크기와 연산량을 거의 늘리지 않으면서 최적화 경로를 제공한다. 따라서 plain/residual 비교가 공정해진다.

또 하나의 중요한 관찰은 역전파 관점에서의 경로다. $y=x+F(x)$이면, 미분은

$$
\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}
$$

처럼 identity 항이 포함된다. 이는 어떤 블록에서 $\frac{\partial F}{\partial x}$가 불안정해도, **최소한 identity 성분이 gradient 경로를 제공한다는 의미**로 해석할 수 있다. 물론 이는 완전한 증명이라기보다 직관에 가깝지만, 논문이 강조하는 것처럼 shortcut이 최적화 난점을 줄이는 방향으로 작동한다는 설명과 잘 맞는다.

#### 여러 블록을 연쇄했을 때의 누적 형태
$(1)$의 관점을 조금 더 확장해 보면, ResNet은 네트워크 전체를 **잔차의 누적**으로 보는 해석을 제공한다. 블록 인덱스를 $l$로 두고 다음과 같이 쓸 수 있다고 하자.

$$
x_{l+1} = x_l + F_l(x_l)
$$

그러면 $l$에서 $L$까지의 여러 블록을 합성한 출력은 다음처럼 전개된다.

$$
x_L = x_l + \sum_{i=l}^{L-1} F_i(x_i)
$$

여기서 중요한 점은, 어떤 형태로든 $x_l$이 $x_L$에 직접 더해지는 경로가 존재한다는 것이다. 즉, 각 블록이 만드는 변화량들의 합 위에 원 신호가 계속 유지되는 구조다. 이 해석은 3.1절에서 말한 **preconditioning 직관**(초기에는 $F$가 0에 가까우면 곧바로 identity가 된다)과도 맞닿아 있다.

또한 역전파 관점에서도, 최종 손실 $\mathcal{L}$에 대해

$$
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_{l+1}} \left(I + \frac{\partial F_l}{\partial x_l}\right)
$$

처럼 각 블록마다 identity 성분이 끼어든다. 이는 깊이가 커질수록 곱셈 체인만으로 gradient가 급격히 약해지거나 폭발하는 경향을 완화한다는 직관으로 연결된다.

이 논문은 이 수식이 곧바로 엄밀한 안정성 보장을 제공한다고 주장하지는 않는다. 다만 shortcut이 없는 plain net에서는 이 identity 성분이 사라지므로, 동일한 depth에서도 학습 곡선이 크게 달라진다는 것을 이후 실험으로 보여준다.

#### Conv Feature Map에서의 덧셈
논문은 $(1)$, $(2)$를 fully-connected 벡터 표기처럼 쓰지만, 실제 구현은 convolution feature map에서도 동일하다. $x$와 $F(x)$는 같은 공간 크기와 같은 채널 수를 가진 feature map이어야 하고, 덧셈은 채널별 element-wise addition으로 수행된다. 이 제약이 곧 projection shortcut – $(2)$이 필요한 이유이며, downsampling 시점(해상도 변화, 채널 변화)에서 shortcut 경로에 1×1 conv를 두는 것이 가장 자연스러운 구현이 된다.

### 🔹 차원 불일치와 Projection Shortcut
$(1)$을 성립시키려면 $x$와 $F(x)$의 차원이 같아야 한다. 하지만 네트워크가 깊어지면서 채널 수가 늘거나 해상도가 줄어드는 시점에서는 **이 조건이 깨진다**. 논문은 이 경우 shortcut에 선형 변환 $W_s$를 적용해 차원을 맞출 수 있다고 말한다.

$$
y = F(x,\{W_i\}) + W_s x \tag{2}
$$

$W_s$는 일반적으로 1×1 convolution(혹은 FC에서의 선형 변환)으로 구현되며, 논문은 **identity shortcut이 충분하고 경제적** 이라며 차원 매칭이 필요한 경우에만 $W_s$를 쓰겠다고 한다.

이 선택은 이후 bottleneck 블록에서 매우 중요해진다. 만약 모든 shortcut에 projection을 넣으면, high-dimensional feature에서 1×1 conv가 반복되며 시간/메모리/파라미터가 크게 증가한다. 반대로 identity를 기본으로 두고 필요할 때만 projection을 쓰면, bottleneck 설계를 경제적으로 유지할 수 있다.

논문은 이 결정을 단순히 계산량 절감으로만 설명하지 않는다. projection shortcut이 많아지면 모델이 커지고, 비교 실험에서 plain과의 공정성이 깨질 수 있다. 따라서 기본은 $(1)$의 parameter-free shortcut으로 두고, 차원이 달라져 $(1)$이 불가능한 지점에서만 $(2)$를 쓴다는 규칙이 자연스럽다. 이 규칙은 Fig. 3에서 dotted shortcut이 차원 증가 구간에만 등장하는 방식으로 시각화된다.

### 🔸 Residual Block 의사코드
논문에서 residual block은 다양한 형태(2-layer, bottleneck 3-layer)로 등장하지만, 계산 흐름은 공통적이다.

```text
Algorithm: Residual Block Forward (basic form)
Input: x
Parameters: residual layers for F, optional projection Ws

residual = x
out = F(x)                     # stack of 2 or 3 conv/BN/ReLU layers
if dimension mismatch:
    residual = Ws(x)           # projection shortcut (Eq. 2)
y = out + residual             # element-wise addition
y = ReLU(y)                    # in the paper’s described block
return y
```

이 의사코드는 논문 Fig. 2의 구조와 $(1)$, $(2)$를 그대로 반영한다. 결국 ResNet의 설계는 입력 경로는 최대한 단순하게 유지하고 변화량만 학습한다는 철학으로 요약된다.

#### Fig. 5의 두 가지 블록의 Residual 관점에서의 비교
논문은 ImageNet에서 **두 가지 형태**의 residual branch를 사용한다.

- ResNet-18/34: 2-layer block(`3×3-3×3`)  
- ResNet-50/101/152: bottleneck 3-layer block(`1×1-3×3-1×1`)

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/1f801180-1458-4ab2-a42d-f70453b55370/image.png" width="40%">
</p>

두 블록은 모두 $(1)$, $(2)$의 관점에서 동일하다. 입력 $x$를 shortcut으로 전달하고, residual branch가 $F(x)$를 계산한 뒤 더한다. 차이는 $F(x)$가 얼마나 경제적으로 계산되느냐에 있다. bottleneck은 3×3이 처리하는 채널 폭을 줄인 상태에서 공간 연산을 수행하도록 강제한다. 이는 ResNet이 깊이를 늘릴 때 연산량을 통제하려는 목적과 정합적이다.

특히 bottleneck에서 마지막 1×1 conv는 단순한 채널 복원 이상의 역할을 가진다. shortcut과 더하기를 하려면 residual branch 출력이 입력과 같은 채널 수를 가져야 하므로, 마지막 1×1은 블록의 출력 채널을 고정된 인터페이스로 맞추는 역할을 한다. 결과적으로 깊이가 커져도 블록 간 연결이 단순하게 유지된다.

#### Downsampling이 들어갈 때의 블록 구조
논문은 downsampling을 각 stage의 첫 블록에서 수행한다(`conv3_1`, `conv4_1`, `conv5_1`에서 stride 2). 이때 shortcut도 동일한 해상도 변화를 따라야 하므로, $(2)$의 projection shortcut이 자연스럽게 등장한다. 실제 구현에서는 다음 두 가지 변형이 흔히 쓰인다.

- shortcut에 stride 2 1×1 conv를 두는 방식(논문에서 option B를 기본으로 삼는 방식)  
- shortcut에 pooling을 둔 뒤 1×1 conv를 두는 방식(일부 구현에서 계산 특성을 위해 사용)

논문 자체는 첫 번째를 대표적인 projection shortcut으로 제시하고, 두 번째는 본문에서 중심으로 다루지는 않는다. 중요한 것은, downsampling 시점에서는 $(1)$의 단순 identity shortcut이 불가능하므로 $(2)$가 구조적으로 강제된다는 점이다.

### 🔹 Network Architectures: Plain vs Residual 비교
논문은 residual 구조의 효과를 설득력 있게 보여주기 위해, 비교 대상인 plain network를 신중하게 설계한다. 핵심은 _같은 depth/연산량 조건에서 shortcut의 효과만_ 을 보겠다는 것이다. 이를 위해 논문은 Fig. 3에서 VGG-19(참고), 34-layer plain, 34-layer residual을 나란히 제시한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/307a5c62-9116-4f89-8207-514c11c8b171/image.png" width="80%">
</p>

#### VGG 기반 Plain Baseline
논문이 쓰는 34-layer plain baseline은 레이어 구성이 **VGG의 철학**(작은 3×3 conv의 반복)을 따른다. 다만 VGG-19처럼 매우 넓은 채널을 쓰는 대신, stage별 채널 폭을 64/128/256/512(기준)으로 두고 반복 횟수로 깊이를 만든다. 이로 인해 VGG 대비 연산량이 낮으면서도, 충분히 깊은 plain 구조를 만들 수 있다.

논문은 이 점을 FLOPs로도 강조한다. 텍스트 추출본에 따르면 34-layer baseline은 3.6B FLOPs(multiply-adds)이며, 이는 VGG-19의 19.6B FLOPs 대비 _18% 수준_ 이라고 적는다. 즉, ResNet 계열은 깊이만 늘린 무거운 모델이 아니라, 비교적 경제적인 연산량 안에서 깊이를 확장하려는 설계라는 점을 함께 보여준다.

여기서의 요지는, plain baseline이 일부러 약하게 만든 모델이 아니라는 점이다. 당시 기준으로 합리적인 설계를 가진 plain model에서도, 깊이가 충분히 커지면 degradation이 나타난다는 것을 보여주는 데 목적이 있다.

#### Plain Baseline으로부터의 Residual 네트워크 설계
34-layer residual 네트워크는 34-layer plain과 동일한 conv 구성 위에 shortcut을 추가한다. 이때 shortcut은 파라미터가 없는 **identity 연결**이 기본이며, 차원이 바뀌는 지점에서만 projection – $(2)$ 또는 padding을 사용한다. 논문은 projection에 대해 option A/B/C를 비교하지만, 가장 핵심 메시지는 다음이다.

- degradation을 해결하는 데는 항상 모든 shortcut에 projection이 필요한 것이 아니다.
- 대부분의 블록은 $(1)$의 identity shortcut만으로도 효과가 크다.
- projection은 차원 매칭이 필요한 구간(해상도/채널 변화)에서만 쓰는 것이 자연스럽다.

이 구조는 Table 1과도 직접 연결된다. 네 stage(`conv2_x~conv5_x`)가 있고, 각 stage에서 동일 형태의 block이 반복되며, stage가 바뀌는 첫 블록에서만 stride 2로 downsampling이 들어간다. 따라서 block 반복 수와 stage 경계가 아키텍처 이해의 핵심이 된다.

#### 깊이(Depth) 표기가 의미하는 것
논문에서 말하는 34-layer, 50-layer 같은 깊이는 단순히 block 개수가 아니라, 학습 가능한 layer(주로 convolution/FC)를 기준으로 한다. 예를 들어 ResNet-34는 (stem conv1) + (각 basic block이 2 conv × 반복 수) + (마지막 fc)로 총 34개의 학습 layer가 되도록 구성된다. 이 계산이 Table 1의 반복 수와 맞물리며, 논문 전개에서도 Table 1을 반복해서 참조하는 이유가 된다.

### 🔸 구현 디테일: BN 배치, 초기화, 학습/테스트 규약
논문은 residual 구조가 새로운 _학습 알고리즘이 아니라는 점_ 을 분명히 하기 위해, 구현 레시피를 비교적 표준적인 것으로 유지한다. 여기서 표준적이라는 것은, 당시 ImageNet 학습에서 널리 쓰이던 SGD + momentum, weight decay, data augmentation, crop-based testing 등을 의미한다.

#### BatchNorm의 위치
ResNet 블록(특히 Fig. 2, Fig. 5)을 구현할 때 BN은 **각 convolution 뒤, 비선형(ReLU) 앞**에 들어간다. 그리고 residual branch의 마지막 linear 출력(마지막 conv + BN)과 shortcut을 더한 뒤에 ReLU를 적용하는 형태가 기본이다.

#### 테스트 시점의 평가 방식이 중요한 이유
논문은 10-crop testing, fully convolutional multi-scale testing 등 평가 방식에 따라 error가 달라질 수 있음을 전제로 한다. 그래서 Table 2/3/4 같은 비교 표에서는 동일한 10-crop 규약을 고정해 구조의 효과를 비교하고, 최고의 성능을 보고할 때는 multi-scale averaging을 사용한다(Table 5).

이 점은 리뷰를 읽을 때도 중요하다. 예컨대 Table 4(single-model)과 Table 5(ensemble)는 서로 다른 평가 프로토콜을 섞고 있으므로, ResNet의 구조적 효과를 논증할 때는 plain vs ResNet 비교가 있는 Table 2/3과 Fig. 4 같은 쪽이 더 직접적인 증거로 쓰인다.

#### 모델이 깊어질수록 구현에서 신경써야 하는 부분
논문이 직접 언급하듯, residual 구조가 최적화 난점을 크게 줄이더라도, 매우 깊은 모델에서는 학습률 스케줄이나 초기 몇백 iteration의 안정화 같은 디테일이 여전히 영향을 준다(CIFAR-10 warm-up 예시). 따라서 ResNet의 기여를 깊은 모델이 무조건 쉽게 학습된다는 의미로 오해하기보다는, 깊이를 늘릴 때 가장 큰 병목이었던 **degradation을 구조적으로 제거해 준 것**으로 이해하는 것이 정확하다.

---

## 4️⃣ 실험

### 🔹 ImageNet 분류: 구조와 학습 곡선으로 보는 효과
논문은 _ImageNet 2012_ 분류(1000 classes, 1.28M train, 50k val, 100k test)에서 top-1/top-5 error를 보고한다. 학습은 SGD 기반으로 수행하며, 일반적인 설정(모멘텀, weight decay, 10-crop 테스트, multi-scale 테스트 등)을 사용한다.

텍스트에 명시된 대표 설정을 정리하면 다음과 같다.

- mini-batch size: 256  
- 초기 learning rate: 0.1 (오류가 plateau에 도달할 때 10배 감소)  
- weight decay: 0.0001  
- momentum: 0.9  
- dropout 미사용  
- 테스트: 10-crop(비교 실험), fully-convolutional multi-scale averaging(최고 성능)

#### Implementation의 데이터 증강/테스트 규약
논문 3.4절에서는 실험 해석에서 빠뜨리기 쉬운 부분을 꽤 구체적으로 적어 둔다. ResNet이 구조 아이디어로 평가받는 논문이긴 하지만, ImageNet 규모에서는 데이터 증강과 테스트 프로토콜이 결과를 크게 좌우할 수 있기 때문에, 저자들은 비교 실험에서 프로토콜을 고정하고(best result에서는 프로토콜을 확장)하는 방식으로 논증을 구성한다.

**학습 데이터 증강(ImageNet)** 은 다음과 같이 요약된다.

- scale augmentation: 이미지의 짧은 변(shorter side)을 $[256, 480]$ 범위에서 랜덤 샘플링해 리사이즈  
- random crop: 224×224 crop을 랜덤으로 샘플링(수평 반전 포함)  
- mean subtraction: per-pixel mean을 빼는 전처리  
- color augmentation: 표준 색상 증강을 사용

이 레시피는 당시의 표준 관행을 따르되, 모델 구조의 효과를 비교하기 위해 모든 실험에서 동일하게 적용된다는 점이 중요하다.

**테스트 프로토콜**은 두 갈래로 분리된다.

1. 비교 실험(plain vs ResNet, 옵션 비교 등)에서는 10-crop testing을 사용한다.  
2. 최고 성능 보고에서는 fully-convolutional 테스트로 바꾼 뒤, 여러 scale에서 점수를 평균낸다.

논문은 multi-scale averaging에서 짧은 변을 다음 집합으로 리사이즈한다고 적는다.

$$
s \in \{224, 256, 384, 480, 640\}
$$

즉, 같은 모델이라도 **테스트 방식에 따라 결과가 달라질 수 있음을 인정**하고, 논증에 필요한 비교(구조 효과)와 최종 제출(최고 성능)의 목적을 분리해 서술한다.

#### BN 배치와 초기화 언급의 의미
논문은 BN을 각 convolution 뒤, activation 앞에 둔다고 명시한다. BN의 위치가 바뀌면 블록의 수치적 거동이 달라질 수 있고, residual add 이후의 activation과 결합될 때 학습 안정성이 변한다.

따라서 ResNet이 단지 구조만 바꿔서 되는 트릭이 아니라, 당시 알려진 표준 학습 레시피 안에서 구조가 만들어내는 효과를 보여주는 논문이라는 점이 다시 강조된다.

이 설정은 이후 실험(plain vs ResNet, 더 깊은 bottleneck) 전체에 걸쳐 공통 베이스라인으로 사용되며, 구조의 효과를 분명히 드러내는 역할을 한다.

#### Table 1의 해석
논문 Table 1은 ResNet 아키텍처의 **표준 레이아웃**을 정리한다. `conv1`은 7×7 stride 2, 그 뒤 3×3 maxpool stride 2가 이어지고, 이후 `conv2_x`부터 `conv5_x`까지 4개의 stage가 이어진다. downsampling은 `conv3_1`, `conv4_1`, `conv5_1`에서 stride 2로 수행된다고 명시된다. 이 표를 통해 ResNet의 핵심이 단지 residual block 자체뿐 아니라, block을 어떤 stage에서 몇 번 반복하는지가 모델 깊이를 결정한다는 점이 드러난다.

실제로 자주 쓰이는 반복 수는 다음과 같다(논문 Table 1, Fig. 5 대응).

- ResNet-18: `[2, 2, 2, 2]` (basic block)  
- ResNet-34: `[3, 4, 6, 3]` (basic block)  
- ResNet-50: `[3, 4, 6, 3]` (bottleneck)  
- ResNet-101: `[3, 4, 23, 3]` (bottleneck)  
- ResNet-152: `[3, 8, 36, 3]` (bottleneck)

여기서 bottleneck은 Fig. 5의 `1×1-3×3-1×1` 구조이며, stage 출력 채널은 64/128/256/512(기준)으로 늘어나는 대신 bottleneck의 마지막 1×1에서 4배 확장된 채널로 출력된다(예: 64→256).

Fig. 3은 ImageNet용 네트워크 구조 예시를 제시한다. VGG-19(참고), 34-layer plain, 34-layer residual을 비교하며, residual 네트워크의 dotted shortcut이 차원 증가를 나타낸다고 설명한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/293f3f5e-5ec0-412f-929c-6ff0e0df4f0d/image.png" width="80%">
</p>

#### Table 1을 텍스트로 재구성
논문은 Table 1에서 아키텍처를 stage 단위로 제시한다. 모델을 읽는 핵심은, 어떤 block을 어떤 stage에서 몇 번 반복하는지다. 아래는 논문 Table 1의 내용을 글에서 추적하기 쉽도록 정리한 형태다.

#### ResNet-18/34 (basic block, Fig. 2 기반)
| stage | output size | block | ResNet-18 | ResNet-34 |
|---|---:|---|---:|---:|
| conv1 | 112×112 | 7×7 conv, 64, stride 2 | 1 | 1 |
| pool | 56×56 | 3×3 max pool, stride 2 | 1 | 1 |
| conv2_x | 56×56 | [3×3, 64; 3×3, 64] | 2 | 3 |
| conv3_x | 28×28 | [3×3, 128; 3×3, 128] | 2 | 4 |
| conv4_x | 14×14 | [3×3, 256; 3×3, 256] | 2 | 6 |
| conv5_x | 7×7 | [3×3, 512; 3×3, 512] | 2 | 3 |
| head | 1×1 | avg pool, fc 1000 | 1 | 1 |

#### ResNet-50/101/152 (bottleneck, Fig. 5 기반)
| stage | output size | block | ResNet-50 | ResNet-101 | ResNet-152 |
|---|---:|---|---:|---:|---:|
| conv1 | 112×112 | 7×7 conv, 64, stride 2 | 1 | 1 | 1 |
| pool | 56×56 | 3×3 max pool, stride 2 | 1 | 1 | 1 |
| conv2_x | 56×56 | [1×1, 64; 3×3, 64; 1×1, 256] | 3 | 3 | 3 |
| conv3_x | 28×28 | [1×1, 128; 3×3, 128; 1×1, 512] | 4 | 4 | 8 |
| conv4_x | 14×14 | [1×1, 256; 3×3, 256; 1×1, 1024] | 6 | 23 | 36 |
| conv5_x | 7×7 | [1×1, 512; 3×3, 512; 1×1, 2048] | 3 | 3 | 3 |
| head | 1×1 | avg pool, fc 1000 | 1 | 1 | 1 |

이 표를 보면 ResNet-34와 ResNet-50은 stage 반복 수가 같지만, block 내부 레이어 수가 달라 총 깊이가 달라진다는 점이 분명해진다. 또한 50/101/152는 bottleneck을 통해 깊이를 늘리되 계산량을 통제하는 전략을 취한다.

#### Bottleneck의 경제적 이점
논문은 bottleneck 설계를 단지 구조적으로 예쁘기 때문에 쓰는 것이 아니라, 학습 시간을 **감당 가능한 범위로 유지**하기 위한 실무적 선택이라고 설명한다. 2-layer basic block(`3×3-3×3`)을 그대로 유지한 채 50/101/152처럼 매우 깊게 만들 수도 있지만, 이는 계산량이 빠르게 증가한다.

Bottleneck은 1×1 conv로 채널 폭을 줄인 뒤 3×3 공간 연산을 수행하고, 마지막 1×1 conv로 채널을 복원한다. 이 설계 덕분에, 깊이를 크게 늘려도 FLOPs가 폭발하지 않는다.

- 34-layer baseline: 3.6B FLOPs  
- ResNet-50: 3.8B FLOPs  
- ResNet-152: 11.3B FLOPs  
- VGG-16/19: 15.3B / 19.6B FLOPs

즉, ResNet-50은 34-layer baseline과 거의 비슷한 수준의 계산량으로 더 깊어질 수 있고, ResNet-152도 당시의 VGG-16/19보다 계산량이 낮게 유지된다. 이 숫자들은 ResNet의 핵심이 단지 정확도를 올리는 것뿐 아니라, 정확도를 올리기 위해 깊이를 늘릴 때의 비용을 함께 설계한 아키텍처라는 점을 강조한다.

가장 핵심 실험은 18-layer와 34-layer의 plain vs ResNet 비교다. Table 2는 10-crop 테스트 기준의 top-1 error를 제공한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/6c853512-1cea-485c-97e7-2d6e786884ef/image.png" width="40%">
</p>

즉, plain에서는 34-layer가 18-layer보다 더 나쁘지만($27.94 \to 28.54$), ResNet에서는 34-layer가 18-layer보다 확실히 더 좋다($27.88 \to 25.03$). 이 결과는 residual 구조가 degradation problem을 완화하고, 깊이가 실제 성능 향상으로 이어지게 만든다는 것을 보여준다.

이 표를 논문 논증의 관점에서 다시 보면, ResNet의 효과는 **두 가지 층위**로 나타난다.

1. 최적화 측면: 34-layer ResNet은 18-layer ResNet보다 낮은 training error로 내려간다.  
2. 일반화 측면: training error가 낮아지는 방향이 validation 성능 개선으로도 이어진다.

반대로 plain 네트워크에서는 더 깊은 모델이 학습 내내 더 높은 training error를 보이며, 이는 추가 깊이가 단순히 낭비되는 것이 아니라 optimizer가 좋은 해를 찾지 못한다는 의미로 해석된다.

Fig. 4는 학습 과정의 training/validation error 곡선을 통해 이 현상을 시각적으로 확인한다. plain(왼쪽)에서는 더 깊은 모델이 학습 내내 더 높은 training error를 보이지만, ResNet(오른쪽)에서는 깊은 모델이 더 낮은 training error로 내려가며 일반화까지 좋아진다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/404914a2-9aaf-484c-896a-acf3f433c7aa/image.png" width="70%">
</p>

#### 18-Layer에서 차이가 상대적으로 작게 보이는 이유

논문은 _흥미로운 관찰_ 을 하나 덧붙인다. Table 2에서 18-layer plain과 18-layer ResNet의 성능은 거의 비슷하다($27.94$ vs $27.88$). 이만 보면 residual이 꼭 필요 없어 보일 수도 있다. 하지만 논문은 Fig. 4의 학습 곡선을 근거로, ResNet이 단지 최종 성능을 올리는 것뿐 아니라 **수렴을 더 빠르게 만든다**는 점을 강조한다.

텍스트 추출본에는 다음과 같은 취지의 설명이 포함된다.

- 18-layer처럼 충분히 얕은 경우에는, 현재의 SGD solver도 plain net에서 좋은 해를 찾을 수 있다.
- 이 경우에도 ResNet은 초기 학습 단계에서 더 빠른 수렴을 제공해 최적화를 쉽게 만든다.

이 관찰은 ResNet의 기여를 더 정교하게 이해하게 만든다. Residual 연결은 단지 아주 깊은 모델에서만 의미가 있는 장치가 아니라, 비교적 얕은 모델에서도 최적화 경로를 단순화해 학습을 빠르게 하는 효과를 낸다. 다만 얕은 모델에서는 plain도 충분히 학습되기 때문에, 최종 성능 차이가 작게 보일 수 있다.

반대로 34-layer처럼 더 깊어지면, plain에서는 solver가 좋은 해로 내려가지 못하는 현상이 직접적으로 드러나고(훈련 오차 상승), residual에서는 그 현상이 사라진다. 즉, ResNet의 효과는 깊이에 따라 연속적으로 나타나지만, 어느 깊이 이상에서는 그 효과가 최종 성능 차이로도 크게 표면화된다고 해석할 수 있다.

### 🔸 Project Shortcut 옵션 A/B/C와 결론
차원 증가 시 shortcut을 어떻게 처리할지에 대해 논문은 **세 가지 옵션**을 비교한다.

- **A**: 차원 증가 시 zero-padding(파라미터 없음), 나머지는 identity  
- **B**: 차원 증가 시 projection – $(2)$, 나머지는 identity  
- **C**: 모든 shortcut을 projection

논문은 Table 3에서 A/B/C가 모두 plain보다 훨씬 좋고, B가 A보다 약간 좋으며, C는 B보다 아주 조금 좋지만 추가 파라미터의 영향으로 해석된다고 말한다. 중요한 결론은 projection이 degradation 해결에 필수는 아니며, 이후 실험에서는 비용을 줄이기 위해 option C는 사용하지 않는다는 것이다.

이 논의는 identity shortcut이 본질적이라는 논문의 메시지를 강화한다. projection은 차원 매칭이 필요한 경우에만 쓰는 것이 경제적이며, bottleneck에서도 identity를 최대한 유지하는 것이 유리하다는 결론으로 이어진다.

이 결론은 ResNet을 이해할 때 중요한 실무적 메시지로도 연결된다. shortcut을 더 정교하게 만들기 위해 무조건 projection을 쓰는 것이 아니라, 최소한의 projection으로도 optimization 난점을 상당 부분 해결할 수 있다는 것이다. 즉, ResNet의 힘은 복잡한 shortcut이 아니라, **항상 존재하는 identity 경로** 그 자체에서 온다.

#### Option A의 의미
Option A는 차원이 늘어나는 시점에 shortcut 경로로 projection을 넣지 않고, zero-padding으로 채널을 맞춘다. 즉 늘어난 채널 부분은 shortcut을 통해 정보가 직접 전달되지 않고, residual branch를 통해서만 학습된다. 논문은 이 때문에 A가 B보다 약간 성능이 낮을 수 있다고 해석한다. 하지만 A도 이미 plain 대비 큰 개선을 주며, 이는 projection의 존재보다 **identity 경로의 존재가 훨씬 큰 효과**를 가진다는 논증으로 이어진다.

### 🔹 Bottleneck 설계와 깊은 모델 – (50/101/152)
더 깊은 모델을 만들기 위해 논문은 Fig. 5의 bottleneck block을 제안한다. **ResNet-34**에서는 2-layer block(`3×3-3×3`)을 쓰지만, **ResNet-50/101/152**에서는 3-layer(`1×1-3×3-1×1`)로 바꾼다. 첫 1×1은 채널을 줄여 계산을 줄이고, 마지막 1×1은 채널을 복원한다. 이렇게 하면 depth는 크게 늘리면서도 시간 복잡도는 합리적으로 유지할 수 있다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/20aba7e4-b325-4566-b242-195cea05e8d4/image.png" width="40%">
</p>

여기서 bottleneck이 중요한 이유는 단순히 파라미터를 줄이기 때문만이 아니다. ResNet은 stage가 깊어질수록 채널 수가 커지는 구조를 가지는데, 이때 3×3 conv를 큰 채널에서 반복하면 _비용이 급증_ 한다. bottleneck은 1×1로 채널을 줄여 3×3이 처리하는 채널 폭을 제한하고, 다시 1×1로 확장해 표현력을 유지한다. 따라서 깊이를 늘릴 때 비용을 통제한다는 논문의 목적과 정확히 맞아떨어진다.

Table 3은 ImageNet val에서 VGG-16과 다양한 ResNet 변형의 error를 제공한다(10-crop). 텍스트 추출본에서는 수치들이 일렬로 나열되어 있으므로, 논문 표의 핵심 비교를 다음과 같이 정리할 수 있다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/e3c18b56-faae-48c5-b01f-4eb21a702523/image.png" width="40%">
</p>

Table 3에서 관찰되는 큰 흐름은 다음과 같다.

- ResNet-34(B/C)는 plain 대비 크게 개선된다.
- 더 깊은 ResNet-50/101/152는 추가적인 개선을 제공한다(특히 top-5에서 의미 있는 감소).

또한 Table 4는 single-model 기준으로 다양한 SOTA 방법과 ResNet을 비교한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/eb5cc62b-d3ab-417a-adfd-b356257f7a7a/image.png" width="40%">
</p>

마지막으로 Table 5는 ensemble 결과를 비교하며, ResNet(ILSVRC’15)이 top-5 test error $3.57$을 보고한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/620d35f9-b93e-4620-98fc-38fdfc35a9c1/image.png" width="40%">
</p>

### 🔸 CIFAR-10: 1000+ 레이어의 학습 가능성
논문은 CIFAR-10에서도 degradation이 나타나며, ResNet이 이를 해결해 더 깊은 모델에서 정확도가 향상됨을 보여준다. 또한 $n=200$으로 **1202-layer 네트워크**까지 실험해, 최적화 어려움 없이 training error가 거의 0에 도달할 수 있음을 보고한다. 다만 1202-layer는 overfitting 등으로 110-layer보다 test 성능이 나쁜데, 이는 작은 데이터셋에서 **모델 용량이 과도할 수 있음**을 시사한다.

Fig. 6은 CIFAR-10에서 plain vs ResNet의 학습 곡선을 제시한다. plain은 깊어질수록 training error가 줄지 않고 오히려 커지지만, ResNet은 깊어질수록 training error가 안정적으로 감소하며 test 성능도 개선된다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/1a130b2e-eb4a-429c-9ff5-67ef60c5c417/image.png" width="70%">
</p>

Table 6은 CIFAR-10 test error를 다양한 방법과 비교한다. ResNet-110의 경우 여러 번 실행한 best와 (mean±std)를 보고한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/14fced24-9686-4ac6-9fe1-aee19bdd5209/image.png" width="40%">
</p>

Table 6에서 ResNet 관련 핵심 수치는 다음과 같이 요약할 수 있다.

- `ResNet-20`: $8.75$  
- `ResNet-32`: $7.51$  
- `ResNet-44`: $7.17$  
- `ResNet-56`: $6.97$  
- **`ResNet-110`**: $\boxed{6.43~(6.61\pm0.16)}$
- `ResNet-1202`: $7.93$

즉, 110-layer까지는 깊이가 성능 향상으로 이어지지만, 1202-layer는 최적화는 되더라도 **일반화가 악화될 수 있음**을 보여준다. 논문은 이 지점에서 더 강한 정규화(maxout, dropout 등)와의 결합 가능성을 언급하며, 본 논문은 최적화 문제에 초점을 맞추기 위해 이를 사용하지 않았다고 정리한다.

#### 1202-Layer가 말해주는 것
논문이 CIFAR에서 가장 인상적으로 보여주는 장면은, 1000+ 레이어에서도 optimization difficulty가 크게 나타나지 않는다는 점이다. 실제로 텍스트 추출본은 $n=200$(1202-layer) 모델이 training error $<0.1$%까지 내려갈 수 있다고 설명한다. 즉, residual 구조는 최소한 훈련 관점에서는 **매우 깊은 모델을 가능**하게 만든다.

하지만 test error는 110-layer($6.43$)보다 1202-layer($7.93$)가 나쁘다. 논문은 이를 overfitting의 징후로 해석하며, 작은 데이터셋에서 모델 용량이 과도해질 수 있다고 지적한다. 텍스트 추출본에는 1202-layer 네트워크가 19.4M 파라미터 규모라는 언급이 함께 나온다. 따라서 이 실험은 다음을 동시에 말해준다.

- degradation problem은 주로 **최적화 문제**였고, residual은 이를 해결한다.
- 그 다음 단계로 남는 문제는 일반화이며, 이는 **정규화/데이터 규모/모델 용량**과 깊게 연결된다.

논문이 maxout이나 dropout 같은 강한 정규화를 굳이 쓰지 않은 이유도 여기에서 나온다. 본 논문은 일반화 테크닉을 결합해 최고 성능을 찍는 것이 아니라, optimization 관점에서 residual의 효과를 분리해 보여주는 데 초점을 두고 있기 때문이다.

#### Layer Response 분석으로 보는 Residual의 성질
논문은 CIFAR-10 실험 뒤에 Fig. 7을 통해 residual 연결이 학습된 표현에 어떤 흔적을 남기는지 분석한다. 여기서 layer response는 각 3×3 layer의 출력(특히 BN 이후, 다른 비선형(ReLU/addition) 이전)을 기준으로 하고, 그 표준편차(std)를 레이어별로 계산한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/5ef84db2-e593-4f9e-83fb-a96a039a17dd/image.png" width="40%">
</p>

이 분석에서 논문이 강조하는 해석은 다음과 같다.

- ResNet의 response는 plain 대비 전반적으로 작다.
- 더 깊은 ResNet일수록(20 → 56 → 110) 각 층이 신호를 수정하는 정도가 더 작아지는 경향이 있다.

이 결과는 Sec. 3.1의 주장(잔차 함수 $F$는 $0$에 가까운 변화량을 학습하는 경향이 있다)을 뒷받침한다. 즉, residual learning은 단지 학습을 가능하게 만들었을 뿐 아니라, 학습된 모델 자체도 입력 표현을 큰 폭으로 뒤집기보다 **작은 수정의 누적으로 구성되는 형태**로 나타난다는 것이다.

---

## 💡 해당 논문의 시사점과 한계
ResNet의 가장 큰 의의는 깊이를 무작정 늘리는 것이 아니라, 깊어질수록 발생하는 최적화 문제를 구조적으로 해결하는 방식으로 접근했다는 점이다. $(1)$의 단순한 더하기가 제공하는 identity 경로는 깊은 네트워크를 사실상 잔차의 누적으로 재해석하게 만들고, degradation problem을 실험적으로 해소했다. 그 결과 ResNet-50/101/152 같은 매우 깊은 모델이 ImageNet에서 SOTA를 달성했고, 이 표현은 detection/localization 등 다운스트림 과제에서도 큰 개선으로 이어졌다.

한계도 있다. 매우 깊은 모델(1202-layer)에서는 최적화는 되더라도 일반화가 나빠질 수 있음을 보여주며, 이는 작은 데이터셋에서 용량이 과도해질 때 정규화가 중요해질 수 있음을 시사한다. 즉, ResNet이 열어준 설계 공간은 이후 많은 후속 작업으로 확장되며, 본 논문은 그 출발점에서 가장 단순한 형태의 residual 연결이 갖는 힘을 보여준다.

#### 분야 관점에서의 영향
ResNet의 영향은 두 층짜리 블록 하나를 제안했다는 수준이 아니다. 실제 제품/서비스에서는 더 좋은 backbone이 곧바로 downstream 모델의 성능과 비용을 바꾸는데, ResNet은 backbone을 **더 깊게 키우는 길을 사실상 표준화**했다. 또한 residual 연결은 이후 네트워크 설계에서 필수적인 구성 요소로 자리잡아, 다양한 모듈(예: attention, feed-forward block, U-Net 계열의 skip)로 재해석되어 사용되었다. 결국 ResNet은 하나의 아키텍처라기보다, 깊은 모델을 설계할 때의 _기본 문법_ 을 제공했다고 볼 수 있다.

여기서 딥러닝 분야라는 표현을 조금 더 구체적으로 풀면, 다음과 같은 변화가 있었다고 볼 수 있다.

1. **더 깊은 backbone이 실무 기본값이 됨**: ResNet-50/101 같은 깊이는 이후 많은 파이프라인에서 기본 선택지가 된다.  
2. **전이 학습의 표준화**: ImageNet으로 사전학습한 ResNet이 downstream fine-tuning의 기본 출발점이 된다.  
3. **모듈 설계의 재사용**: residual block 패턴이 분류를 넘어 encoder-decoder, multi-branch, attention 모듈로 흡수된다.  
4. **학습 안정성에 대한 관점 변화**: 더 깊은 모델을 만들면 우선 학습이 되느냐부터 걱정해야 했지만, residual 연결 이후에는 같은 계산 예산에서 모델 용량을 어디까지 키울지로 논쟁이 이동한다.

#### 연구/실무에서 남긴 메타 레벨의 변화
ResNet 이후의 연구 흐름을 보면, 새로운 모델을 제안할 때 먼저 ResNet 계열 backbone과 비교하는 것이 **사실상 기본 관례**가 되었다. 이는 단지 ResNet이 유명해서가 아니라, 깊이 확장에 대한 baseline이 명확해졌기 때문이다. 예전에는 네트워크를 더 깊게 만들면 학습이 되는지 자체가 연구 주제가 될 정도로 불확실성이 컸지만, residual 연결 이후에는 깊이 확장 자체는 비교적 안정적인 엔지니어링 선택이 되고, 그 위에서 폭(width), 모듈 구조, attention, multi-branch 등 다른 설계 축을 더 자유롭게 탐색할 수 있게 되었다.

실무 관점에서도 같은 구조가 반복된다. 모델을 바꾸는 가장 비용 효율적인 방법 중 하나는 backbone을 교체하는 것이고, ResNet 계열은 오랫동안 그 교체의 기본 선택지가 되어 왔다. 특히 detection/segmentation처럼 backbone 성능이 전체 성능을 크게 좌우하는 과제에서는, ResNet-50/101/152가 매우 오랫동안 표준으로 사용되었다. 논문 본문에서 분류 결과를 detection/localization까지 확장해 증명한 것이, 이 흐름의 초기에 매우 강한 근거를 제공했다.

#### ResNet이 해결한 것과 남긴 것
ResNet이 해결한 것은 깊이를 늘릴 때 가장 먼저 부딪히는 **최적화 장벽(degradation)** 이다. 하지만 residual 연결이 일반화 문제까지 자동으로 해결해 주는 것은 아니다. CIFAR의 1202-layer 결과처럼, 최적화는 성공해도 일반화가 나빠질 수 있고, 이 지점에서는 데이터 규모/정규화/모델 용량 조절이 다시 핵심이 된다.

따라서 ResNet의 영향은 단순히 더 깊게 만들 수 있다는 것에 그치지 않는다. 깊이를 안정적으로 늘릴 수 있게 되었기 때문에, 일반화와 효율(연산량, 메모리) 같은 문제를 더 정교하게 다루는 후속 연구들이 자연스럽게 이어질 수 있었다는 점이 더 큰 의미다.

이 논문이 이후 수많은 후속 연구에서 기본 참고 문헌으로 등장하는 이유도, 성능 표 하나의 결과라기보다 설계 패턴과 실험 논증 방식이 재사용 가능했기 때문이다.

---

## 👨🏻‍💻 ResNet 구현하기
이 파트에서는 [`lucid`](https://github.com/ChanLumerico/lucid)라이브러리의 [`resnet.py`](https://github.com/ChanLumerico/lucid/blob/main/lucid/models/imgclf/resnet.py)에 구현된 ResNet을 논문 관점으로 읽는다. 논문은 ImageNet용 ResNet-18/34(2-layer block)와 ResNet-50/101/152(bottleneck), 그리고 CIFAR용 매우 깊은 버전까지 다루지만, Lucid 구현은 이미지 분류 모델로서 ResNet 계열을 구성할 수 있는 일반화된 빌더를 제공한다.

Lucid 구현의 핵심 대응 관계는 다음과 같다.

- 논문 Fig. 2의 residual block – $(1)$ → `_BasicBlock`, `_Bottleneck`의 `out += identity`
- 논문 $(2)$의 projection shortcut → `ResNet._make_layer()`에서 생성하는 `downsample`(1×1 conv + BN 또는 avg-down)
- 논문 Fig. 5의 bottleneck block → `_Bottleneck`의 (1×1, 3×3, 1×1) 구조

이 글에서는 논문 본문에서 직접 다루는 **ResNet-18/34/50/101/152**의 구현 흐름에 집중한다.

#### Lucid 구현에서의 논문 설정 재현
Lucid `ResNet`은 여러 변형 옵션을 갖지만, 논문에 가장 가까운 형태는 기본 stem(7×7 conv)과 기본 downsample(stride가 걸린 1×1 conv)이며, `avg_down=False`, `stem_type=None` 조합으로 볼 수 있다. deep stem이나 avg-down은 다른 논문/후속 변형에서 자주 등장하는 선택지이므로, 이 리뷰에서는 논문과 직접 대응되는 기본값을 중심으로 설명한다.

### 0️⃣ 사전 설정 및 모델 엔트리
`resnet.py`는 ResNet의 공통 빌더(`ResNet`)와 residual block 구현(`_BasicBlock`, `_Bottleneck`)을 제공하고, 논문에서 다루는 깊이(18/34/50/101/152)에 해당하는 모델을 `@register_model` 팩토리로 노출한다. 아래 해설은 이 다섯 모델과, 그 모델들이 공유하는 구현 요소들만을 다룬다.

또한 이 파일의 구성은 논문을 읽는 순서와도 잘 맞는다. 먼저 $(1)$, $(2)$에 해당하는 residual 덧셈은 `_BasicBlock/_Bottleneck`에, stage 반복과 downsample 규칙은 `ResNet`과 `_make_layer`에, 마지막으로 Table 1의 반복 수는 각 `resnet_*` 팩토리 함수에 들어 있다. 따라서 논문에서 수식→블록→아키텍처로 전개되는 흐름을 코드에서도 거의 그대로 추적할 수 있다.

### 1️⃣ 전체 네트워크 골격: `ResNet`
`ResNet` 클래스는 어떤 block을 몇 번 쌓을지를 인자로 받아 네트워크를 조립한다. 논문 Table 1의 핵심(각 stage에서 block 반복 수가 모델을 결정)을 코드로 일반화한 형태다.

```python
class ResNet(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        layers: list[int],
        num_classes: int = 1000,
        in_channels: int = 3,
        stem_width: int = 64,
        stem_type: Literal["deep"] | None = None,
        avg_down: bool = False,
        channels: tuple[int] = (64, 128, 256, 512),
        block_args: dict[str, Any] = {},
    ) -> None:
        super().__init__()
        deep_stem = stem_type == "deep"
        self.in_channels = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
```

여기서 `block`은 residual block 구현체(`_BasicBlock`, `_Bottleneck`) 중 하나이고, `layers`는 4개 stage에서 block을 몇 번 반복할지를 나타낸다. 예를 들어 ResNet-34는 `[3, 4, 6, 3]`이다(논문 Table 1의 반복 수와 대응).

논문 Table 1에서 stage는 `conv2_x`부터 `conv5_x`까지 네 구간으로 나뉘며, 출력 해상도는 $56→28→14→7$로 줄어든다. Lucid 구현에서도 stem과 maxpool 이후 `layer1..layer4`를 차례로 적용하며, `layer2..layer4`의 첫 블록 stride가 2인 점이 곧 해상도 감소 시점을 의미한다. 따라서 `layers` 리스트는 단지 반복 수가 아니라, 논문이 설계한 stage 구조를 코드로 표현하는 핵심 인자다.

#### `channels`가 논문 설계 규칙을 담는 방식
논문은 Fig. 3의 plain baseline을 설명하면서 두 가지 단순한 설계 규칙을 제시한다.

1. 같은 output feature map 크기에서는 같은 수의 필터를 사용한다.  
2. feature map 크기가 절반으로 줄어들면, 필터 수는 2배로 늘린다(층당 시간 복잡도 유지 목적).

Lucid `ResNet`의 `channels: tuple[int] = (64, 128, 256, 512)`는 이 규칙을 그대로 코드로 고정한 값이라고 볼 수 있다. `layer1..layer4`가 각각 `conv2_x`$\ldots$`conv5_x`에 대응하고, stage가 바뀔 때 stride 2로 해상도가 절반이 되므로, 그때 채널 수가 2배로 바뀌는 구성이 된다.

즉, 논문 Table 1을 보고 stage별 채널이 어떻게 바뀌는지 이해했다면, Lucid에서는 그 규칙이 `channels` 튜플 하나로 응축되어 있다고 보면 된다. 반대로 `channels`를 바꾸면 ResNet의 폭(width) 스케일링을 손쉽게 실험할 수 있지만, 이 리뷰에서는 논문과 직접 대응되는 기본값($64,128,256,512$)을 기준으로 한다.

#### `stem_width`와 `self.in_channels`의 의미
`stem_width`는 deep stem을 쓸 때 stem 내부 채널 폭을 결정한다. 코드에서

- 표준 stem(`stem_type=None`)이면 `self.in_channels = 64`  
- deep stem(`stem_type="deep"`)이면 `self.in_channels = stem_width * 2`

로 설정된다. deep stem이 3×3 conv를 여러 번 쌓는 구성이라는 점을 생각하면, stem의 마지막 출력 채널을 어떻게 둘지가 곧 뒤의 stage(`layer1`) 입력 채널을 결정한다. Lucid는 deep stem에서 마지막 출력 채널을 `stem_width * 2`로 두어, stem 내부 폭을 바꿔도 이후 stage가 과도하게 좁아지지 않도록 설계했다.

#### `block_args`가 하는 일
`block_args: dict[str, Any] = {}`는 `ResNet`이 block 구현체를 호출할 때 추가 인자를 전달하기 위한 통로다. `_make_layer()`를 보면 첫 블록과 이후 블록을 만들 때 모두 `**block_args`가 전달된다.

이 구조는 논문 관점에서 보면 다음과 같이 읽을 수 있다.

- 논문 Fig. 2/5의 블록 형태(기본/병목)를 고정한 상태에서, 블록 내부의 세부 동작을 변형할 수 있다.  
- 하지만 논문 자체가 다루는 _ResNet-18/34/50/101/152_ 는 이 변형이 필요하지 않으므로, 리뷰에서는 `block_args`가 비어 있는 기본 형태를 기준으로 생각하면 된다.

다만 코드 독해 관점에서는, `block_args`가 곧 블록 생성 시그니처로 그대로 흘러 들어간다는 점(예: `_Bottleneck.__init__`의 추가 인자들)을 눈여겨봐야 한다. 같은 `ResNet` 골격 위에서 다양한 변형을 수용할 수 있는 확장 지점이기 때문이다.

또한 stem 구성과 downsample 방식도 옵션으로 제공한다.

- `stem_type="deep"`: 7×7 대신 3×3을 여러 번 쓰는 deep stem
- `avg_down=True`: projection shortcut에서 stride를 avgpool로 먼저 처리한 뒤 1×1 conv를 쓰는 avg-down 방식

논문 원문은 표준 7×7 stem과 stride 2 projection을 사용하지만, Lucid는 여러 변형을 실험 가능하게 일반화해 두었다.

### 2️⃣ Stem 구성: 표준 7×7 또는 Deep-Stem
`ResNet.__init__`는 stem을 두 가지 방식 중 하나로 만든다.

표준 stem은 논문 Fig. 3/Table 1에서 흔히 보이는 7×7 stride 2 + BN + ReLU다.

```python
self.stem = nn.Sequential(
    nn.Conv2d(
        in_channels, self.in_channels, 7, stride=2, padding=3, bias=False
    ),
    nn.BatchNorm2d(self.in_channels),
    nn.ReLU(),
)
```

deep stem은 3×3 conv를 세 번 쌓아 7×7을 대체하는 변형이다.

```python
self.stem = nn.Sequential(
    nn.Conv2d(in_channels, stem_width, 3, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(stem_width),
    nn.ReLU(),
    nn.Conv2d(stem_width, stem_width, 3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(stem_width),
    nn.ReLU(),
    nn.Conv2d(stem_width, self.in_channels, 3, padding=1, bias=False),
)
```

이 선택은 ResNet의 핵심 아이디어(residual block)와 직접 관련되기보다는, 초기 표현 학습의 성질과 계산량/성능 트레이드오프를 실험하기 위한 확장으로 볼 수 있다.

논문 관점에서 중요한 것은 stem 이후에 maxpool이 이어지고, 이후 stage들이 residual block 반복으로 구성된다는 큰 흐름이다. 즉 ResNet의 본질은 conv1의 커널 크기 선택이 아니라, `conv2_x` 이후 블록에서 $(1)$, $(2)$의 덧셈이 어떻게 반복되는가에 있다.

### 3️⃣ Stage 구성과 downsample: `_make_layer`
ResNet은 4개의 stage(`layer1`$\ldots$`layer4`)를 만들며, 각 stage는 `_make_layer`가 담당한다.

```python
self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

self.layer1 = self._make_layer(
    block, channels[0], layers[0], stride=1, block_args=block_args
)
self.layer2 = self._make_layer(
    block, channels[1], layers[1], stride=2, block_args=block_args
)
self.layer3 = self._make_layer(
    block, channels[2], layers[2], stride=2, block_args=block_args
)
self.layer4 = self._make_layer(
    block, channels[3], layers[3], stride=2, block_args=block_args
)
```

논문 Table 1에서 downsampling은 각 stage의 첫 블록에서 stride 2로 수행된다. Lucid도 `layer2`$\ldots$`layer4`의 첫 블록에 stride 2를 부여하는 방식으로 이를 반영한다.

`_make_layer`는 shortcut에 해당하는 downsample을 만들지 여부를 결정한다. $(1)$에서의 identity shortcut이 가능한 조건은 **`stride=1`이고 채널 수가 맞는 경우**다. 그렇지 않으면 $(2)$에 해당하는 projection shortcut이 필요하다.

```python
def _make_layer(
    self,
    block: Type[nn.Module],
    out_channels: int,
    blocks: int,
    stride: int = 1,
    block_args: dict[str, Any] = {},
) -> nn.Sequential:
    downsample = None
    if stride != 1 or self.in_channels != out_channels * block.expansion:
        if self.avg_down:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        else:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
```

여기서 `block.expansion`은 bottleneck 계열에서 채널이 4배로 늘어나는 점(논문 Fig. 5의 마지막 1×1이 확장)을 반영한다. 즉, projection shortcut이 필요한지 여부는 블록 출력 채널 기준으로 판단된다.

이 로직이 곧 논문 $(2)$의 구현 조건이다. 해상도가 바뀌거나 채널 수가 바뀌면, 단순히 `identity=x`로는 `out += identity`가 불가능하므로 downsample 경로를 만들어 차원을 맞춘다. 반대로 같은 stage 내부에서 stride=1이고 채널 수가 같은 블록은 downsample 없이 $(1)$의 identity shortcut을 그대로 사용한다. 논문이 말하는 경제적 identity shortcut이 코드로는 downsample 조건문으로 구체화된다.

그 다음 첫 블록은 stride와 downsample을 함께 전달받고, 이후 블록들은 `stride=1`로 반복된다.

```python
layers = [
    block(self.in_channels, out_channels, stride, downsample, **block_args)
]
self.in_channels = out_channels * block.expansion

for _ in range(1, blocks):
    layers.append(block(self.in_channels, out_channels, stride=1, **block_args))

return nn.Sequential(*layers)
```

#### 첫 블록과 이후 블록의 역할 분담
논문 Table 1에서 각 stage의 첫 블록(`conv3_1`, `conv4_1`, `conv5_1`)은 downsampling(stride 2)과 채널 증가가 동시에 일어난다. 그래서 이 블록에서는 shortcut이 identity로 유지될 수 없고, projection shortcut – $(2)$이 필요해진다. Lucid 코드에서 그 역할이 정확히 다음 두 줄로 나타난다.

- 첫 블록 생성 시 `stride`와 `downsample`을 함께 전달  
- `self.in_channels = out_channels * block.expansion`으로 이후 블록들이 기대하는 입력 채널을 갱신

`self.in_channels`를 갱신하는 부분은 코드 독해에서 특히 중요하다. 첫 블록을 지난 뒤에는 feature map의 채널 수가 이미 stage의 출력 채널(혹은 bottleneck이면 확장된 채널)로 바뀌므로, 같은 stage 안에서 반복되는 나머지 블록들은 입력/출력 채널이 맞아 떨어지고, 따라서 shortcut을 identity로 두는 $(1)$ 형태가 가능해진다.

반대로 이 갱신이 없다면, 두 번째 블록부터는 `in_channels`가 잘못 전달되어 convolution의 입력 채널 수가 맞지 않거나, 불필요한 downsample이 계속 만들어지는 등의 문제가 생길 수 있다. 즉, `self.in_channels`는 ResNet을 조립하는 상태(state)이고, stage 경계에서만 값이 바뀌는 것이 ResNet의 구조를 코드 수준에서 보장한다.

#### `nn.Sequential`로 stage를 묶는 이유
`_make_layer`는 여러 블록을 리스트로 쌓은 뒤 `nn.Sequential(*layers)`로 반환한다. 이 설계는 논문 Table 1의 stage 개념과 잘 맞는다.

- stage는 동일한 블록의 반복으로 정의된다.
- stage 내부에서는 stride=1로 반복되며, shortcut은 주로 identity로 유지된다.
- stage 경계에서만 downsampling과 projection shortcut이 등장한다.

즉, stage를 하나의 모듈로 묶으면(`layer1`$\ldots$`layer4`), `ResNet.forward`에서는 논문처럼 stage를 순서대로 적용하는 구조가 된다. Lucid의 forward가 `for layer in [self.layer1, ...]`로 반복하는 것도, stage 단위로 모델을 이해하라는 논문 전개와 자연스럽게 대응된다.

### 4️⃣ 전체 Forward
논문 ResNet의 큰 흐름은 stem과 4개의 stage를 거친 뒤 global average pooling과 linear classifier로 마무리된다. Lucid의 `ResNet.forward`는 이를 직접 구현한다.

```python
def forward(self, x: Tensor) -> Tensor:
    x = self.stem(x)
    x = self.maxpool(x)

    for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
        x = layer(x)

    x = self.avgpool(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fc(x)

    return x
```

`avgpool`은 `AdaptiveAvgPool2d((1,1))`로 구현되어 입력 해상도에 관계없이 $1×1$로 줄이는 방식이다. 논문에서의 global average pooling과 같은 목적을 가진다.

#### `fc` 입력 차원과 `block.expansion`의 연결
논문 Table 1을 보면 마지막 stage(`conv5_x`)의 출력 채널은 block 형태에 따라 달라진다.

- **basic block**(_ResNet-18/34_): `conv5_x`는 $512$ 채널을 유지한다.  
- **bottleneck**(_ResNet-50/101/152_): `conv5_x`의 마지막 $1×1$이 채널을 4배로 확장하므로 $2048$ 채널이 된다.

Lucid는 이 차이를 `block.expansion`으로 일반화한다. `_BasicBlock.expansion=1`, `_Bottleneck.expansion=4`이므로, GAP 이후 flatten된 벡터의 차원은 항상 `512 * block.expansion`이 된다. 즉, 같은 `ResNet.forward`를 쓰더라도 block 종류에 따라 자동으로 분류기 입력 차원이 맞춰지며, 이는 논문이 Table 1에서 정의한 인터페이스(마지막 stage 출력 채널)와 정확히 대응된다.

### 5️⃣ Residual Block (2-layer): `_BasicBlock`
논문 Fig. 2의 기본 2-layer residual block은 `_BasicBlock`이 담당한다. 두 개의 3×3 conv를 거친 뒤, identity 또는 downsample된 shortcut을 더하고 ReLU를 적용한다.

```python
class _BasicBlock(nn.Module):
    expansion: ClassVar[int] = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        basic_conv_args = dict(kernel_size=3, padding=1, bias=False)

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, stride=stride, **basic_conv_args
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, stride=1, **basic_conv_args)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out
```

`out += identity`가 $(1)$, $(2)$의 add를 구현한다. 또한 downsample이 존재할 때만 identity를 projection으로 바꾸는 방식은 필요할 때만 $(2)$를 사용한다는 논문 메시지와 대응된다.

논문 Fig. 2의 관점에서 보면, `_BasicBlock`에서 `conv1+bn+relu`는 residual branch의 첫 층, `conv2+bn`은 residual branch의 마지막 선형 출력에 해당한다. 그 뒤 `out += identity`가 shortcut add이고, 마지막 `relu2`가 add 이후 비선형이다. 즉, Lucid의 `_BasicBlock`은 논문이 제시한 블록 구성 요소를 그대로 코드로 옮긴 형태라고 볼 수 있다.

#### Convolution에서 `bias=False`가 등장하는 이유
`_BasicBlock`의 두 conv는 모두 `bias=False`로 만들어진다(`basic_conv_args`). 이는 **곧바로 뒤에 BN이 붙기 때문**이다. BN은 학습 가능한 shift/scale(affine)을 통해 channel-wise bias 역할을 포함할 수 있으므로, conv bias를 두면 표현이 중복되고(파라미터 낭비), 수치적으로도 불필요한 자유도가 추가된다.

논문도 BN을 각 convolution 뒤, activation 앞에 둔다고 명시하고, Fig. 2/5의 블록 구성에서도 BN이 사실상 기본 전제로 깔려 있다. 따라서 Lucid에서 conv bias를 끄는 것은 논문이 전제하는 구현 관행과도 잘 맞는다.

#### Downsample이 `out`이 아니라 `x`에 적용되는 이유
코드를 보면 `identity = x`를 먼저 저장해 두고, downsample이 필요할 때는 `identity = self.downsample(x)`를 수행한다. 즉, projection shortcut은 residual branch 출력이 아니라, shortcut 경로인 입력 $x$에만 적용된다.

이는 $(2)$의 의미를 그대로 반영한다.

- $(1)$: $y = F(x) + x$  
- $(2)$: $y = F(x) + W_s x$

여기서 $W_s$가 곧 downsample이며, 어디까지나 shortcut 경로의 차원을 맞추는 용도다. residual branch의 변환 $F(x)$는 블록 내부의 conv/BN에 의해 이미 정의되어 있으므로, shortcut에서만 차원을 조정하는 것이 논문의 설계 철학(기본은 identity, 필요한 경우에만 projection)과 일치한다.

#### Add 이후 ReLU의 의미
`_BasicBlock`은 `out += identity` 이후에 `relu2`를 적용한다. 이는 논문이 설명하는 형태(더하기 이후에 비선형 적용)와 대응된다. add 이후 activation을 두면, 블록 출력이 다시 비선형 변환을 거치면서 다음 블록의 입력 분포가 일정한 범위로 제한되고, residual 누적 구조가 너무 선형적으로만 흐르는 것을 막아준다.

### 6️⃣ Bottleneck Block (3-layer): `_Bottleneck`
논문 Fig. 5의 bottleneck block(`1×1-3×3-1×1`)은 `_Bottleneck`으로 구현된다. 여기서 `expansion=4`는 마지막 1×1에서 채널을 4배로 확장하는 점을 나타낸다.

```python
class _Bottleneck(nn.Module):
    expansion: ClassVar[int] = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        cardinality: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        se: bool = False,
        se_args: dict = {},
    ) -> None:
        super().__init__()
        width = int(math.floor(out_channels * (base_width / 64)) * cardinality)

        self.conv1 = nn.ConvBNReLU2d(
            in_channels, width, kernel_size=1, stride=1, conv_bias=False
        )
        self.conv2 = nn.ConvBNReLU2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=dilation,
            groups=cardinality,
            conv_bias=False,
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                width,
                out_channels * self.expansion,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

        self.se = nn.SEModule(out_channels * self.expansion, **se_args) if se else None
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
```

논문 ResNet은 SE나 cardinality 같은 확장을 포함하지 않지만, Lucid는 같은 ResNet 골격에서 다양한 변형(ResNeXt/SE-ResNet 등)을 실험할 수 있도록 bottleneck을 일반화해 두었다. 그럼에도 마지막에 더하기(add)로 residual을 구성한다는 본질은 동일하게 유지된다.

논문 Fig. 5의 bottleneck을 구현 관점에서 다시 쓰면, 첫 1×1은 채널을 줄여 3×3의 비용을 낮추고, 마지막 1×1은 채널을 다시 확장해 다음 블록과의 인터페이스(특히 shortcut 덧셈)를 맞춘다. 이 과정에서 shortcut 경로는 가능한 한 identity로 유지되고, 차원이 바뀌는 블록에서만 1×1 projection이 들어간다. Lucid `_Bottleneck`도 `downsample` 조건을 통해 이 규칙을 그대로 따른다.

### 7️⃣ 모델 등록 함수들
마지막으로 Lucid는 논문에서 다루는 깊이의 ResNet을 factory 함수로 제공한다. 논문 Table 1에서의 반복 수를 `layers`로 표현하고, 어떤 블록을 쓰는지로 모델을 구분한다.

```python
@register_model
def resnet_18(num_classes: int = 1000, **kwargs) -> ResNet:
    layers = [2, 2, 2, 2]
    return ResNet(_BasicBlock, layers, num_classes, **kwargs)


@register_model
def resnet_34(num_classes: int = 1000, **kwargs) -> ResNet:
    layers = [3, 4, 6, 3]
    return ResNet(_BasicBlock, layers, num_classes, **kwargs)


@register_model
def resnet_50(num_classes: int = 1000, **kwargs) -> ResNet:
    layers = [3, 4, 6, 3]
    return ResNet(_Bottleneck, layers, num_classes, **kwargs)


@register_model
def resnet_101(num_classes: int = 1000, **kwargs) -> ResNet:
    layers = [3, 4, 23, 3]
    return ResNet(_Bottleneck, layers, num_classes, **kwargs)


@register_model
def resnet_152(num_classes: int = 1000, **kwargs) -> ResNet:
    layers = [3, 8, 36, 3]
    return ResNet(_Bottleneck, layers, num_classes, **kwargs)
```

`resnet_18/34`는 2-layer basic block을, `resnet_50/101/152`는 bottleneck block을 사용한다는 점이 논문 Fig. 5와 정확히 대응된다. 또한 반복 수(`layers`)가 ResNet-34와 ResNet-50에서 동일하더라도, block이 basic인지 bottleneck인지에 따라 실질 깊이와 계산량이 달라진다는 점을 코드가 잘 보여준다.

즉 Lucid에서 논문 ResNet을 생성하는 방식은, 논문 Table 1의 반복 수를 그대로 `layers`로 옮기고, Fig. 5의 블록 종류를 `_BasicBlock/_Bottleneck`으로 선택해 조립하는 과정이라고 정리할 수 있다.

---

## ✅ 정리
**ResNet** 논문은 깊이가 성능을 올려야 한다는 직관이 실제 학습에서는 degradation problem으로 깨질 수 있음을 보여주고, 이를 residual learning이라는 간단하면서도 강력한 구조로 해결한다. $(1)$의 $y=x+F(x)$는 네트워크를 완전히 새로운 표현을 만드는 층들의 적층이 아니라 입력 표현에 대한 작은 수정의 누적으로 재해석하게 만들고, identity shortcut이 제공하는 최적화 경로 덕분에 매우 깊은 모델에서도 훈련 오차가 안정적으로 내려가며 일반화 성능까지 좋아진다. ImageNet에서 _ResNet-50/101/152_ 가 SOTA를 달성하고, 그 표현이 detection/localization까지 확장되어 큰 개선을 주는 결과는 residual이 단순한 트릭이 아니라 표현 학습의 기본 구조로 자리잡을 수 있음을 보여준다.

이 논문을 읽고 나면, 깊은 네트워크 설계에서 **우선순위가 바뀐다는 점**이 가장 크게 남는다. 더 깊게 만들수록 학습이 무너지는 현상을 데이터/정규화/초기화 탓으로만 돌리기보다, 아키텍처의 파라미터화가 **solver가 탐색하기 쉬운 형태**인지(특히 identity 근처에서 출발 가능한지)를 먼저 점검하게 된다. 또한 ImageNet 분류 성능을 넘어서 COCO/PASCAL/ILSVRC의 detection·localization 결과까지 연결해 보여줌으로써, backbone 개선이 downstream 파이프라인 전체의 성능을 끌어올리는 핵심 축임을 명확히 한다.

마지막으로 이 리뷰의 관점에서 ResNet을 한 번 더 압축하면 다음 포인트로 정리할 수 있다.

- 문제 정의: 깊어질수록 훈련 오차가 나빠지는 degradation은 overfitting이 아니라 optimization 문제다.
- 핵심 수식: $y=x+F(x)$, 필요 시 $(2)$ projection으로 차원을 맞춘다.
- 실험 논증: plain vs ResNet 비교(특히 18/34)로 최적화 난점 해소를 직접 보여준다.
- 설계 확장: bottleneck으로 깊이를 늘리되 연산량을 통제한다(Table 1, Fig. 5).
- 범용성: 분류 성능 향상이 detection/localization으로 이어진다는 것을 표로 입증한다(Table 7~13).


#### 📄 출처
He, Kaiming, et al. "Deep Residual Learning for Image Recognition." *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, arXiv:1512.03385.
