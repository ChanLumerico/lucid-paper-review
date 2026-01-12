# [DenseNet] Densely Connected Convolutional Networks

## 🤗 소개
DenseNet은 매우 깊은 CNN을 학습할 때 반복해서 등장하는 문제(gradient 소실, 특징 전달의 단절, 파라미터 비효율)를 **연결 방식(connectivity)** 자체로 해결하려는 모델이다. 핵심 아이디어는 단순하다. 같은 feature-map 크기를 공유하는 구간 안에서, 각 레이어가 바로 직전 레이어의 출력만 보는 것이 아니라 **이전의 모든 레이어 출력(feature-map)을 입력으로 받도록** 만든다. 그리고 ResNet처럼 더하기로 합치는 것이 아니라, **채널 축 concat으로 누적**한다.

이 연결 패턴은 레이어 간 정보/그래디언트 경로를 매우 촘촘하게 만들고, 같은 특징을 여러 번 다시 학습하지 않아도 되게 만들어 **파라미터 효율**을 높인다. 논문은 이를 **Dense Convolutional Network(DenseNet)**이라 부르고, **Dense Block**과 **Transition Layer**라는 구성 요소로 네트워크를 쌓는다. 또한 **Bottleneck(1×1)**과 **Compression(Transition에서 채널 축소)**을 결합한 **DenseNet-BC** 변형을 통해, 성능과 연산량/파라미터 사이의 균형이 더 좋아진다고 주장한다.

이 리뷰는 논문 전개를 따라 **Dense Connectivity 수식**과 **ResNet 대비 차이**, **Dense Block/Transition Layer 구조**, **Growth Rate $k$**, **Bottleneck**, **Compression $\theta$** 같은 설계 파라미터가 실제로 어떤 비용 구조를 만드는지, **CIFAR/SVHN/ImageNet 실험 결과**와 **효율 분석**을 상세히 해석한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/7420e3c1-06bd-430f-89c7-708808431fc3/image.png" width="50%">
</p>

---

## 1️⃣ 논문 배경

### 🔹 깊은 CNN에서 반복되는 문제
CNN이 깊어질수록 성능이 좋아질 잠재력은 커지지만, 학습은 어려워진다. 논문은 깊은 네트워크에서 **입력 정보**와 **gradient**가 여러 비선형 변환을 거치며 약해지고, 결과적으로 학습이 불안정해지는 문제를 다시 강조한다. 이 문제를 해결하기 위한 공통된 방향은 **짧은 경로(short path)**를 더 만드는 것이다. 즉, 초반 레이어의 신호가 후반 레이어까지 더 직접적으로 전달될 수 있게 만드는 구조다.

ResNet과 Highway Network는 identity/게이트를 통한 bypass 경로를 추가해 학습을 안정화했다. Stochastic depth는 학습 중 일부 레이어를 드롭해 평균적인 경로 길이를 줄인다. FractalNet은 병렬 경로를 통해 다양한 길이의 경로를 제공한다. 논문은 이 다양한 접근들의 공통점을, 결국 깊은 네트워크에서 **초기-후기 레이어 사이의 연결을 짧게 만드는 것**으로 본다.

#### 깊이에 따른 학습 난이도의 수식적 관점
레이어를 여러 개 쌓으면, $x_0\to x_L$로 가는 변환은 많은 합성 함수의 연쇄가 된다. gradient 관점에서는 연쇄 법칙에 의해 **Jacobian**들이 곱해지는데, 이 곱이 1보다 작거나(또는 크거나) 불안정하게 변하면 **gradient 소실/폭주**가 나타난다. skip connection은 이 곱셈 사슬에 **짧은 우회 경로**를 추가해, gradient가 길게 곱해지지 않고도 앞 레이어로 전달될 수 있게 만든다.

DenseNet이 흥미로운 지점은 여기서 한 발 더 나아간다는 것이다. ResNet은 레이어마다 하나의 우회 경로(바로 이전 레이어로 돌아가는 identity)를 제공하지만, DenseNet은 같은 해상도 구간에서는 레이어가 많아질수록 우회 경로의 수가 급격히 증가한다. 즉, 특정 레이어에서 loss까지 갈 수 있는 경로가 더 다양해지고, 그중 짧은 경로도 매우 많아진다.

#### DenseNet의 출발점: 짧은 경로의 극대화
논문은 이 공통점을 가장 단순한 형태로 밀어붙인다. 같은 feature-map 크기를 공유하는 구간에서는, 모든 레이어가 서로 직접 연결되면 정보 흐름은 최대화된다. 따라서 DenseNet은 레이어 $\ell$이 $x_{\ell-1}$만을 입력으로 받는 대신, $x_0, x_1, \dots, x_{\ell-1}$ 전체를 입력으로 받도록 만든다.

이 설계는 단지 학습 안정성만이 아니라 **특징 재사용(feature reuse)**이라는 관점에서도 의미가 있다. 기존 네트워크는 뒤 레이어로 갈수록 앞 레이어에서 만들어진 특징을 다시 변환하면서 유사한 특징을 중복 학습할 수 있다. DenseNet은 앞 레이어의 feature-map을 concat으로 계속 보존하므로, 뒤 레이어는 필요한 특징을 재사용하면서 새로운 특징을 조금씩 추가하는 형태로 동작할 수 있다.

#### 연결 수의 규모: L개의 연결에서 L(L+1)/2로
논문 abstract는 전통적인 $L$-layer CNN이 인접 레이어 사이의 $L$개 연결만 가진다고 보고, DenseNet은 같은 해상도 내에서 각 레이어가 이전 모든 레이어와 연결되므로 연결 수가 $\frac{L(L+1)}{2}$로 늘어난다고 강조한다. 이 숫자 자체가 중요한 이유는, DenseNet이 레이어 간 연결을 단지 몇 개 추가한 것이 아니라, **연결 패턴을 근본적으로 바꾼 설계**임을 보여주기 때문이다.

#### 상태(State) 관점에서의 DenseNet
논문은 전통적인 feed-forward 네트워크를 layer-to-layer state 전달 알고리즘으로 해석한다. 각 레이어는 상태를 읽고 다음 상태를 쓴다. ResNet은 identity를 더해 상태 보존을 명시화한다. DenseNet은 더 급진적으로, 상태를 덮어쓰지 않고 **기존 상태를 그대로 유지한 채 새로운 정보만 추가**한다. 즉, 각 레이어는 전체 상태(이전 모든 feature-map)에 접근하고, 자신의 출력(feature-map)도 이후 모든 레이어에 그대로 전달한다.

### 🔸 Add 대비 Concat 결합: 정보 혼합 방식의 전환
DenseNet은 ResNet과 달리, 특징을 더하기로 합치지 않는다. ResNet은 $H_{\ell}(x_{\ell-1}) + x_{\ell-1}$ 형태로 합치므로, 두 경로의 정보가 섞여 하나의 텐서가 된다. DenseNet은 **concat으로 모아** 다음 레이어의 입력을 만든다. 이 차이는 단순한 취향이 아니라, 정보 흐름과 표현의 구조를 바꾸는 선택이다.

#### Add와 Concat의 근본적 차이: 보존 Vs 혼합
Add는 두 텐서를 같은 공간/채널 구조에서 **동일한 좌표끼리 더해** 하나의 텐서를 만든다. 이 경우 다음 레이어는 add로 섞인 결과만 볼 수 있고, 어떤 정보가 identity에서 왔고 어떤 정보가 변환에서 왔는지 분해해 보기는 어렵다.

Concat은 채널 축을 늘려 두 텐서를 나란히 둔다. 따라서 다음 레이어는 이전 특징을 그대로 확인할 수 있고, 필요하면 1×1 conv 같은 연산으로 채널 혼합을 통해 조합을 만들 수 있다. DenseNet이 주장하는 feature reuse는 사실상 이 분해된 입력 구조에서 더 자연스럽게 나타난다.

#### Concat 효과
Concat은 서로 다른 레이어에서 나온 특징을 채널 축으로 나란히 쌓는다. 즉, 뒤 레이어는 입력을 받을 때 이미 분해된 특징 묶음을 받는다. 이는 add처럼 서로 섞여버린 representation을 다시 분해할 필요가 없다는 의미도 된다. 논문은 이 점이 feature reuse를 촉진한다고 주장한다.

각 레이어는 loss까지 가는 경로가 많다. 특히 후반 레이어로 갈수록 앞 레이어 출력이 계속 입력으로 들어가므로, 앞 레이어의 출력은 여러 지점에서 활용된다. 논문은 이를 deep supervision과 유사한 효과로 설명한다. 즉, 앞 레이어가 만든 특징이 뒤에서 바로 쓰이기 때문에, 학습 신호가 더 직접적으로 전달될 수 있다.

#### Dense Connectivity의 추가 효과: 규제(Regularization)
논문은 dense connection이 일종의 규제 효과를 가져 과적합을 줄일 수 있다고 관찰한다. 특히 데이터가 작거나 증강이 없는 setting에서 DenseNet이 큰 개선을 보인다는 분석(Section 4.3의 overfitting 논의)은, Dense connectivity가 단지 최적화 문제만 완화하는 것이 아니라, 학습된 표현이 더 일반화되도록 유도할 수 있다는 방향의 주장으로 이어진다.

---

## 2️⃣ 관련 연구

### 🔹 Skip Connection 계열과의 관계: 경로를 짧게 만드는 공통 목표
논문은 DenseNet을 ResNet/Highway/Stochastic depth/FractalNet 같은 계열과 같은 문제의식에서 출발한 구조로 위치시킨다. 이들은 모두 깊은 네트워크의 학습 문제를 경로 길이를 줄이는 방식으로 해결한다. DenseNet이 달라지는 지점은, 그 경로 단축을 부분적 bypass가 아니라 **완전한 dense connectivity**로 구현했다는 것이다.

ResNet의 identity add는 정보가 직접 지나갈 수 있는 경로를 제공하지만, representation은 **더하기로 섞인다**. DenseNet은 이 섞임을 피하고 **concat**을 선택해 feature-map을 보존한다. 논문은 이 차이가 **표현 효율**과 **학습 안정성** 모두에 영향을 준다고 주장한다.

#### Highway/ResNet/Stochastic Depth 계열의 공통 관측: 깊이에 따른 우회 경로 중요성
논문은 Highway Network가 수백 레이어를 end-to-end로 학습할 수 있었던 이유를 bypass 경로와 gating unit에서 찾는다. 그 다음 ResNet은 gating 없이도 순수 identity mapping만으로 학습이 가능하다는 것을 보여주면서, 우회 경로가 본질적인 요소임을 강화한다.

그리고 stochastic depth는 한 걸음 더 나아가, 학습 중 일부 레이어를 랜덤하게 드롭해 평균 경로 길이를 줄이고(즉, 더 짧은 경로를 자주 만들고), 그 결과 1000+ 레이어의 residual 네트워크도 학습 가능하다는 사례를 제시한다. 논문은 여기서 중요한 관찰을 끌어온다.

- 매우 깊은 네트워크에는 구조적 redundancy가 존재할 수 있고  
- 모든 레이어가 항상 필요한 것은 아닐 수도 있으며  
- 그렇다면 layer 간 연결을 더 직접적으로 만들면 학습이 쉬워질 수 있다  

DenseNet은 이 관찰을 deterministic하게 밀어붙여, 같은 해상도 구간에서 가능한 한 많은 직접 연결을 만들어 정보 흐름을 최대로 하자는 방향으로 이어진다.

#### Multi-Level Feature 활용 연구들과의 연결
세그멘테이션/디텍션 등에서 서로 다른 깊이의 특징을 함께 쓰는 연구들이 존재해 왔다. DenseNet은 이러한 multi-level feature 활용을 네트워크의 기본 연결 방식으로 끌어올린 형태로 볼 수 있다. 즉, 특정 task head에서만 다중 스케일 특징을 결합하는 것이 아니라, backbone 내부에서 지속적으로 결합한다.

#### Cascade(연쇄) 구조 연구의 계보: 완전연결 MLP에서 CNN으로
논문은 DenseNet의 레이아웃과 비슷한 연쇄(cascade) 구조가 1980년대 신경망 문헌에서도 연구된 적이 있음을 짚는다. 다만 당시의 초점은 convolution이 아니라 fully-connected MLP에서 레이어를 층별로 학습시키는 방식에 가까웠고, 네트워크 규모도 현대 CNN과 비교하기 어려울 정도로 작았다. 이후 배치 기반 gradient descent로 학습하는 fully-connected cascade 네트워크가 제안되었지만, 논문은 이러한 방식이 결국 수백 개 수준의 파라미터로만 확장 가능했다고 정리한다.

DenseNet이 이 계보를 다시 끌어온 방식은 두 가지로 요약된다.

1. Convolutional feature-map을 유지한 채, 레이어 간 정보를 **concat으로 안전하게 보존**한다.  
2. 학습을 레이어별로 쪼개지 않고, end-to-end로 학습 가능한 형태로 설계를 고정한다.  

즉, DenseNet은 과거의 cascade 아이디어를 현대 CNN의 표준 학습/정규화/구현 관점에서 다시 정리해, 실제 대규모 비전 데이터셋(ImageNet)까지 스케일시키는 쪽에 초점을 둔다.

#### Cross-Layer Connection의 이론적 틀과의 병행
Related Work는 DenseNet과 유사한 cross-layer connection을 이론적으로 다루는 병렬 연구가 있었음을 언급한다. 이 지점은 논문이 실험만으로 설득하려는 것이 아니라, 연결 패턴 자체가 어떤 표현/최적화 성질을 가질 수 있는지에 대한 이론적 관점이 이미 논의되고 있었음을 보여준다.

### 🔸 Inception 및 폭 확장과의 대비: Concat의 공통점과 목적 차이
Inception은 서로 다른 커널 크기의 출력을 concat해 폭을 넓히는 방식으로 표현력을 높인다. Wide ResNet도 폭 확장을 통해 성능을 끌어올린다. DenseNet도 concat을 사용하지만, 목적은 단순 폭 확장이 아니라 **레이어 간 특징 재사용(Feature Reuse)**이다.

#### 좁은 레이어 강조의 배경
DenseNet에서 각 레이어는 전체 채널을 한 번에 크게 만들지 않고, 상대적으로 작은 수의 채널만 추가한다. 이를 논문은 growth rate $k$로 파라미터화한다. 즉, 폭을 키우는 대신, 많은 레이어가 **작은 업데이트를 누적**하는 방식으로 전체 표현을 만든다. 이 설계가 파라미터 효율의 근거가 된다.

#### Inception의 Concat과 DenseNet의 Concat 비교 필요성
Inception module의 concat은 주로 같은 레이어 깊이에서 서로 다른 receptive field(커널 크기/경로)를 병렬로 만들어 폭을 늘리는 결합이다. 반면 DenseNet의 concat은 **깊이축으로 축적되는 상태의 누적**에 가깝다. 즉, DenseNet에서 concat의 핵심은

- 서로 다른 크기의 필터 출력을 섞어 다양성을 늘린다기보다  
- 이미 만들어진 feature-map을 버리지 않고 다음 레이어 입력으로 계속 제공한다는 점  

에 있다. 그래서 두 모델 모두 concat을 쓰지만, DenseNet이 주장하는 효율은 폭 확장보다는 feature reuse에서 나온다는 흐름으로 읽히게 된다.

#### Network In Network, DSN, Ladder, DFN 등과의 연결: 정보 흐름과 지도 신호
논문은 DenseNet이 직접적으로 다루는 축(연결 패턴) 외에도, 당시 경쟁력 있는 결과를 만든 다른 설계들을 Related Work에서 함께 열거한다.

- Network in Network는 conv 필터 내부에 미니 MLP를 넣어 더 복잡한 특징을 추출한다.  
- Deeply Supervised Network(DSN)는 중간 레이어마다 auxiliary classifier로 직접 supervision을 걸어 초기 레이어가 받는 gradient를 강화한다.  
- Ladder network는 오토인코더에 lateral connection을 추가해 반지도 학습에서 강한 성능을 낸다.  
- Deeply-Fused Nets(DFN)는 서로 다른 네트워크의 중간 레이어를 결합해 정보 흐름을 개선하려는 방향을 제안한다.  

DenseNet은 이들을 직접 결합한 모델은 아니지만, dense connectivity로 인해 중간 레이어가 loss에 더 짧은 경로로 연결되는 효과를 만들면서, DSN이 노린 deep supervision과 유사한 이점을 구조적으로 얻을 수 있다고 해석한다(단, auxiliary loss를 여러 개 두지 않고도).

---

## 3️⃣ DenseNet 설계

### 🔹 표기와 ResNet 대비 수식=
논문은 레이어 $\ell$의 비선형 변환을 $H_{\ell}(\cdot)$로 두고, 출력 feature-map을 $x_{\ell}$로 표기한다. $H_{\ell}$는 BN, ReLU, Conv, Pool 같은 연산의 합성일 수 있다.

전통적 feed-forward는

$$
x_{\ell} = H_{\ell}(x_{\ell-1})
$$

로 연결된다. ResNet은 여기에 identity skip을 더해

$$
x_{\ell} = H_{\ell}(x_{\ell-1}) + x_{\ell-1}\tag{1}
$$

을 얻는다. 이때 장점은 gradient가 identity 경로로 직접 흐를 수 있다는 점이다. 하지만 논문은 add로 합치는 방식이 정보 흐름에 제약이 될 수 있다고 지적한다. 즉, 정보가 섞이면서 일부가 희석되거나, 다음 레이어가 특정 특징을 직접 재사용하기 어렵게 될 수 있다.

#### $H_{\ell}$ 구성: BN-ReLU-Conv 표준화
논문은 DenseNet의 레이어 변환 $H_{\ell}$가 단일 conv가 아니라 여러 연산의 합성일 수 있음을 강조한다. 특히 DenseNet 계열에서는 BN과 ReLU를 conv 앞에 두는 BN-ReLU-Conv 구성이 반복된다. 논문 Table 1 캡션에서도 각 conv 레이어가 BN-ReLU-Conv 시퀀스에 해당한다고 명시한다.

이 표준화는 두 가지 역할을 한다.

1. 깊은 네트워크에서 학습을 안정화한다(BN의 효과).  
2. Dense connectivity로 입력 채널 수가 계속 변하는 상황에서도, 각 레이어가 일정한 형태의 변환을 수행하도록 만든다.

#### Composite Function 표기의 의의
논문이 $H_{\ell}$를 단일 연산이 아니라 composite function이라고 부르는 이유는, DenseNet의 핵심이 연결 패턴이기 때문이다. 연결 패턴을 설명할 때 $H_{\ell}$ 내부의 세부 구현에 과도하게 매몰되면 핵심을 놓치기 쉽다. 따라서 논문은 BN/ReLU/Conv 같은 표준 블록을 $H_{\ell}$로 추상화하고, Eq.(1)과 Eq.(2)가 만드는 정보 흐름 차이를 전면에 둔다.

#### ResNet의 Add 결합이 갖는 제약의 정교화
Eq.(1)에서 $H_{\ell}(x_{\ell-1})$와 $x_{\ell-1}$는 같은 shape을 갖고 더해진다. 이때 identity 경로는 매우 유용하지만, add 이후의 텐서는 두 정보를 분리할 수 없는 혼합 상태가 된다. 다음 레이어는 이 혼합 상태에서 필요한 정보를 다시 추출해야 한다.

DenseNet은 이 문제를 concat으로 바꾸어, 필요한 특징이 명시적으로 입력으로 제공되도록 만든다. 즉, 학습은 다음 레이어가 특징을 다시 찾는 작업을 덜 하게 되고, 대신 새로운 특징을 추가하는 작업에 집중할 수 있다는 것이 논문 주장이다.

#### DenseNet의 변화 지점: 경로 수와 결합 연산
ResNet도 경로를 늘리지만, 결합은 add다. DenseNet은 경로를 더 극단적으로 늘리면서 결합을 concat으로 바꾼다. 따라서 DenseNet을 단지 skip connection이 많은 네트워크로만 보면 핵심을 놓친다. DenseNet의 본질은, **이전 레이어의 출력을 상태로 보존하면서 누적**한다는 점이다.

### 🔸 Dense Connectivity: 모든 이전 Feature-Map을 입력으로
DenseNet의 핵심 수식은 다음이다.

$$
x_{\ell} = H_{\ell}([x_0, x_1, \dots, x_{\ell-1}])\tag{2}
$$

여기서 $[\cdot]$는 channel-wise concatenation이다. 즉, 레이어 $\ell$의 입력은 이전 모든 레이어의 출력이 채널 축으로 쌓인 텐서다. 그리고 레이어 $\ell$의 출력 $x_{\ell}$도 이후 레이어들의 입력에 포함된다.

#### Concat 입력 Shape: 레이어 수에 따른 채널 증가
Eq.(2)를 실제 텐서 shape 관점에서 보면, dense block 안에서 레이어가 진행될수록 입력 채널 수가 증가한다. 첫 레이어는 대략 $C_0$ 채널을 입력으로 받고, 두 번째는 $C_0+k$, 세 번째는 $C_0+2k$처럼 증가한다. 따라서 같은 3×3 conv라도 레이어가 깊어질수록 입력 채널이 커져 비용이 늘 수 있다.

DenseNet이 bottleneck과 compression을 함께 제안하는 이유가 여기서 자연스럽게 나온다. 첫째, bottleneck은 3×3에 들어가는 채널을 제한한다. 둘째, compression은 block 사이에서 채널을 줄여 다음 block에서의 채널 성장을 완화한다. oncat 구조에서는 이전 특징이 사라지지 않는다. 따라서 $H_{\ell}$는 기존 특징을 다시 만들어내는 데 용량을 낭비할 필요가 없다. 논문은 이로 인해 레이어가 더 좁아도 된다고 주장하며, 그 좁음의 정도를 growth rate $k$로 정의한다.

#### 정보/그래디언트 흐름: 많은 짧은 경로의 집합
Eq.(2)는 손으로 그리면 연결이 매우 많아 보이지만, 관점은 간단하다. 어떤 레이어의 출력은 이후 모든 레이어로 바로 입력된다. 따라서 loss에서 앞 레이어로 가는 경로 길이가 매우 짧은 경로들이 많이 생긴다. 논문은 이를 vanishing-gradient 완화의 핵심 근거로 제시한다.

#### 암묵적 Deep Supervision의 수학적 재정리
각 레이어 $x_{\ell}$은 이후 여러 $H_j$의 입력에 반복해서 등장한다. 즉, loss $\mathcal{L}$에 대한 gradient $\partial \mathcal{L}/\partial x_{\ell}$는 단일 경로가 아니라 여러 경로의 합으로 전달된다. DenseNet은 구조적으로 이런 다중 경로를 많이 만들기 때문에, 초기 레이어가 받는 학습 신호가 약해지지 않는 경향을 기대할 수 있고, 논문은 이를 deep supervision과 유사하다고 설명한다.

#### Eq.(2)의 구현 관점 해석: 다중 입력의 단일 텐서화
논문은 Eq.(2)에서 $H_{\ell}$가 여러 입력을 받는 것처럼 보이지만, 실제 구현에서는 이를 채널 축 concat으로 **단일 텐서**로 합쳐 $H_{\ell}$에 넣는다고 설명한다. 즉, 구현 관점에서 **Dense Connectivity**의 본질은

1. 과거 feature-map을 유지한다(**메모리**)  
2. 이를 채널 축으로 합친다(**concat 비용**)  
3. 합쳐진 텐서에 대해 BN/ReLU/Conv를 수행한다(**연산 비용**)  

의 조합이다. 그래서 DenseNet은 파라미터 수는 줄일 수 있어도, activation 메모리/데이터 이동 비용까지 포함한 실제 구현 효율은 별도의 고려가 필요하며, 논문도 메모리 효율 구현을 별도 리포트로 안내한다.

### 🔹 Dense Block과 Transition Layer
Dense connectivity를 전체 네트워크에 무제한으로 적용할 수는 없다. spatial resolution이 바뀌면 concat이 불가능하기 때문이다(텐서 크기가 다르다). 그래서 논문은 네트워크를 여러 개의 dense block으로 나누고, block 사이를 transition layer로 연결한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/c1be6088-1076-4fbc-aa70-9652d3bb358d/image.png" width="80%">
</p>

#### Dense Block의 정의
dense block은 동일한 $H\times W$를 공유하는 구간에서 Eq.(2)를 적용한 연속 레이어 묶음이다. block 안에서는 각 레이어가 이전 모든 레이어 출력을 concat으로 입력받고, 출력은 채널 축으로 누적된다.

#### Dense Block Forward의 의사코드
Dense block을 구현 관점에서 보면, 각 레이어는 다음 절차를 따른다.

1. 현재까지 누적된 feature-map을 입력으로 받는다.  
2. 새 feature-map을 만든다.  
3. 그 결과를 다시 누적한다.  

이를 의사코드로 쓰면 다음과 같다.

```text
Algorithm: Dense Block forward
Input: x0 (C0 x H x W)
for l in 1..m:
  xl = Hl(concat(x0, x1, ..., x_{l-1}))   # BN/ReLU/Conv 조합
  x_{l} is k new channels
return concat(x0, x1, ..., xm)
```

이 의사코드는 Eq.(2)의 의미를 그대로 구현한 것이다. 그리고 마지막 반환이 concat인 것이 중요하다. Dense block은 특정 레이어의 출력만 반환하는 것이 아니라, block 내부의 모든 레이어 출력이 누적된 텐서를 반환한다.

#### Transition Layer의 역할
transition layer는 두 dense block 사이에서

1. 채널 수를 조절(보통 줄임)하고  
2. Spatial resolution을 줄인다(다운샘플링)

논문은 transition이 1×1 conv와 2×2 average pooling의 조합으로 이루어진다고 설명한다. 이 transition은 이후 DenseNet-C(Compression)에서 중요한 역할을 한다.

#### Transition에서 Avgpool 채택의 배경
논문은 transition의 pooling을 2×2 average pooling으로 두는 구성을 사용한다. 이 선택 자체가 논문의 핵심 기여는 아니지만, DenseNet의 연결 방식과 결합하면 직관적으로 이해할 포인트가 생긴다.

- Dense block은 이전 레이어 feature-map을 그대로 유지하고 계속 전달한다.  
- Transition은 그 누적된 특징들을 1×1 conv로 재조합한 뒤 해상도를 줄인다.  

이때 average pooling은 feature-map의 평균적인 응답을 보존하는 다운샘플링이므로, 강한 활성만 남기는 maxpool보다 더 부드럽게 정보를 전달하는 경향이 있다. DenseNet이 주장하는 feature reuse 관점에서는, transition이 이전 block의 다양한 특징을 다음 block으로 요약해 전달하는 역할을 한다고 볼 수 있고, avgpool은 그 요약의 기본 연산으로 자연스럽게 들어간다.

#### CIFAR 계열에서의 Feature-Map 크기 전개(논문 구현 설명)
논문은 ImageNet을 제외한 데이터셋(CIFAR/SVHN)에서는 dense block을 3개로 두고, 각 block의 feature-map 크기가 32×32 → 16×16 → 8×8로 변한다고 설명한다. 이는 transition layer의 2×2 average pooling으로 해상도를 절반으로 줄이기 때문이다. 이 전개는 DenseNet이 block 단위로 해상도를 바꾸는 전형적인 설계와 동일하다.

#### ImageNet에서의 4-Block 구조와 Stem
논문은 ImageNet에서는 224×224 입력에 대해 dense block을 4개로 두는 DenseNet-BC 구조를 사용한다고 말한다. 또한 초기에 7×7 stride 2 convolution과 pooling을 두는 stem을 사용하고, 이후 block을 쌓는다. 이 구성은 ResNet류의 ImageNet stem과 유사하며, DenseNet이 연결 패턴은 새롭지만 입력 처리/다운샘플링 흐름은 당시의 표준을 따르는 측면이 있음을 보여준다.

### 🔸 Growth Rate $k$: 채널 증가 양상
DenseNet에서 각 레이어는 많은 채널을 만들지 않고, $k$개의 feature-map만 새로 추가한다. 이 $k$를 growth rate라고 한다. 그래서 dense block 내부에서 레이어가 하나 추가될 때마다 채널 수는 $k$만큼 증가한다.

#### 채널 수의 전개
block 입력 채널이 $C_0$이고, 레이어가 $m$개 있다면, block을 통과한 후 채널은 대략

$$C_0 + m\cdot k$$

가 된다(정확히는 각 레이어 출력이 $k$채널이므로 누적 합). 이 전개는 DenseNet이 왜 좁은 레이어를 쓸 수 있는지와 연결된다. 뒤 레이어는 이미 누적된 많은 특징을 입력으로 받으므로, 새로 추가하는 $k$채널이 작아도 표현을 확장할 수 있다.

#### 논문 표기 정리: $k_0 + k\times(\ell-1)$ 입력 채널
논문은 dense block 내에서 $H_{\ell}$이 $k$개의 feature-map을 만든다고 가정하면, $\ell$번째 레이어의 입력 채널 수가 $k_0 + k\times(\ell-1)$로 증가한다고 설명한다. 여기서 $k_0$는 block 입력의 채널 수다.

이 표현을 리뷰 관점에서 한 번 더 풀어 쓰면 다음처럼 정리할 수 있다.

- 레이어가 깊어질수록 입력 채널이 선형으로 늘어난다.  
- 따라서 같은 3×3 conv라도 레이어가 진행될수록 입력 채널 증가로 비용이 커질 수 있다.  
- Bottleneck/transition compression은 이 선형 증가가 만드는 비용 폭증을 제어하기 위한 장치다.  

#### Growth Rate가 비용을 어떻게 결정하는가: 레이어 단위의 파라미터 근사
DenseNet-B의 한 레이어를 기준으로 보면,

- 1×1 conv: 입력 채널이 $C_{in}$이고 출력이 $4k$이면 파라미터는 대략 $C_{in}\cdot 4k$
- 3×3 conv: 입력 채널이 $4k$이고 출력이 $k$이면 파라미터는 대략 $4k\cdot k\cdot 3\cdot 3 = 36k^2$

즉, 3×3의 파라미터는 $k^2$에 의해 결정되고, 1×1의 파라미터는 현재 누적된 채널 수 $C_{in}$에 비례한다. DenseNet은 $k$를 비교적 작게 유지하고, bottleneck으로 3×3의 입력을 $4k$로 고정함으로써 비용을 제어한다는 그림이 나온다.

또한 레이어가 진행될수록 $C_{in}$이 증가하므로, 1×1의 비용이 점점 커질 수 있다. 이 때문에 compression이 중요해진다. transition에서 채널을 줄이면 이후 block에서의 $C_{in}$ 성장이 완화되고, 결과적으로 전체 비용이 제어된다.

#### Growth Rate가 작은데도 성능이 나오는 이유
논문은 실험에서 상대적으로 작은 $k$로도 충분한 성능을 얻을 수 있음을 보여준다. 이는 DenseNet이 특징을 중복 학습하지 않고 재사용하기 때문에, 폭을 크게 키우지 않아도 성능이 유지된다는 논지다.

### 🔹 Bottleneck(DenseNet-B): 1×1로 비용을 줄인 뒤 3×3
DenseNet은 concat으로 입력 채널이 계속 늘기 때문에, 3×3 conv를 그대로 쓰면 비용이 빠르게 커질 수 있다. 이를 완화하기 위해 논문은 3×3 conv 앞에 1×1 conv를 넣는 bottleneck 구조를 제안한다. 즉, 한 레이어의 변환 $H_{\ell}$를

- BN → ReLU → Conv(1×1)  
- BN → ReLU → Conv(3×3)

로 구성한다. 이때 1×1 conv는 중간 채널을 대략 $4k$로 만든다(논문에서 bottleneck factor 4를 사용).

#### Bottleneck의 직관
입력 채널이 크더라도, 3×3 conv를 수행하는 채널 수를 작은 중간 채널로 제한하면 비용이 줄어든다. 동시에 1×1 conv는 채널 간 선형 결합을 통해 필요한 특징 조합을 만들어 3×3 conv가 더 효율적으로 공간 변환을 할 수 있게 한다.

### 🔸 ㅊompression(DenseNet-C)과 DenseNet-BC: Transition 채널 축소
논문은 transition layer에서 채널을 줄이는 compression도 제안한다. dense block 출력 채널이 $m$이라면, transition의 1×1 conv가 만드는 출력 채널을

$$
\lfloor \theta m \rfloor\quad (0<\theta\le 1)
$$

로 두고, $\theta$를 compression factor라고 부른다. 논문 실험에서는 $\theta=0.5$를 사용한다.

#### DenseNet-BC의 의미
- DenseNet-B: bottleneck(1×1) 사용  
- DenseNet-C: compression($\theta<1$) 사용  
- DenseNet-BC: 둘 다 사용

DenseNet-BC는 DenseNet의 비용 구조를 더 공격적으로 줄이면서도 성능이 유지되는지 확인하기 위한 표준 변형으로 자리잡았다. 실제로 ImageNet 실험(Table 1)에서 논문은 DenseNet-BC 구조를 사용한다.

---

## 4️⃣ 모델 아키텍처와 복잡도

### 🔹 ImageNet용 DenseNet 구조
논문은 ImageNet에서 여러 깊이의 DenseNet을 비교한다. 대표 모델은 `DenseNet-121/169/201/264`이며, growth rate는 $k=32$로 둔다. 또한 bottleneck과 compression을 적용한 DenseNet-BC 설정이 기본이다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/79c5d040-e394-4137-a7de-56735ff3a7e4/image.png" width="80%">
</p>

#### DenseNet-121/169/201/264의 Block 구성
DenseNet 이름의 숫자는 전체 깊이(레이어 수)를 의미하고, 실제 구현에서는 dense block별 레이어 수로 구조가 정해진다.

| 모델 | 블록별 레이어 수 |
|---|---:|
| *DenseNet-121* | `(6, 12, 24, 16)` |
| *DenseNet-169* | `(6, 12, 32, 32)` |
| *DenseNet-201* | `(6, 12, 48, 32)` |
| *DenseNet-264* | `(6, 12, 64, 48)` |

이 표는 DenseNet의 설계가 사실상 block_config로 요약된다는 점을 보여준다. 즉, 같은 $k$에서 깊이를 늘리는 것은 dense block 내부 레이어 수를 늘리는 것으로 구현된다.

#### ImageNet에서의 Stem 채널 수: 2k 7×7 Conv
논문은 ImageNet DenseNet-BC에서 초기 convolution이 2k개의 7×7 필터(stride 2)로 구성된다고 설명한다. $k=32$면 초기 채널 수는 64가 된다. 이는 많은 구현에서 num_init_features=64로 잡는 것과 일치하며, ResNet류 stem과 유사한 채널 폭으로 시작한다는 뜻이기도 하다.

#### Table 1의 핵심 관찰
1. 같은 $k$에서 깊이를 늘리면 성능이 좋아질 수 있다.  
2. Dense block 내부는 concat으로 채널이 증가하지만, transition에서 채널을 줄여 비용을 제어한다.  
3. DenseNet은 ResNet에 비해 파라미터가 훨씬 크지 않으면서도 좋은 성능을 낸다는 것이 논문 주장이다.

Table 1 캡션은 각 conv 레이어가 BN-ReLU-Conv 시퀀스를 의미한다고 명시한다. 즉, DenseNet-BC에서 하나의 dense layer는 실질적으로

- `BN-ReLU-Conv`(1×1)  
- `BN-ReLU-Conv`(3×3)

의 조합이다. 여기에 transition에서는 BN-ReLU-Conv(1×1)과 average pooling이 들어간다. DenseNet의 깊이 숫자를 해석할 때는, 이런 구성 요소들이 포함된다는 점을 염두에 두는 것이 안전하다.

### 🔸 파라미터/연산 효율의 구조적 근거
DenseNet은 연결이 많아 보이지만, 각 레이어가 만드는 채널 수는 작다($k$). 그리고 이전 특징을 재사용하므로, 비슷한 수준의 표현력을 만들기 위해 폭을 크게 키울 필요가 줄어든다. 이 두 요소가 합쳐져, 파라미터 효율이 개선된다는 것이 논문의 큰 논지다.

#### Dense Connectivity가 만드는 비용과 이득의 균형
concat은 메모리 관점에서 부담이 될 수 있다. feature-map을 계속 유지해야 하기 때문이다. 논문도 구현에서의 메모리 비효율 가능성을 언급하고, 이후 메모리 효율 구현을 제안한다. 즉, DenseNet은 파라미터는 줄일 수 있지만, 활성화 메모리와 구현상의 최적화가 중요한 모델이다.

#### 파라미터 효율성의 성립 조건
DenseNet이 파라미터 효율적이라는 주장은, 단순히 연결을 많이 만들었기 때문에 자동으로 따라오는 성질이 아니다. 논문 흐름대로 정리하면 다음 전제가 함께 붙는다.

1. 각 레이어가 만드는 새로운 채널 수는 $k$로 제한된다.  
2. 뒤 레이어는 필요한 특징을 이전 레이어 출력에서 직접 가져올 수 있어, 같은 특징을 다시 만들 필요가 줄어든다.  
3. Bottleneck/transition compression을 통해, concat으로 증가하는 입력 채널이 연산 비용을 과도하게 키우지 않도록 제어한다.  

즉, Dense connectivity는 정보 흐름을 극대화하는 구조이고, $k$/bottleneck/$\theta$는 그 구조를 비용 측면에서 감당 가능하게 만드는 조절 장치라고 읽는 것이 안전하다.

#### ResNet의 폭/깊이 증가와 비교되는 지점
논문 Discussion은 ResNet과 DenseNet이 겉으로는 concat vs add라는 작은 차이처럼 보이지만, 실제로는 매우 다른 동작을 만든다고 강조한다. ResNet은 identity 경로로 정보 보존을 제공하지만, 각 레이어는 여전히 독립적인 feature-map을 만들어야 하고, add로 섞인 상태에서 중복이 생길 수 있다.

DenseNet은 앞 레이어 특징을 그대로 입력으로 제공하므로, 뒤 레이어가 앞 레이어 특징을 다시 학습할 필요가 줄어든다. 이 구조적 차이가 파라미터 효율의 핵심이라는 것이 논문 주장이다.

---

## 5️⃣ 모델 실험

### 🔹 CIFAR-10/100, SVHN 결과
논문은 CIFAR-10/100과 SVHN에서 DenseNet이 강력한 성능을 낸다고 보고한다. 특히 data augmentation 유무와 함께 비교하면서, DenseNet이 더 적은 파라미터로도 낮은 에러를 낼 수 있음을 강조한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/aabaec80-42e4-41ef-82c1-06e8ed74e907/image.png" width="70%">
</p>

#### 데이터셋과 전처리/증강 설정
논문은 CIFAR-10/100을 $32×32$ 자연 이미지로 두고, 50k train / 10k test 중 5k를 validation으로 홀드아웃한다고 설명한다. 표준 증강은 mirroring/shifting이며, 이를 C10+, C100+처럼 + 표기로 구분한다. 전처리는 채널 평균/표준편차로 정규화한다.

SVHN은 $32×32$ 숫자 이미지로, 추가 학습 데이터까지 포함해 사용하고, 일반적으로 data augmentation 없이 학습한다고 설명한다. 또한 픽셀 값을 255로 나눠 $[0,1]$ 범위로 정규화한다는 설정을 명시한다.

ImageNet은 1.2M train, 50k val, 1k classes이며, 학습 증강은 ResNet류와 동일한 표준 증강을 사용하고, 테스트는 single-crop 또는 10-crop $224×224$로 평가한다고 정리한다.

#### 학습 설정: LR 스케줄과 정규화
논문은 모든 모델을 SGD로 학습시키며, CIFAR는 batch size $64$로 $300$ epochs, SVHN은 $40$ epochs를 사용한다고 명시한다. 초기 lr은 $0.1$이고, 총 epoch의 50%와 75% 지점에서 10배씩 감소시킨다. ImageNet은 batch size $256$으로 $90$ epochs 학습하며, epoch $30$과 $60$에서 lr을 10배 감소시킨다.

또한 weight decay는 $1e-4$, Nesterov momentum은 $0.9$를 따른다고 설명한다. 증강이 없는 C10/C100/SVHN 설정에서는 각 conv(첫 번째 제외) 뒤에 dropout을 넣고 dropout rate $0.2$를 사용했다고 적는다. 이 디테일은 DenseNet이 증강이 없는 setting에서도 큰 개선을 보이는 근거를 해석할 때 중요한 배경이다.

#### Table 2에서의 핵심 메시지
1. 동일하거나 더 적은 파라미터로 ResNet 계열보다 낮은 에러를 낼 수 있다.  
2. Data augmentation이 없는 setting에서도 DenseNet의 이점이 크게 나타날 수 있다(논문은 dropout을 사용해 학습했다고 서술한다).  
3. $k$를 키우면 성능이 좋아질 수 있지만, 비용과의 균형이 필요하다.

#### SVHN에서의 깊이 증가 효과 한계
논문은 SVHN에서 DenseNet-BC의 250-layer 설정이 100-layer 설정보다 더 개선되지 않는 현상을 언급하고, 그 이유를 task 난이도/과적합 가능성으로 설명한다. SVHN은 상대적으로 쉬운 분류 문제일 수 있고, 모델이 지나치게 깊어지면 훈련 데이터에 과도하게 맞추는 방향으로 가기 쉽다는 것이다.

이 관찰은 DenseNet이 항상 깊어질수록 좋아진다는 단순한 결론을 피하게 만든다. Dense connectivity가 최적화를 돕는 것은 맞지만, 데이터셋이 주는 정보량과 정규화 조건(dropout 포함)에 따라 표현력이 과해질 수도 있고, 그래서 depth와 k, 그리고 BC의 조합은 여전히 문제/데이터에 맞춰 선택해야 한다는 메시지가 된다.

#### 논문이 직접 강조하는 최고 성능 설정: DenseNet-BC (L=190, k=40)
논문은 Table 2의 하단 결과를 대표 트렌드로 뽑는다. DenseNet-BC의 매우 깊은 설정($L=190, k=40$)이 CIFAR 계열에서 SOTA를 달성했다고 요약하며, 특히 다음 숫자를 직접 언급한다.

- `C10+` error **3.46%**  
- `C100+` error **17.18%**

즉, Dense connectivity가 단지 학습을 안정화하는 수준이 아니라, 충분한 깊이/적절한 bottleneck $+$ compression과 결합될 때 강력한 표현력을 제공할 수 있다는 논지다.

#### 증강이 없는 Setting에서의 상대 개선
논문은 C10/C100(증강 없음)에서 DenseNet의 개선이 특히 두드러진다고 강조한다. 본문에 제시된 수치 예시는 다음과 같다.

- `C10`: **7.33% → 5.19%** (상대 약 29% 감소)  
- `C100`: **28.20% → 19.64%** (상대 약 30% 감소)

이 관찰은 Dense connectivity가 파라미터 효율을 높이고, 과적합을 덜 유발하는 방향으로 작동할 수 있다는 논지로 이어진다. 물론 완전히 과적합이 사라지는 것은 아니며, 논문은 특정 설정에서 $k$를 늘렸을 때 오히려 에러가 소폭 증가하는 사례도 언급한다. 이때 bottleneck과 compression(DenseNet-BC)이 이를 완화하는 데 도움이 된다고 해석한다.

#### 모델 용량(Capacity) 경향: L 및 k 증가 효과
논문은 Table 2를 해석하면서, compression/bottleneck이 없는 기본 DenseNet에서는 깊이 $L$과 growth rate $k$가 늘어날수록 성능이 좋아지는 경향을 관찰했다고 말한다. 이는 단순히 최적화가 쉬워졌다는 것뿐 아니라, DenseNet이 모델 크기가 커질 때 그 표현력을 실제로 활용할 수 있음을 의미한다.

본문에 제시된 대표 예시는 C10+에서의 변화다.

- 파라미터 1.0M 수준에서 error **5.24%**  
- 파라미터 7.0M 수준에서 error **4.10%**  
- 파라미터 27.2M 수준에서 error **3.74%**

즉, DenseNet은 모델을 키웠을 때 성능이 일관되게 개선되는 경향을 보이며, 깊은 residual 네트워크에서 보고된 최적화 난이도 문제나 성능 열화가 뚜렷하게 나타나지 않는다는 것이 논문 주장이다.

#### Pre-Activation ResNet과의 비교로 보는 파라미터 효율
논문은 DenseNet이 같은 정확도를 더 적은 파라미터로 낼 수 있음을 강조하며, 특히 pre-activation ResNet과의 비교를 언급한다. 본문에 등장하는 비교 예시는 다음과 같은 형태다.

- `C10+`에서 DenseNet-BC가 pre-activation ResNet-1001과 비슷한 error를 훨씬 적은 파라미터로 달성한다.
- 예시로 **4.51% vs 4.62%**(`C10+`), **22.27% vs 22.71%**(`C100+`) 같은 비교가 언급된다.  

이 비교의 핵심은 단순히 어떤 모델이 더 좋은가가 아니라, Dense connectivity가 같은 성능을 얻는 데 필요한 파라미터(또는 연산) 규모를 줄이는 방향으로 작동한다는 구조적 주장이다.

### 🔸 ImageNet 결과: DenseNet-121/169/201/264
ImageNet에서는 top-1/top-5 error를 single-crop 기준으로 비교한다. 논문은 DenseNet-121부터 264까지의 결과를 Table 3에 정리하고, ResNet 대비 효율을 Fig. 3과 Fig. 4로 시각화한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/770b755d-3767-4368-9858-624ce704e22a/image.png" width="30%">
</p>

#### Single-Crop Vs 10-Crop: Table 3의 표기 방식
논문 Table 3은 top-1/top-5 error를 single-crop / 10-crop 형태로 함께 적는다. 즉, 같은 행에 두 개의 숫자 쌍이 들어가며, 첫 번째가 single-crop, 두 번째가 10-crop이다. 10-crop은 test-time augmentation에 해당하므로, 구조 비교 관점에서는 single-crop을 우선으로 보는 것이 일반적이다. 반면 10-crop은 모델의 상한 성능(더 강한 평가 프로토콜)에서 DenseNet이 어느 수준까지 갈 수 있는지 보여주는 참고치로 볼 수 있다.

위 표는 두 가지를 동시에 보여준다.

1. 깊이가 증가할수록 성능이 좋아지는 경향  
2. Bottleneck $+$ compression(DenseNet-BC)이 단순 DenseNet보다 더 좋은 결과를 내거나, 최소한 효율/성능 관점에서 유리한 설정이 될 수 있음

#### DenseNet-BC의 ImageNet 이점: 구조적 근거
DenseNet-BC는 두 가지 비용 제어 장치를 동시에 켠다.

1. **Bottleneck(1×1)**: 3×3 conv에 들어가는 채널을 제한해 연산량을 낮춘다.  
2. **Compression(transition에서 채널 축소)**: block 사이에서 채널을 줄여, 다음 block에서 입력 채널이 과도하게 커지는 것을 막는다.  

Dense connectivity는 concat 때문에 채널이 계속 늘어나는 구조이므로, ImageNet처럼 해상도가 크고 채널 폭이 큰 구간에서는 비용이 빠르게 커질 수 있다. DenseNet-BC는 그 비용을 억제하면서도 Dense connectivity의 이점을 유지하는 설계로 이해할 수 있고, Table 3의 개선 수치가 그 근거로 제시된다.

#### ImageNet 실험에서의 학습 스케줄
논문은 ImageNet에서 batch size $256$, 총 $90$ epochs 학습을 사용하며, `lr=0.1`에서 시작해 epoch 30/60에서 10배 감소한다고 명시한다. 또한 `weight decay=1e-4`, `Nesterov momentum=0.9`를 사용한다. 이 설정은 당시 ResNet 계열의 표준 학습 레시피와 맞닿아 있고, 논문은 ResNet과의 공정 비교를 위해 가능한 한 같은 설정을 유지하려고 한다.

### 🔹 효율 분석: 파라미터/연산 대비 성능과 Feature Reuse
논문은 DenseNet이 왜 효율적인지에 대해 단순히 결과 숫자만 보여주지 않고, 비교 플롯과 내부 분석을 함께 제시한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/35bcc6ab-44ca-4c46-a52d-4adbde3a2905/image.png" width="80%">
</p>

#### Fig. 4 해석: 동일 Error 대비 비용 감소
Fig. 4는 파라미터 수 또는 FLOPs를 축으로 두고 validation error를 비교한다. DenseNet이 더 적은 파라미터/연산으로도 더 낮은 error를 달성할 수 있다는 점이 핵심 메시지다. 즉, Dense connectivity가 단순히 학습을 안정화하는 트릭을 넘어, 실제 비용-성능 곡선을 유리하게 이동시킨다는 주장이다.

#### Fig. 4 우측 패널(학습/테스트 곡선)의 추가적 시사점
논문은 DenseNet-BC-100과 pre-activation ResNet-1001을 비교하면서, 훈련 loss는 ResNet이 더 낮게 갈 수 있지만 테스트 에러는 비슷할 수 있다는 관찰을 Fig. 4(우측)로 보여준다. 이 장면을 DenseNet의 주장 방식에 맞춰 풀면 다음처럼 정리된다.

- ResNet-1001은 모델 용량이 매우 크기 때문에 훈련 데이터에 더 강하게 맞출 수 있고(훈련 loss 감소), 그 결과가 곧바로 테스트 성능의 큰 개선으로 이어지지 않을 수 있다.  
- DenseNet-BC-100은 훨씬 적은 파라미터로도 비슷한 테스트 에러를 달성할 수 있고, 이는 feature reuse를 통해 표현을 구성하는 방식이 효율적이라는 논지와 연결된다.  

즉, 논문은 단순히 테스트 에러 수치 하나로만 비교하지 않고, 훈련 손실과 테스트 에러 사이의 관계까지 보여주면서 DenseNet의 장점을 모델 크기/일반화 관점으로 설득하려 한다.

#### Fig. 5 해석: 앞 레이어 특징 재사용
DenseNet은 구조적으로 앞 레이어 특징을 계속 입력으로 전달하므로, 뒤 레이어가 앞 레이어 특징을 실제로 쓸 수밖에 없다. 논문은 연결 거리(몇 레이어 떨어졌는지)에 따른 필터 가중치 크기 등을 분석해, 특징 재사용이 실제로 나타난다는 간접 근거를 제시한다. 이는 Dense connectivity가 단순히 경로를 늘린 것 이상의 효과를 갖는다는 논증의 일부다.

#### Fig. 5 관찰 사항의 정리
논문 본문은 Fig. 5 heatmap을 통해 다음과 같은 관찰을 정리한다.

1. 같은 dense block 내부에서, 레이어들은 매우 이른 레이어의 특징까지도 넓게 사용한다. 이는 block 내부에서 feature reuse가 실제로 일어난다는 근거다.
2. Transition layer도 이전 dense block의 여러 레이어 출력에 넓게 의존한다. 즉, 정보가 block 처음에서 끝까지 몇 번의 경유만으로도 흐를 수 있다.
3. 두 번째/세 번째 dense block에서는 transition layer 출력(삼각형의 윗줄)에 대한 평균 가중치가 작게 나타나는 경향이 있다. 논문은 이를 transition이 다소 중복된 특징을 내보낼 수 있음을 시사하는 근거로 보고, DenseNet-BC에서 transition 출력을 압축하는 것이 합리적이라는 해석으로 연결한다.
4. 최종 분류기는 block 전체의 특징을 보지만, 상대적으로 후반 특징에 더 집중하는 경향도 보인다. 이는 네트워크 말단에서 더 고수준 특징이 생성된다는 일반적 직관과 맞닿아 있다.

이 해석은 Dense connectivity가 단지 학습을 쉽게 만든다는 주장만이 아니라, 실제로 레이어 간 특징 재사용이 학습된 가중치 패턴에 반영된다는 점을 보여준다.

### 🔸 구현/메모리 관점: Concat 비용
논문은 DenseNet이 **파라미터 효율**이 좋지만, naive 구현에서는 **메모리 비효율**이 발생할 수 있음을 언급한다. concat이 많아지면 중간 **activation**을 많이 저장해야 하기 때문이다. 따라서 실제 사용에서는 **메모리 효율 구현**이 중요하며, 논문은 이를 위한 구현 아이디어(공유 메모리/재계산 등)를 함께 논의한다.

이 지점은 실무적으로 매우 중요하다. DenseNet은 파라미터 수만 보고 선택하면 좋아 보이지만, 실제 학습/추론에서의 activation 메모리와 framework 구현 최적화가 성능을 좌우할 수 있다.

#### Naive 구현의 메모리 사용 증가 원인: Concat 기반 중복 텐서
DenseNet의 dense block은 레이어가 진행될수록 입력 채널이 커지고, 그 입력은 이전 레이어 출력들을 concat해 만든다. 여기서 naive 구현이 흔히 겪는 문제는, 매 레이어마다

1. 지금까지의 출력을 모두 합친 큰 텐서를 새로 만들고  
2. 그 큰 텐서를 BN/ReLU/Conv에 통과시키고  
3. Backprop을 위해 중간 결과들을 추가로 저장하는

과정이 반복되면서, 메모리 사용량이 구조적으로 커질 수 있다는 점이다. DenseNet은 파라미터 수는 적어도 activation이 많아질 수 있고, 특히 해상도가 큰 앞쪽 stage에서 이 문제가 더 민감해진다.

#### 메모리 효율 구현의 직관적 요약: 저장 최소화와 재계산
논문이 별도 리포트를 안내하는 이유를 직관적으로 풀면, DenseNet의 핵심은 이전 레이어 출력들을 따로 보존해 두고(각 레이어의 $x_{\ell}$), 필요할 때만 concat을 구성하는 방향으로 구현을 최적화해야 한다는 것이다. 또한 학습에서 모든 중간 activation을 저장하는 대신, 일부는 재계산하는 전략(checkpointing)을 쓰면 메모리 사용량을 줄일 수 있다.

이런 전략은 DenseNet만의 전유물은 아니지만, DenseNet처럼 레이어 출력이 여러 번 재사용되는 구조에서는 특히 효과가 크다. 결국 DenseNet은 구조적 아이디어와 구현 전략이 함께 맞물려야, 논문이 주장하는 효율(파라미터/연산)과 실제 사용 경험(GPU 메모리/속도)이 최대한 일치하게 된다.

#### 논문이 제시하는 구체적 디테일: 메모리 효율 구현의 필요성
논문은 학습 설정을 설명하는 Section 4.2에서, DenseNet의 naive 구현이 GPU 메모리 비효율을 포함할 수 있다고 짚고, 메모리 효율 구현에 대한 기술 보고서를 별도로 참조하라고 안내한다. 이는 DenseNet이 구조적으로 많은 intermediate feature를 입력으로 참조하기 때문에, 구현이 단순하면 activation 저장과 concat 비용이 커질 수 있음을 의미한다.

따라서 DenseNet을 실제로 쓰려면 다음 사항까지 함께 고려해야 한다.

1. 프레임워크가 concat을 효율적으로 처리하는지  
2. Activation checkpointing 또는 재계산 전략을 사용할지  
3. Compression/bottleneck으로 채널을 얼마나 공격적으로 줄일지  

---

## 6️⃣ 논의할 점

### 🔹 Feature Reuse가 만드는 파라미터 효율
논문 Discussion은 DenseNet이 ResNet과 겉보기에는 비슷해 보일 수 있지만(Eq.(1)과 Eq.(2)의 차이가 add vs concat이라는 점), 그 작은 차이가 네트워크의 행동을 크게 바꾼다고 강조한다. DenseNet은 concat으로 인해 이전 레이어의 feature-map이 이후 모든 레이어에 직접 접근 가능해지고, 그 결과 feature reuse가 촉진되어 더 compact한 모델이 가능해진다고 주장한다.

이 주장을 뒷받침하기 위해 논문은 Fig. 3과 Fig. 4에서 파라미터 수와 FLOPs 대비 validation error를 비교한다. DenseNet-BC가 대략 ResNet 대비 1/3 수준의 파라미터로도 비슷한 정확도를 달성할 수 있다는 식의 설명이 등장하며, 이는 ImageNet과 CIFAR 트렌드 모두에서 관찰된다고 말한다.

 <p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/132f15c2-5ce8-4100-a9b1-6dcde1c5b741/image.png" width="100%">
</p>

#### 논문이 든 대표 비교
Fig. 4 설명에서 논문은 매우 인상적인 비교를 제시한다. DenseNet-BC(약 0.8M 파라미터)가 1001-layer pre-activation ResNet(10.2M 파라미터)과 유사한 정확도를 달성할 수 있다는 것이다. 그리고 학습 곡선을 보면 ResNet이 더 낮은 training loss로 수렴하지만 test error는 비슷하다는 관찰을 함께 제공한다. 이 관찰은 DenseNet의 효율이 단지 최적화가 쉬워서가 아니라, 파라미터 사용 방식 자체가 달라서 나타날 수 있음을 시사한다.

#### DenseNet-BC 효율성 주장 근거
논문은 DenseNet 변형(_DenseNet, DenseNet-B, DenseNet-C, DenseNet-BC_)을 비교하고, DenseNet-BC가 가장 파라미터 효율적인 경향을 보인다고 정리한다. 직관적으로는 다음과 같다.

- Bottleneck은 3×3의 입력 채널을 제한해 연산을 줄인다.  
- Compression은 transition에서 채널을 줄여 다음 block에서의 비용 성장을 억제한다.  
- Dense connectivity는 feature reuse로 표현 효율을 높인다.  

즉, DenseNet-BC는 Dense connectivity의 장점을 유지하면서, concat으로 인해 생길 수 있는 비용 증가를 구조적으로 제어한 형태다.

### 🔸 Implicit Deep Supervision, Stochastic Depth, & Feature Reuse
논문 Discussion은 세 가지 추가 논점을 다룬다.

#### Implicit Deep Supervision: 짧은 연결이 만드는 학습 신호 강화
DenseNet에서 각 레이어는 많은 후속 레이어의 입력으로 직접 들어간다. 따라서 loss까지의 경로가 다양해지고, 앞 레이어가 더 직접적인 gradient를 받을 수 있다. 논문은 이를 일종의 deep supervision으로 해석한다. 중요한 점은, DSN처럼 명시적 auxiliary classifier를 붙이지 않아도 연결 구조만으로 비슷한 효과가 나타날 수 있다는 주장이다.

#### DSN 대비 차별점: Classifier 다중화 없는 Supervision
논문은 DSN의 장점을 간단히 정리한다. DSN은 내부 레이어에 auxiliary classifier를 붙여, 중간 표현이 분별력 있는 특징을 학습하도록 강제하고(즉, 더 강한 gradient를 앞쪽에 밀어 넣고), 그 결과 매우 깊은 네트워크의 학습을 돕는다.

DenseNet은 이 방향을 **명시적 head**가 아니라 **연결 구조**로 구현한다는 해석이 논문의 핵심이다. dense connectivity로 인해 앞 레이어 출력은 이후 레이어들에 직접 입력되고, 최상단의 단일 classifier가 주는 학습 신호가 (대략) 2~3개의 transition layer만 거치면 block 내부의 거의 모든 레이어로 전달될 수 있다. 따라서 논문은 DenseNet이 deep supervision을 암묵적으로 수행한다고 표현한다.

그리고 논문이 강조하는 또 다른 차이는 손실/그래디언트의 복잡도다. DSN처럼 loss를 여러 개 두면 최적화에서 다루는 항이 많아지고, 각 항의 상호작용을 고려해야 한다. DenseNet은 하나의 loss를 모든 레이어가 공유하는 구조이므로, supervision 효과를 얻으면서도 목표 함수가 상대적으로 단순하다는 장점을 주장한다.

#### DenseNet 관점의 Stochastic Depth 해석
논문은 stochastic depth가 학습 중 일부 residual layer를 랜덤하게 드롭해, 특정 확률로 두 레이어가 직접 연결되는 효과를 만든다고 말한다. pooling layer는 드롭되지 않으므로, 같은 pooling 사이의 두 레이어가 직접 연결될 확률이 생긴다. 이 관점을 따르면 stochastic depth는 확률적으로 dense connectivity를 샘플링하는 효과를 낼 수 있고, DenseNet 관점이 stochastic depth의 성공을 이해하는 데 도움을 줄 수 있다는 논지다.

#### 확률적 직접 연결의 구체화
같은 pooling 구간(즉, 같은 feature-map 크기) 안에 여러 residual layer가 있다고 하자. stochastic depth는 학습 중 일부 레이어를 드롭하므로, 어떤 두 레이어 사이의 모든 중간 레이어가 드롭되는 경우가 생긴다. 그때는 두 레이어가 사실상 바로 연결된 것처럼 동작한다(중간 변환이 사라지므로).

DenseNet은 이 상황을 확률적으로 항상 반복되는 샘플링으로 해석한다. DenseNet은 deterministic하게 모든 레이어가 서로 직접 연결되어 있고, stochastic depth는 확률적으로 일부 직접 연결이 생긴다. 두 방법이 동일하진 않지만, 짧은 경로가 최적화/일반화에 도움이 될 수 있다는 논점은 공유하며, DenseNet의 관점이 stochastic depth의 성공을 다른 언어로 설명해줄 수 있다는 것이 논문이 던지는 연결고리다.

#### Feature Reuse를 실제로 확인하는 실험
논문은 DenseNet이 실제로 앞 레이어의 특징을 쓰는지 확인하기 위해, 학습된 DenseNet(L=40, k=12)을 대상으로 레이어 간 연결 가중치를 분석한다. 각 convolution 레이어가 이전 레이어들의 feature-map에 어느 정도 의존하는지를 평균 절대 가중치로 측정해 heatmap을 만든 것이 Fig. 5다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/1207e4f8-6cd8-4b60-aaac-70e59a39909b/image.png" width="40%">
</p>

#### Fig. 5 Heatmap 축의 해석
논문은 dense block 내부의 각 convolution 레이어 $\ell$에 대해, 그 레이어가 이전 레이어 $s$의 출력 feature-map을 얼마나 사용하는지의 정도를 요약한다. 구체적으로는 convolution 필터 가중치에서, 특정 입력 source(이전 레이어 출력)에 해당하는 부분의 평균 절대값(L1 norm)을 계산하고, 이를 입력 feature-map 개수로 정규화해 비교 가능하게 만든다.

이렇게 만든 2차원 배열에서

- 세로축은 현재 레이어(어떤 레이어가 입력을 받는가)  
- 가로축은 source 레이어(어느 이전 레이어의 출력이 입력으로 들어오는가)  

로 해석할 수 있고, 색이 진할수록 해당 source에 더 크게 의존한다고 볼 수 있다. DenseNet에서 concat이 입력으로 들어오므로, 한 레이어는 block 내의 여러 source에서 온 채널들을 동시에 입력으로 받는데, Fig. 5는 그 의존도를 시각화한 것이다.

#### 가중치 크기를 의존도 Surrogate로 사용하는 근거
가중치의 절대값이 곧바로 causal한 중요도를 의미하진 않지만, 적어도 학습된 필터가 어떤 입력 채널들을 강하게 사용하려고 하는지의 경향을 보여주는 지표로는 유용하다. DenseNet의 핵심 주장이 feature reuse라면, 학습된 모델이 실제로 이전 레이어들의 feature-map에 넓게 가중치를 분산시키는지가 정성적으로라도 관찰될 필요가 있고, Fig. 5는 그 목적에 맞춘 분석이다.

논문이 Fig. 5에서 정리하는 핵심 관찰은 다음과 같은 방향이다.

1. 깊은 레이어도 block 초반 레이어 특징을 직접 활용한다.  
2. Transition layer는 이전 block 전체에서 정보를 받아 다운샘플링한다.  
3. Transition 출력이 일부 중복 특징을 포함할 수 있으며, compression이 이를 줄이는 데 도움이 될 수 있다.  
4. 최종 분류기는 block 전체 특징을 보되, 후반 특징에 상대적으로 더 집중하는 경향도 보인다.

이 분석은 DenseNet의 핵심 주장(feature reuse)을 정량적으로 뒷받침하려는 시도이며, Dense connectivity가 단지 학습을 쉽게 하는 장치가 아니라 표현의 구성 방식 자체를 바꾼다는 논지와 연결된다.

---

## 💡 해당 논문의 시사점과 한계
DenseNet의 가장 큰 의의는 skip connection의 역할을 경로 단축으로만 보지 않고, **특징을 보존하고 재사용하는 메커니즘**으로 재정의한 점이다. ResNet의 identity add는 학습을 안정화했지만, DenseNet은 concat으로 특징을 분해된 형태로 유지해 이후 레이어가 쉽게 재사용할 수 있게 만든다. 또한 growth rate $k$라는 파라미터화는, 모델이 폭을 무작정 키우지 않고도 깊이를 늘리며 표현을 확장할 수 있는 설계 규칙을 제공한다.

실험적으로도 DenseNet은 CIFAR/SVHN/ImageNet에서 강한 성능과 효율을 동시에 보여주며(Table 2, 3), 파라미터/연산 대비 성능 분석(Fig. 4)을 통해 단순한 운이 아니라 구조적 이점임을 설득하려 한다.

#### DenseNet의 설계적 기여: Feature Reuse의 구조적 강제
DenseNet을 읽고 나면, 네트워크를 설계할 때 질문이 하나 바뀐다. 단지 경로를 짧게 만들었는가가 아니라, **이미 계산한 특징을 어떻게 다루는가**가 중요한 축이 된다. DenseNet은 이전 특징을 사라지게 하지 않고(더하기로 섞지 않고), 채널로 보존해 다음 레이어가 필요할 때 직접 가져다 쓰게 만든다.

이 관점은 이후 모델들을 이해할 때도 유용하다. 어떤 모델이 skip/branch를 쓰더라도, 그것이 정보를 혼합(add)하는지, 보존(concat)하는지, 그리고 그 선택이 파라미터/연산/메모리의 어떤 트레이드오프를 만드는지를 질문하는 습관을 남긴다.

#### 논문의 추가 시사점
논문은 Discussion에서 DenseNet과 ResNet이 단지 concat vs add로만 다른 것처럼 보이지만, 그 결과 행동이 크게 달라진다고 강조한다. 특히 다음 세 가지 논점은 이후 설계에서도 반복해서 등장하는 기준점이 된다.

1. Model compactness: Dense connectivity는 feature reuse를 촉진해, 같은 성능을 더 적은 파라미터로 달성할 수 있게 한다.  
2. Implicit deep supervision: 짧은 연결로 각 레이어가 loss에 더 직접적으로 연결되어 학습 신호가 강화된다.  
3. Feature reuse의 관찰 가능성: 구조적 주장에 그치지 않고, Fig. 5처럼 학습된 가중치 패턴을 통해 재사용이 실제로 일어난다는 근거를 제시한다.

또한 논문은 stochastic depth와의 관계도 흥미롭게 언급한다. stochastic depth는 학습 중 레이어를 랜덤 드롭해 일부 경로를 짧게 만드는 방식인데, 이를 DenseNet 관점에서 해석하면 (특정 확률로) 두 레이어가 직접 연결된 효과가 나타날 수 있다는 것이다. 물론 두 방법은 다르지만, 짧은 경로의 중요성이라는 공통된 직관을 공유한다는 점에서 DenseNet은 기존 연구들을 하나의 관점으로 묶어 해석하는 역할도 한다.

#### 한계와 실무적 고려
DenseNet의 한계는 주로 **메모리와 구현**에 있다. concat 기반 누적은 activation을 많이 유지하게 만들고, naive 구현에서는 메모리 사용량이 커질 수 있다. 또한 concat은 데이터 이동 비용이 발생할 수 있어, 하드웨어/프레임워크에 따라 실제 속도 이득이 파라미터 수만큼 크게 나타나지 않을 수 있다. 논문도 메모리 비효율 가능성과 그 완화에 대해 언급한다.

또한 DenseNet-BC처럼 bottleneck과 compression을 잘 튜닝해야 비용-성능 균형이 좋아진다. 즉, Dense connectivity 자체만으로 모든 비용 문제가 해결되는 것은 아니며, 실무에서는 $k$, $\theta$ 같은 설계 파라미터와 구현 최적화가 함께 필요하다.

---

## 👨🏻‍💻 DenseNet 구현하기
DenseNet을 [`lucid`](https://github.com/ChanLumerico/lucid/tree/main)로 구현한 코드는 [`dense.py`](https://github.com/ChanLumerico/lucid/blob/main/lucid/models/imgclf/dense.py)에 위치하고 있다. 이 파일 내 구현 코드를 하나씩 살펴보며 다음과 같은 순서로 설명하고자 한다.

1. DenseNet의 핵심인 `_DenseLayer`의 concat 누적  
2. `_DenseBlock`이 이를 반복하며 채널이 증가하는 구조  
3. `_TransitionLayer`가 채널/해상도를 줄여 block을 이어주는 방식  
4. `DenseNet` 클래스가 전체 네트워크를 조립하는 과정  

### 0️⃣ 전체 대응 관계 요약
- 논문 Eq.(2) Dense connectivity(이전 모든 feature concat) → `_DenseLayer.forward`의 `lucid.concatenate([x, out], axis=1)`
- 논문 dense block(동일 해상도에서 레이어 반복) → `_DenseBlock`의 `ModuleList` + 반복 `forward`
- 논문 transition layer(1×1 + avg pool) → `_TransitionLayer`
- 논문 DenseNet-121/169/201/264(Table 1) → `densenet_121/169/201/264`의 `block_config`

#### Lucid 구현이 타깃하는 DenseNet 변형을 먼저 정리
`lucid/models/imgclf/dense.py`는 CIFAR용 3-block DenseNet을 별도로 구현하진 않고, ImageNet에서 흔히 쓰는 stem(7×7 stride 2 conv + maxpool)과 4개의 dense block, 그리고 transition 기반 다운샘플링 흐름을 구현한다. 또한

- Bottleneck factor는 `bottleneck=4`로 고정되어 있고  
- Transition의 compression은 `out_channels = in_channels // 2`로 고정되어 있어  

구조적으로 DenseNet-B + compression(대략 $\theta=0.5$)에 해당하는 DenseNet-BC 방향을 기본으로 갖는다.

즉, Lucid 구현은 논문이 ImageNet에서 주로 사용하는 DenseNet-BC 흐름을 코드 형태로 이해할 수 있게 해 주는 구현이라고 볼 수 있다.

### 1️⃣ `_DenseLayer`: Bottleneck + 3×3 Conv 후 Concat으로 누적
Lucid의 `_DenseLayer`는 DenseNet-B 스타일의 bottleneck을 기본으로 갖는다. 흐름은 BN→ReLU→Conv(1×1)으로 채널을 `bottleneck * growth_rate`로 만들고, BN→ReLU→Conv(3×3)으로 `growth_rate` 채널을 생성한 뒤, 입력 `x`와 `out`을 채널 축으로 concat한다.

```python
class _DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bottleneck: int = 4,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels, bottleneck * growth_rate, kernel_size=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(bottleneck * growth_rate)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            bottleneck * growth_rate,
            growth_rate,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))

        return lucid.concatenate([x, out], axis=1)
```

여기서 Dense connectivity의 핵심이 마지막 줄에 있다. `lucid.concatenate([x, out], axis=1)`는 채널 축(axis=1)에서 입력과 새 특징을 붙인다. 즉, 레이어는 입력을 덮어쓰지 않고 보존하면서 새로운 $k$채널만 추가한다. 이것이 논문이 말한 growth rate 관점과 정확히 맞물린다.

#### `_DenseLayer`의 채널 증가 요약
입력 채널이 $C$이고 growth rate가 $k$라면, `_DenseLayer.forward`의 반환은 채널이 $C+k$가 된다. 이 누적이 dense block 내부에서 반복되므로, block 전체의 출력 채널은 `in_channels + num_layers * growth_rate`가 된다. Lucid의 `DenseNet.__init__`이 `in_channels += num_layers * growth_rate`로 채널 수를 업데이트하는 이유가 바로 이 전개를 코드로 반영한 것이다.

### 2️⃣ `_DenseBlock`: 레이어를 반복하면서 채널이 점진적으로 증가
`_DenseBlock`은 `num_layers`만큼 `_DenseLayer`를 만들고, 각 레이어의 `in_channels`가 `growth_rate`만큼 증가하도록 구성한다.

```python
class _DenseBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        growth_rate: int,
        bottleneck: int = 4,
    ) -> None:
        super().__init__()
        layers = [
            _DenseLayer(in_channels + i * growth_rate, growth_rate, bottleneck)
            for i in range(num_layers)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
```

이 구현은 논문 Fig. 1의 dense block을 그대로 코드로 옮긴 형태다. `i`번째 레이어의 입력 채널을 `in_channels + i * growth_rate`로 잡는 이유는, 이전 레이어들이 `growth_rate` 채널씩 누적되기 때문이다. 그리고 `forward`에서 `x = layer(x)`를 반복하면, 매번 concat이 일어나 채널이 점점 커진다.

#### Dense Block 내부의 텐서 흐름(채널 축)
Dense block 안에서 텐서의 공간 해상도 $H\times W$는 유지되고, 채널만 증가한다.

- Block 입력: $(N, C_0, H, W)$  
- 1번째 dense layer 이후: $(N, C_0+k, H, W)$  
- 2번째 dense layer 이후: $(N, C_0+2k, H, W)$  
- $m$번째 이후: $(N, C_0+mk, H, W)$  

즉, DenseNet의 기본 동역학은 채널 축을 누적하는 방향이며, 해상도 변화는 transition layer가 전담한다.

### 3️⃣ `_TransitionLayer`: 1×1 Conv로 채널 줄이고 Avg Pool로 다운샘플링
Dense block 사이에는 transition layer가 들어간다. Lucid 구현은 `BN→ReLU→Conv(1×1)`로 채널을 `out_channels`로 만들고, 2×2 average pooling으로 해상도를 줄인다.

```python
class _TransitionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(self.relu(self.bn(x)))
        x = self.pool(x)
        return x
```

여기서 `out_channels`가 compression의 역할을 한다. `DenseNet` 클래스는 transition을 만들 때 `out_channels = in_channels // 2`로 설정해, 논문에서 말한 $\theta=0.5$ compression과 같은 방향을 택한다(정확히는 floor 없이 정수 나눗셈). 즉, Lucid DenseNet은 DenseNet-BC 스타일의 compression을 기본으로 구현한 것으로 해석할 수 있다.

#### Transition이 담당하는 두 축: 채널 축소 + 해상도 축소
transition은 Dense block이 만든 누적 채널을 줄여 다음 block의 비용을 제어하고, 동시에 pooling으로 해상도를 절반으로 줄여 이후 block이 더 넓은 receptive field를 다루도록 만든다. DenseNet의 효율 설계는 사실상

- Dense block에서 채널이 증가하는 구조  
- Transition에서 채널을 다시 줄이는 구조  

의 반복으로 요약된다. Lucid 구현도 `in_channels += ...`와 `in_channels = out_channels` 업데이트를 통해 이 반복을 명시적으로 표현한다.

### 4️⃣ `DenseNet`: Stem + (Block, Transition) 반복 + 활성화
Lucid의 `DenseNet`은 입력 stem으로 7×7 stride 2 conv와 3×3 maxpool을 사용한 뒤, `block_config`에 따라 dense block과 transition을 순서대로 쌓는다. 마지막은 `BN→ReLU→global average pooling→FC`로 분류를 수행한다.

```python
class DenseNet(nn.Module):
    def __init__(
        self,
        block_config: tuple[int],
        growth_rate: int = 32,
        num_init_features: int = 64,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.conv0 = nn.ConvBNReLU2d(
            3, num_init_features, kernel_size=7, stride=2, padding=3, conv_bias=False
        )
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        in_channels = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(in_channels, num_layers, growth_rate)
            self.blocks.append(block)

            in_channels += num_layers * growth_rate

            if i != len(block_config) - 1:
                out_channels = in_channels // 2
                transition = _TransitionLayer(in_channels, out_channels)
                self.transitions.append(transition)

                in_channels = out_channels

        self.bn_final = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool0(self.conv0(x))

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        x = self.avgpool(self.relu(self.bn_final(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
```

이 코드에서 논문 구조와 직접 대응되는 부분은 `blocks`와 `transitions`를 함께 쌓는 루프다. `in_channels += num_layers * growth_rate`는 dense block 내부에서 채널이 누적되는 전개를 그대로 반영하고, `out_channels = in_channels // 2`는 transition의 compression을 구현한다. 또한 마지막 분류 head는 DenseNet 논문이 사용하는 global average pooling 기반 head와 같은 형태다.

#### `DenseNet.forward`의 단계적 정리
Lucid 코드의 forward는 논문 DenseNet의 표준 흐름을 그대로 따른다.

1. Stem: `conv0`(7×7 stride 2) → `pool0`(3×3 maxpool stride 2)  
2. 반복: Dense block을 통과 → (마지막 block이 아니라면) transition으로 다운샘플  
3. Tail: BN → ReLU → global average pooling → flatten → FC  

이 흐름에서 Dense connectivity 자체는 block 내부에서 구현되고, 전체 네트워크 관점의 해상도 변화는 transition이 담당한다는 점이 분명해진다.

### 5️⃣ 모델 팩토리: `densenet_121/169/201/264`
마지막으로 Lucid는 `@register_model` 데코레이터로 네 가지 변형을 등록한다. 차이는 `block_config`뿐이다.

```python
@register_model
def densenet_121(num_classes: int = 1000, **kwargs) -> DenseNet:
    block_config = (6, 12, 24, 16)
    return DenseNet(block_config, num_classes=num_classes, **kwargs)


@register_model
def densenet_169(num_classes: int = 1000, **kwargs) -> DenseNet:
    block_config = (6, 12, 32, 32)
    return DenseNet(block_config, num_classes=num_classes, **kwargs)


@register_model
def densenet_201(num_classes: int = 1000, **kwargs) -> DenseNet:
    block_config = (6, 12, 48, 32)
    return DenseNet(block_config, num_classes=num_classes, **kwargs)


@register_model
def densenet_264(num_classes: int = 1000, **kwargs) -> DenseNet:
    block_config = (6, 12, 64, 48)
    return DenseNet(block_config, num_classes=num_classes, **kwargs)
```

논문 Table 1의 DenseNet-121/169/201/264는 각 dense block의 레이어 수로 정의된다. Lucid 코드도 같은 방식으로 네트워크를 구성한다. 따라서 Lucid DenseNet은 논문에서 제시한 대표 아키텍처 계열을 block_config로 직접 표현하고 있다고 볼 수 있다.

#### Lucid 최소 사용 예시
Lucid 구현을 확인하려면 팩토리 함수를 직접 호출해 모델을 만들 수 있다.

```python
from lucid.models import densenet_121

model = densenet_121(num_classes=1000)
```

이렇게 생성된 모델은 `growth_rate=32`, `num_init_features=64` 기본값을 사용한다. 논문에서 언급한 다양한 설정(L, k, DenseNet-BC 변형)을 그대로 재현하려면, `DenseNet(block_config, growth_rate=..., num_init_features=...)` 형태로 직접 인자를 조절하는 방식이 필요하다(단, Lucid 구현은 transition compression을 고정으로 두고 있어 논문 설정과 완전히 동일하게 일반화되진 않는다).

---

## ✅ 정리
**DenseNet**은 각 레이어가 이전 모든 레이어의 feature-map을 입력으로 받아 concat으로 누적하는 dense connectivity를 제안한다. 이 설계는 정보/그래디언트 흐름을 강화하고, 특징 재사용을 촉진해 파라미터 효율을 높인다는 것이 논문 주장이다. Dense block과 transition layer로 해상도 변화 문제를 해결하고, growth rate $k$로 레이어 폭을 제어하며, bottleneck(DenseNet-B)과 compression(DenseNet-C)을 결합한 DenseNet-BC로 비용-성능 균형을 강화한다. 실험에서는 CIFAR/SVHN/ImageNet에서 강한 성능과 효율을 보고하고, 파라미터/FLOPs 대비 성능 분석과 내부 연결 가중치 분석으로 feature reuse를 뒷받침하려 한다.
동시에 concat 기반 누적은 activation 메모리와 구현 효율 측면의 비용을 만들 수 있어, 실제 사용에서는 메모리 효율 구현과 하이퍼파라미터 선택이 함께 중요해진다.

#### 📄 출처
Huang, Gao, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger. "Densely Connected Convolutional Networks." *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, arXiv:1608.06993.
