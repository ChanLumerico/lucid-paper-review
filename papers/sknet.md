# [SKNet] Selective Kernel Networks
SKNet은 CNN의 receptive field(RF) 크기를 고정된 설계 값으로 두지 않고, **입력 자극(특히 객체 스케일)**에 따라 각 블록이 어떤 커널 스케일 정보를 더 반영할지 **동적으로 선택**하도록 만든 네트워크다. 논문이 내세우는 출발점은 생물학적 관찰이다. 시각 피질 뉴런의 RF 크기는 항상 동일하게 고정되어 있지 않고, 자극에 의해 조절되는 경향이 있다는 연구들이 있다. 반면 표준 CNN에서는 같은 레이어의 뉴런들이 동일한 커널 크기, 동일한 RF 크기를 공유하는 것이 일반적이다.

이 논문은 multi-scale 정보를 같은 레이어에서 다루는 전통적 아이디어(Inception류)를 출발점으로 삼되, 단순 concat/선형 결합만으로는 RF 크기 적응이 충분히 강해지기 어렵다고 주장한다. 대신 **Selective Kernel(SK) convolution**이라는 빌딩 블록을 제안해, 여러 커널 스케일의 branch를 만들고(Split), 그 branch들을 요약한 전역 표현으로 선택 가중치를 계산한 뒤(Fuse), softmax 기반 attention으로 branch를 가중합해 출력 feature를 만든다(Select). 논문은 이 과정을 통해 fusion layer의 뉴런들이 입력에 따라 유효 RF 크기를 바꿔가며 정보를 수집할 수 있다고 해석한다.

이 리뷰는 논문 전개를 따라 SK convolution의 수식과 설계 직관, ResNeXt 계열 backbone에 SK unit을 통합해 SKNet을 구성하는 방법, ImageNet/경량 모델/ CIFAR/ablation/분석 실험을 상세히 해석한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/010d64d3-ec74-49b7-9b19-e13254358a75/image.png" width="70%">
</p>

---

## 1️⃣ 논문 배경

### 🔹 고정 RF의 한계와 적응 RF의 문제의식
표준 CNN에서 receptive field는 커널 크기, stride, dilation, pooling 등 설계로 결정된다. 같은 레이어의 뉴런들은 동일한 커널 크기를 갖는 convolution을 공유하므로, 같은 stage 내부에서 RF 크기는 사실상 고정된다. 물론 네트워크 깊이가 증가하면 RF는 커지지만, 이것은 입력에 따라 바뀌는 적응이 아니라 **구조에 의해 결정되는 정적 증가**다.

논문은 _생물학적 시각 피질(V1)_ 의 관찰을 언급하며, 뉴런의 RF 크기가 자극에 의해 조절되는 현상(고전적 RF, non-classical RF 등)을 소개한다. 여기서 핵심은 같은 영역의 뉴런들이 단지 서로 다른 RF 크기를 가질 뿐만 아니라, **한 뉴런의 유효 RF 크기 자체가 자극에 의해 변할 수 있다**는 관찰이다. 논문은 이 관찰을 CNN 설계로 옮길 때, 단순히 서로 다른 커널을 병렬로 두는 것(예: Inception의 3×3/5×5/7×7 branch)만으로는 충분하지 않을 수 있다고 주장한다.

#### CRF/NCRF 직관을 CNN으로 옮겼을 때의 차별점
논문이 말하는 맥락을 단순화하면, 고전적 receptive field(CRF)는 뉴런이 **직접적으로 반응**하는 핵심 영역이고, 그 바깥의 자극(nCRF)은 **직접적인 자극이 아니더라도** 뉴런 반응을 조절할 수 있다는 관찰이다. 즉, 뉴런이 실제로 활용하는 문맥 범위는 고정된 원형/사각형 영역이라기보다, 자극에 의해 확장되거나 축소될 수 있는 유효 범위다.

CNN 관점에서 보면, 3×3만을 반복해서 쌓는 구조는 각 레이어에서 직접적으로 보는 문맥을 제한한다. 반대로 큰 커널은 더 넓은 문맥을 한 번에 가져오지만 비용이 크다. SKNet은 이 둘을 양자택일로 두지 않고, **여러 문맥 스케일을 동시에 만들어 두고 그중 무엇을 얼마나 쓸지 선택**하도록 설계해, 유효 RF를 입력 조건부로 바꾸는 방향을 취한다.

#### 선형 결합과 비선형 선택의 차이
Inception처럼 여러 branch를 concat하고 다음 레이어에서 이를 선형 결합하면, 입력에 따라 스케일 비중이 달라지는 효과가 생길 수 있다. 하지만 이 효과는 선택 신호가 명시적이지 않고, 선택을 담당하는 파라미터가 다음 레이어의 convolution에 흡수되어 있어, 선택 메커니즘 자체를 독립적으로 ablation하거나 분석하기 어렵다.

반면 SKNet은 선택 가중치를 **softmax attention으로 분리해 계산**한다. 즉, 어떤 branch가 얼마나 기여했는지가 중간 변수(attention weight)로 드러나고, 그 변수가 입력에 따라 변한다는 사실을 분석으로 보여줄 수 있다(Fig. 3, 4). 논문이 linear aggregation이 충분히 강한 적응을 제공하지 못할 수 있다고 말하는 이유가 여기와 맞닿아 있다.

#### Inception식 Multi-Branch와 RF 적응의 차이
Inception류 블록은 다양한 커널로 multi-scale 정보를 만들어 concat하거나 합친다. 다음 층에서 이 정보를 선형 결합하면, 결과적으로는 입력에 따라 어떤 스케일 정보가 더 중요해지는 현상이 생길 수 있다. 하지만 논문은 이를 **암묵적이고 약한 적응**으로 본다. 다음 층의 convolution이 branch들의 출력을 선형으로 섞는다는 사실만으로는, 자극에 따라 RF 크기가 선택적으로 조정된다고 보기 어려우며, 무엇보다 그 선택을 명시적으로 모델링하고 분석하기가 어렵다.

#### SKNet의 설계 철학
논문이 던지는 설계 질문은 다음과 같이 정리할 수 있다.

1. 같은 위치의 특징이라도 입력 스케일에 따라 **더 넓은 문맥**이 필요할 수 있다.  
2. 그렇다면 동일 레이어에서도 **여러 스케일**의 convolution 결과를 만들고,  
3. 그중 어떤 스케일을 더 반영할지 입력에 따라 선택하도록 만들면,  
4. 유효 RF 크기가 입력에 따라 **동적으로 변하는 효과**를 얻을 수 있다.

이때 중요한 것은 선택이 단순한 고정 파라미터(학습된 상수)로 결정되는 것이 아니라, 입력 feature에서 계산되는 **soft attention** 형태로 이루어진다는 점이다. SKNet은 이 선택 메커니즘을 블록 단위의 primitive로 정의하고, 이를 깊게 쌓아 네트워크를 구성한다.

### 🔸 Selective Kernel(SK) 유닛
논문은 SK convolution이 _Split–Fuse–Select_ 의 세 연산자로 구성된다고 명시한다. 직관적으로는 다음과 같다.

- **Split**: 서로 다른 커널(또는 dilation으로 근사한 커널)을 갖는 여러 branch로 feature를 변환한다.  
- **Fuse**: 여러 branch 출력을 요약해 전역 정보를 만들고, 선택을 위한 compact descriptor를 만든다.  
- **Select**: softmax attention으로 branch를 가중합해 최종 출력을 만든다.

이 구조가 노리는 효과는, 입력에 따라 attention이 달라지면서 어떤 branch가 더 크게 기여하는지가 달라지고, 그 결과 유효 RF 크기가 동적으로 바뀌는 것처럼 동작하는 것이다. 또한 논문은 이 과정이 과도한 연산 증가 없이 구현될 수 있으며, 실제로 ImageNet에서 기존 attention 기반 모델보다 **유사 또는 더 낮은 복잡도**에서 더 좋은 성능을 달성한다고 주장한다.

#### SKNet을 관통하는 메시지
이 논문은 단순히 새로운 backbone을 내세우기보다, 커널 선택을 통해 RF를 적응시키는 **메커니즘**을 제안하고, 이를 ResNeXt 스타일의 bottleneck에 통합해 강력한 모델을 만든다. 따라서 SKNet은 하나의 특정 구조라기보다, 여러 backbone에 부착 가능한 block 설계로 이해하는 편이 자연스럽다(논문도 lightweight 네트워크로의 적용을 별도로 실험한다).

#### Deformable Convolution과의 관계
논문은 **deformable convolution** 계열도 관련 연구로 언급한다. deformable conv는 sampling 위치를 학습해 공간적으로 어디를 볼지 조절할 수 있지만, SKNet이 강조하는 것은 위치 오프셋을 바꾸는 것이 아니라 **스케일(커널 RF) 선택을 통해 문맥 범위를 조절**하는 쪽이다. 즉, 같은 위치에서 서로 다른 RF 크기의 정보를 만들어 두고, 그중 어떤 RF를 더 반영할지 soft attention으로 결정한다는 점이 SK의 핵심 차별점이다.

---

## 2️⃣ 관련 연구

### 🔹 Multi-Branch CNN과 SKNet의 위치
논문은 multi-branch 설계의 계보를 여러 방식으로 훑는다. Inception류는 서로 다른 커널을 병렬로 배치해 multi-scale 정보를 동시에 만들고, 이를 concat하는 방식으로 표현력을 높였다. 그 외에도 highway network, shake-shake, fractal 구조 등 다양한 multi-branch/skip 계열 설계가 존재한다.

논문이 강조하는 차이는 **두 가지**다.

1. SKNet의 branch 설계는 Inception처럼 각 branch를 복잡하게 커스터마이즈하기보다, 상대적으로 단순한 형태(다른 커널 크기 혹은 dilation)로 구성된다.  
2. 중요한 것은 branch 결과를 단순히 합치거나 concat하는 것이 아니라, **입력에 의해 유도되는 attention으로 branch를 선택**한다는 점이다.

#### 단순 여러 Branch의 부족함
branch를 많이 두는 것은 multi-scale 정보를 담는 데 유리할 수 있다. 하지만 논문은 RF 적응이라는 관점에서는, branch를 만들었다는 사실만으로는 충분하지 않고, **branch 간 정보 흐름을 제어하는 선택 장치**가 있어야 한다고 본다. 그 장치가 곧 SK convolution의 Select 단계다.

### 🔸 Grouped/Depthwise/Dilated Convolution과 Attention 메커니즘
논문은 비용을 줄이기 위한 convolution 변형(그룹/깊이별/확장(dilation))과 attention 계열 연구들을 함께 연결한다.

- **Grouped convolution**은 연산/파라미터를 줄이면서도 정확도를 높일 수 있고, ResNeXt에서는 이를 _cardinality_ 로 해석한다.  
- **Depthwise convolution**은 모바일 네트워크에서 표준적이며, pointwise와 조합해 효율을 극대화한다.  
- **Dilated convolution**은 커널 크기를 키우지 않고 RF를 키우는 방법이다.

SK convolution은 큰 커널(예: 5×5)을 직접 쓰면 비용이 커지기 때문에, 논문에서는 **dilated 3×3으로 큰 커널을 근사**하는 방식을 사용한다. 즉, RF 적응을 위해 multi-scale branch를 만들되, 그 branch 자체는 효율적으로 구성해야 한다는 관점을 취한다.

또한 SK의 Select 단계는 attention의 한 형태다. 다만 spatial attention처럼 위치마다 가중치를 주는 것이 아니라, **branch(스케일) 축에 대한 softmax 선택**을 수행한다. 그리고 논문은 이 선택이 채널별로 다르게 일어나도록 수식을 구성해, 스케일 선택이 채널마다 다를 수 있음을 강조한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/667450f5-5805-40bd-b55a-99a46dbb6174/image.png" width="40%">
</p>

---

## 3️⃣ 모델 방법론

### 🔹 Selective Kernel Convolution 개요: Split–Fuse–Select
논문은 SK convolution을 Fig. 1과 함께 정의한다. 기본 아이디어는 간단하다. 입력 feature map $\mathbf{X}$에 대해 서로 다른 커널 크기를 갖는 두(또는 그 이상) 개의 branch 변환을 적용해 서로 다른 스케일의 출력 feature를 만든다. 이후 이 branch들을 요약한 전역 신호로 선택 가중치를 만들고, 그 가중치로 branch 출력을 가중합한다.

논문은 2-branch 케이스(커널 3과 5)를 기본 예로 설명하면서도, 동일한 원리를 더 많은 branch로 일반화할 수 있다고 언급한다.

#### Mixture-Of-Experts 관점의 SK Convolution
SK convolution은 형태적으로는 여러 **expert**(branch conv)가 있고, gating 네트워크가 그 expert들의 출력을 softmax로 섞는 구조다. 다만 일반적인 mixture-of-experts가 레이어 단위로 expert를 바꾸는 것과 달리, SK는 **동일한 위치의 feature를 서로 다른 스케일로 계산한 뒤** 그 스케일들을 섞는다. 이 관점에서 Select 단계는 스케일 축의 gating이고, 그 결과 유효 RF가 입력 조건부로 달라진다.

#### 입력/출력 표기
논문은 입력 feature map을

$$
\mathbf{X} \in \mathbb{R}^{H'\times W'\times C'}
$$

로 두고, 두 branch 변환의 출력은

$$
\widetilde{\mathbf{U}},\,\widehat{\mathbf{U}} \in \mathbb{R}^{H\times W\times C}
$$

로 둔다. 구현에서는 stride에 따라 $H,W$가 달라질 수 있지만, 두 branch 출력은 동일한 shape을 갖는다(그래야 합치고 선택할 수 있다).

### 🔸 Split: 서로 다른 커널/스케일의 Branch 만들기
Split 단계에서 논문은 기본적으로 커널 크기 3과 5를 가진 두 변환 $\widetilde{\mathcal{F}}$와 $\widehat{\mathcal{F}}$를 적용한다.

- $\widetilde{\mathcal{F}}: \mathbf{X}\mapsto\widetilde{\mathbf{U}}$ (기본 3×3)  
- $\widehat{\mathcal{F}}: \mathbf{X}\mapsto\widehat{\mathbf{U}}$ (기본 5×5)

논문은 각 branch 변환이 단순한 convolution 하나가 아니라, **효율적인 grouped/depthwise convolution + BatchNorm + ReLU**의 순차 조합으로 이루어진다고 말한다. 즉, branch에서 multi-scale 정보를 만들되, 비용을 억제하기 위해 그룹/깊이별 구조를 적극적으로 사용한다.

#### 큰 커널의 Dilation 근사 배경
논문은 5×5 커널을 그대로 쓰는 대신, dilation 2의 3×3 convolution으로 근사해 비용을 줄인다. dilation 2의 3×3은 대략 5×5 RF를 커버할 수 있기 때문에, 큰 커널 branch의 목적(더 큰 문맥 수집)을 유지하면서도 연산량을 줄일 수 있다는 판단이다.

이 점은 이후 ablation(Table 6)에서도 다시 나타난다. 논문은 3×3 + dilation 조합이 큰 커널을 직접 쓰는 것과 비교해 성능/복잡도 측면에서 유리할 수 있다는 경험적 결론을 언급한다.

### 🔹 Fuse: 전역 요약과 Compact Descriptor 만들기
Fuse 단계는 선택 가중치의 입력이 되는 전역 신호를 만드는 과정이다. 논문은 먼저 두 branch 출력을 **element-wise sum**으로 합친다.

$$
\mathbf{U} = \widetilde{\mathbf{U}} + \widehat{\mathbf{U}} \tag{1}
$$

이 합을 만드는 이유는, 선택 가중치가 한 branch만을 보고 결정되지 않도록 하기 위해서다. 논문 표현대로 말하면, gates는 여러 branch의 정보를 모두 통합해야 한다.

그 다음 global average pooling으로 채널 통계 벡터 $\mathbf{s}\in\mathbb{R}^{C}$를 만든다. 채널 $c$에 대해

$$
s_c = \mathcal{F}_{gp}(\mathbf{U}_c) = \frac{1}{H\times W}\sum_{i=1}^{H}\sum_{j=1}^{W} U_c(i,j)\tag{2}
$$

로 정의된다. 이것은 SENet의 squeeze와 유사한 역할을 하며, 공간 정보를 요약해 전역 문맥을 채널 축으로 압축한다.

이후 선택을 위한 compact descriptor $\mathbf{z}\in\mathbb{R}^{d\times 1}$를 만든다.

$$
\mathbf{z} = \mathcal{F}_{fc}(\mathbf{s}) = \delta\big(\mathcal{B}(W\mathbf{s})\big)\tag{3}
$$

여기서 $\delta$는 ReLU, $\mathcal{B}$는 BatchNorm, $W\in\mathbb{R}^{d\times C}$다. 그리고 $d$는 reduction ratio $r$로 제어된다.

$$
d = \max(C/r, L)\tag{4}
$$

$L$은 최소 차원(논문 기본값 $32$)이다. 이 설계는 두 가지 목적을 가진다.

1. $C$가 큰 stage에서도 gating 네트워크의 비용을 폭발시키지 않기  
2. 선택을 위한 표현을 너무 과도하게 압축하지 않기($L$로 하한을 둠)

#### Eq.(4)의 의미
채널 수 $C$가 커질수록, 채널 간 상호작용과 선택의 복잡도도 커질 수 있다. 하지만 $d=C$로 두면 비용이 커진다. 그래서 $r$로 $d$를 줄이되, 너무 작아지면 선택이 부정확해질 수 있으므로 최소값 $L$을 둔다. 논문은 이 설계를 통해 효율과 정확도의 균형을 잡으려 한다.

#### Gating 네트워크의 파라미터 규모의 대략적 계산
Fuse/Select 과정에서 학습되는 핵심 파라미터는 크게 세 덩어리로 볼 수 있다.

1. $W\in\mathbb{R}^{d\times C}$: $\mathbf{s}\mapsto\mathbf{z}$로 가는 차원 축소( Eq.(3) )  
2. $A\in\mathbb{R}^{C\times d}$, $B\in\mathbb{R}^{C\times d}$: $\mathbf{z}\mapsto a_c,b_c$로 가는 채널별 분기 점수( Eq.(5) )  

2-branch 기준으로만 보면, gating 파라미터 수는 대략

$$
|W| + |A| + |B| \approx dC + Cd + Cd = 3Cd
$$

정도로 잡을 수 있다(BN 파라미터까지 포함하면 약간 더 늘어난다). 만약 $d=C$라면 $\mathcal{O}(C^2)$가 되어 stage가 깊어질수록 비용이 커질 수 있다. Eq.(4)에서 $d\approx C/r$로 줄이면, 위 항은 대략 $3C(C/r)=3C^2/r$로 내려간다. 즉, $r$은 선택 네트워크의 비용을 직접적으로 낮추는 핵심 손잡이이며, 논문이 $L$로 하한을 둔 이유는 너무 작은 $d$가 선택 표현력을 망가뜨리는 것을 막기 위함으로 이해할 수 있다.

#### Fuse 단계에서 Branch 결합($\mathbf{U}$)의 필요성
Eq.(1)의 $\mathbf{U}=\widetilde{\mathbf{U}}+\widehat{\mathbf{U}}$는 단순한 합이지만, 선택 메커니즘의 입력을 정의한다는 점에서 중요하다. 만약 gating이 한 branch만을 보고 결정되면, gating은 다른 branch의 존재를 반영하기 어렵다. 반대로 합을 통해 공통 요약을 만들면, gating은 두 branch의 활성 패턴을 모두 반영한 전역 신호에서 선택 가중치를 만들게 된다.

또한 $\mathbf{U}$를 spatial pooling의 입력으로 쓰면, gating 네트워크는 공간 위치의 세부 패턴이 아니라 전역적으로 어떤 채널이 얼마나 활성화되었는지(그리고 두 스케일이 어떻게 합쳐졌는지)에 더 민감해진다. 이 설계는 선택 신호가 위치별 노이즈에 과도하게 흔들리기보다, 입력 스케일 변화 같은 전역적 변화에 따라 의미 있게 이동하도록 돕는 장치로 해석할 수 있다.

### 🔸 Select: Softmax Attention으로 스케일 선택
Select 단계에서 핵심은, branch별 가중치를 softmax로 만들고 그 가중치로 branch 출력을 가중합한다는 점이다. 논문은 2-branch 케이스에서 채널별 attention을 다음처럼 정의한다.

$$
a_c = \frac{e^{A_c\mathbf{z}}}{e^{A_c\mathbf{z}} + e^{B_c\mathbf{z}}},\quad b_c = \frac{e^{B_c\mathbf{z}}}{e^{A_c\mathbf{z}} + e^{B_c\mathbf{z}}}\tag{5}
$$

여기서 $A,B\in\mathbb{R}^{C\times d}$이고, $A_c$는 $A$의 $c$번째 행이다. 즉, attention은 단일 스칼라가 아니라 **채널마다 다른 가중치**를 갖는다.

그리고 최종 출력 feature map $\mathbf{V}$는 채널별로 가중합해 얻는다.

$$
V_c = a_c\cdot \widetilde{U}_c + b_c\cdot \widehat{U}_c,\quad a_c+b_c=1\tag{6}
$$

즉, 한 채널에서는 3×3 branch가 더 중요하고, 다른 채널에서는 5×5 branch가 더 중요해질 수 있다. 이 구조가 논문이 말하는 RF 적응의 핵심이다. 큰 객체가 들어오면 더 넓은 문맥을 보는 branch의 비중이 높아지고, 작은 객체면 상대적으로 좁은 문맥 branch의 비중이 높아지는 식의 적응을 기대할 수 있다

#### 유효 RF의 관점
Eq.(6)은 두 branch의 출력을 단순히 선택(0 또는 1)하는 것이 아니라, softmax 가중치로 **연속적으로 섞는** 형태다. 따라서 유효 RF는 3×3 또는 5×5 중 하나로 딱 떨어진다기보다, 입력에 따라 3×3 쪽에 더 가깝게 혹은 5×5 쪽에 더 가깝게 이동한다.

이 점이 중요하다. hard routing(완전한 분기 선택)과 달리, soft attention은 **학습이 안정적이고 미분 가능**하며, 다양한 입력에서 점진적으로 스케일을 조절할 수 있다. 논문이 이후 분석에서 객체 크기가 커질수록 큰 커널 attention이 증가한다고 말할 때, 그것은 $0→1$의 급격한 변화가 아니라 가중치 분포의 이동으로 관찰된다(Fig. 3).

#### 채널별 Attention 채택 배경
논문 수식은 $a_c,b_c$처럼 채널별로 **다른 분기 가중치**를 허용한다. 이는 직관적으로도 자연스럽다. 어떤 채널은 텍스처처럼 작은 스케일 정보가 중요하고, 어떤 채널은 객체의 큰 형태나 문맥이 중요할 수 있다. 같은 입력 이미지 안에서도 채널마다 담당하는 특징이 다르므로, 모든 채널이 동일한 스케일을 선택하도록 강제하는 것보다 채널별로 스케일 선택을 다르게 허용하는 편이 표현력 측면에서 유리할 수 있다.

또한 채널별 attention은 네트워크가 스케일 선택을 더 세밀하게 조정할 수 있게 만든다. 예컨대 객체가 커질 때 모든 채널이 일괄적으로 큰 커널을 선택하는 것이 아니라, 그 객체를 설명하는 데 필요한 일부 채널들에서만 큰 커널 비중을 높이고, 나머지는 유지할 수도 있다. Fig. 3의 채널별 플롯은 이런 가능성을 시각적으로 뒷받침하는 장치로 이해할 수 있다.

#### 2-Branch에서 3-Branch 이상으로의 확장
논문은 Eq.(1), (5), (6)을 확장하면 더 많은 branch에서도 같은 방식으로 선택할 수 있다고 말한다. 실제로 3-branch라면 softmax는 3개의 weight를 만들고, $\mathbf{V}$는 세 branch의 가중합이 된다. 이후 ablation(Table 7)에서 $M=2$와 $M=3$을 비교하며, branch 수가 늘어날 때의 성능/효율 트레이드오프를 분석한다.

#### 일반 $M$-Branch Softmax 형태로 재서술
2-branch에서는 $a_c+b_c=1$이지만, $M$-branch에서는 채널 $c$에 대해 softmax를 다음처럼 일반화할 수 있다.

$$
a^{(m)}_c = \frac{\exp\big((A^{(m)}_c)\mathbf{z}\big)}{\sum_{k=1}^{M}\exp\big((A^{(k)}_c)\mathbf{z}\big)}\quad (m=1,\dots,M)
$$

그리고 출력은 채널별로

$$
V_c = \sum_{m=1}^{M} a^{(m)}_c\cdot U^{(m)}_c
$$

가 된다. 이때 핵심은 softmax로 인해 각 채널에서 branch 가중치의 합이 $1$로 정규화되고, 그 정규화된 가중치가 입력에서 유도된 $\mathbf{z}$의 함수이므로 입력 스케일 변화에 따라 동적으로 움직인다는 점이다.

#### Squeeze-Excitation(SE)와의 관계
Fuse의 global average pooling과 reduction ratio 설계는 **SENet**의 _squeeze/excitation_ 과 유사하다. 하지만 SENet은 채널을 재보정하는 스칼라 $s_c$를 만들고, SK는 스케일 branch들 사이에서 attention을 분배한다. 즉, 둘 다 전역 요약을 gating에 쓰지만, SENet은 채널 축 자체를 스케일링하고, SK는 **커널 스케일 선택**을 한다는 점에서 목표가 다르다.

#### SK Convolution 의사코드
논문 흐름을 의사코드로 정리하면 다음과 같다(2-branch 케이스).

```text
Algorithm: Selective Kernel (SK) convolution (2-branch)
Input: X
Branches: U1 = F1(X), U2 = F2(X)                     # Split
U = U1 + U2                                          # Fuse (Eq. 1)
s = GlobalAvgPool(U)                                 # (Eq. 2)
z = ReLU(BN(W s))                                    # (Eq. 3), d = max(C/r, L) (Eq. 4)
weights = Softmax([A z, B z])                        # channel-wise (Eq. 5)
V = weights[0] * U1 + weights[1] * U2                # Select (Eq. 6)
return V
```

---

## 4️⃣ 네트워크 아키텍쳐와 복잡도

### 🔹 ResNeXt-Style Bottleneck에 SK Unit 통합
논문은 SKNet을 ResNeXt 기반으로 구성한다. 이유는 grouped convolution으로 효율이 좋고, 당시 SOTA 수준의 backbone이었기 때문이다. SKNet은 ResNeXt bottleneck의 3×3(혹은 대형 커널) 위치를 SK convolution으로 대체해, 블록이 스케일을 선택하도록 만든다.

즉, 하나의 **SK unit**은 대략 다음 구조다.

1. 1×1 conv로 채널을 줄이거나(width를 조절)  
2. SK convolution(Split–Fuse–Select)으로 multi-scale 공간 정보를 선택적으로 통합  
3. 1×1 conv로 채널을 복원(expansion)  
4. Residual add + ReLU

논문은 이 구조를 Table 1에서 ResNeXt-50(32×4d), SENet-50, SKNet-50을 나란히 놓고 보여준다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/7e764e34-3be2-4039-abe0-44d3811a981c/image.png" width="70%">
</p>

#### Table 1의 핵심 메시지
Table 1의 비교는 단지 구조 나열이 아니라, SE와 SK를 ResNeXt backbone 위에 올렸을 때의 비용 증가가 얼마나 되는지를 보여준다. 논문은 SKNet-50이 ResNeXt-50 대비 파라미터는 대략 10% 수준, 연산은 5% 수준만 증가한다고 설명한다. 즉, 단순히 모델을 크게 만든 것이 아니라, **비용 증가를 제한한 상태에서 선택 메커니즘을 추가**했다는 주장이다.

#### Table 1의 복잡도 수치
논문 Table 1 하단에는 세 모델의 총 파라미터 수와 GFLOPs가 함께 제시된다.

| 모델 | #P | GFLOPs |
|---|---:|---:|
| ResNeXt-50 (32×4d) | 25.0M | 4.24 |
| SENet-50 | 27.7M | 4.25 |
| SKNet-50 | 27.5M | 4.47 |

여기서 SENet-50은 파라미터는 증가하지만 GFLOPs는 거의 증가하지 않고, SKNet-50은 GFLOPs도 소폭 증가한다. 이는 SE가 채널 게이팅(주로 1×1/FC 기반)이라 공간 해상도에 덜 민감한 반면, SK는 실제로 여러 커널 branch의 conv를 수행하기 때문에 연산이 더 늘 수 있다는 직관과 맞는다.

#### SKNet-26, SKNet-101 언급의 의미
논문은 SKNet-50만 제시하지 않고, stage별 블록 수를 바꾼 SKNet-26(`{2,2,2,2}`)과 SKNet-101(`{3,4,23,3}`)도 함께 언급한다. 이는 SK가 특정 깊이에만 유효한 트릭이 아니라, 깊이를 늘려도 같은 메커니즘을 유지할 수 있음을 암시한다. 그리고 Fig. 2의 파라미터 효율 곡선에서도 이 세 모델이 함께 쓰인다.

### 🔸 SK Convolution의 하이퍼파라미터
논문은 SK convolution을 결정짓는 핵심 하이퍼파라미터로 세 가지를 꼽는다.

- $M$: branch(경로) 수, 즉 선택 가능한 커널 스케일의 개수  
- $G$: grouped convolution의 그룹 수(ResNeXt의 cardinality와 대응)  
- $r$: Fuse 단계의 reduction ratio(Eq.(4))

논문에서 대표 설정으로 SK$[2, 32, 16]$을 제시한다. $M=2$는 2-branch 케이스이고, $G=32$는 ResNeXt-50(32×4d)의 cardinality와 같은 맥락이며, $r=16$은 gating 네트워크의 비용을 줄이기 위한 값이다.

#### $M=2$ 기본 설정의 근거
Branch를 늘리면 표현력은 늘 수 있지만, 비용도 증가한다. 논문은 Table 7에서 $M=2$와 $M=3$을 비교한 결과, $M=3$의 이득이 크지 않고 효율이 나빠질 수 있다고 보고한다. 그래서 성능-효율 트레이드오프 관점에서는 $M=2$가 기본 선택이 된다는 흐름으로 논증한다.

#### $G$와 $r$의 역할
$G$는 branch convolution의 그룹 수를 의미하고, ResNeXt의 cardinality와 같은 성격을 갖는다. 그룹 수가 커지면 채널 혼합은 제한되지만 비용이 줄어들 수 있다. 따라서 SK에서 더 큰 커널(혹은 더 큰 dilation)을 쓰더라도 $G$를 조절해 비용을 맞추는 전략이 가능하다(Table 6에서 5×5/7×7 케이스가 그룹 수 증가와 함께 제시된다).

$r$은 Fuse 단계의 bottleneck 차원 $d$를 줄이는 reduction ratio다. 즉, 선택을 계산하는 네트워크의 비용과 일반화 성질을 조절한다. $r$이 작으면(덜 줄이면) 선택 네트워크의 용량이 커지고 비용이 늘며, $r$이 크면(더 줄이면) 비용은 줄지만 선택이 부정확해질 수 있다. 논문은 이 균형을 위해 기본값 $r=16$을 쓰고, $d$가 너무 작아지지 않도록 최소 차원 $L$을 둔다.

---

## 5️⃣ ImageNet 실험

### 🔹 ImageNet 학습/평가 프로토콜
논문은 ImageNet-2012(1.28M train, 50K val, 1K classes)에서 top-1 error를 기준으로 비교한다. 데이터 증강은 표준적인 random resized crop(224×224)과 horizontal flip을 사용하고, 입력 정규화는 mean subtraction을 언급한다. 또한 label smoothing을 사용한다.

최적화는 synchronous SGD($m=0.9$), batch size $256$, weight decay $1e-4$(대형 모델)로 학습하며, $lr=0.1$에서 시작해 30 epoch마다 10배 감소, 총 100 epoch 학습을 한다고 서술한다. 경량 모델 학습에서는 weight decay를 $4e-5$로 낮추고, scale augmentation을 덜 aggressive하게 조정하는 등 underfitting을 완화하는 설정을 사용한다.

평가는 centre crop을 사용하며, $224×224$ 또는 $320×320$ crop 결과를 보고한다. 논문은 ImageNet 결과를 기본적으로 3회 평균이라고 언급한다.

#### 모델 규모에 따른 학습 레시피 분리 근거
논문은 경량 모델(lightweight models)에서는 weight decay를 $4e-5$로 낮추고, 데이터 전처리에서 scale augmentation을 덜 aggressive하게 조정한다고 말한다. 그 이유를 논문은 underfitting/overfitting 관점에서 설명한다. 작은 모델은 파라미터/연산 예산이 작아서 overfitting보다 underfitting 문제가 더 두드러질 수 있고, 따라서 정규화를 과하게 걸면 학습이 더 어려워질 수 있다.

이 디테일은 Table 4를 해석할 때 중요하다. 논문은 단순히 SE/SK 모듈을 붙였다는 사실뿐 아니라, 그 모듈이 실제로 경량 네트워크에서 유효하게 동작하도록 학습 레시피도 함께 맞췄음을 전제하고 있다.

#### 224× 및 320× 평가 병행 근거
해상도를 키우면 성능이 오를 수 있지만 연산도 늘어난다. 논문은 $224×$와 $320×$ 결과를 나란히 두어, 구조적 차이(SK의 선택 메커니즘)가 작은 해상도에서도 유효한지, 큰 해상도에서도 유지되는지를 함께 보여주려 한다. 또한 SKNet이 단순히 해상도 증가에 기대는 것이 아니라, 선택 메커니즘 자체로 이득을 얻는지 확인하는 장치로도 볼 수 있다.

### 🔸 SOTA 비교: Attention 기반 모델 대비 성능
논문은 Table 2에서 ResNeXt, Inception, BAM/CBAM, SENet, 그리고 SKNet을 유사한 복잡도 선에서 비교한다. Table 2 전체를 그대로 재구성하기보다는, 논문이 가장 강조하는 축(ResNeXt backbone 대비 SENet/SKNet 개선)을 중심으로 핵심 숫자를 뽑아보면 다음과 같다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/e4ef799d-7cee-4f1f-811a-43f8d6674d66/image.png" width="40%">
</p>

#### Table 2 핵심 발췌: ResNeXt-50 Vs SENet-50 Vs SKNet-50
| 모델 | $224×$ top-1 err | $320×$ top-1 err |
|---|---:|---:|
| `ResNeXt-50 (32×4d)` | $22.23$ | $21.05$ |
| `SENet-50` | $21.12$ | $19.71$ |
| **`SKNet-50 (ours)`** | $20.79$ | $19.32$ |

이 비교가 주는 메시지는 단순하다. 같은 계열 backbone에서, SE도 성능을 끌어올리지만, SK는 커널 스케일 선택이라는 다른 축의 메커니즘으로 추가 개선을 만든다. 특히 $224×$ 기준으로 ResNeXt-50 대비 SKNet-50은 1.44%p의 top-1 error 개선을 보이며, 이는 Table 3의 depth/width/cardinality 증가 대비 개선과 비교될 때 더 설득력을 갖는다.

#### Table 2 핵심 발췌: ResNeXt-101 Vs SENet-101 Vs SKNet-101
| 모델 | $224×$ top-1 err | $320×$ top-1 err |
|---|---:|---:|
| `ResNeXt-101` | $21.11$ | $19.86$ |
| `SENet-101` | $20.60$ | $19.42$ |
| **`SKNet-101 (ours)`** | $20.19$ | $18.40$ |

깊은 모델에서도 같은 경향이 유지된다. 즉, SK의 선택 메커니즘은 얕은 모델의 보정 트릭이 아니라, 더 깊고 강한 backbone에서도 유효한 구조 변화로 작동한다는 것이 논문 주장이다.

논문 본문은 SKNet-50이 ResNeXt-101보다 훨씬 작은 비용으로도 더 나은 성능을 낼 수 있다고 강조한다. 여기서 논점은, 단순히 깊이를 늘리는 것과 달리, **스케일 선택 메커니즘이 매우 효율적**이라는 것이다.

#### Table 2 전체가 전달하는 배경 메시지
Table 2에는 Inception 계열(InceptionV3/V4, Inception-ResNetV2), BAM/CBAM 같은 attention 모듈, SENet, 그리고 SKNet이 함께 들어간다. 논문의 의도는 SK를 단지 ResNeXt 내부의 변형으로만 보지 않고, 당시 자주 쓰이던 attention 기반 개선 방향들과 비교했을 때도 경쟁력이 있다는 점을 보여주는 것이다.

또한 $224×$와 $320×$ 결과를 함께 보고하는 이유는, 단일 crop이라는 제한된 조건에서의 구조 비교뿐 아니라 더 큰 입력에서의 상한 성능에서도 SK의 개선이 유지되는지 확인하기 위함이다. Table 2에서 SKNet의 개선이 두 설정에서 모두 유지된다는 점은, SK가 해상도 증가에만 기대는 트릭이 아니라는 논증으로 이어진다.

### 🔹 SK Vs. Depth/Width/Cardinality: 공정 비교
SKNet은 여러 branch를 쓰기 때문에 baseline ResNeXt 대비 파라미터/연산이 약간 늘어난다. 논문은 이를 공정하게 비교하기 위해 ResNeXt의 복잡도를 depth/width/cardinality를 조정해 SKNet 수준으로 올린 변형들과 비교한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/a368458c-1ec4-4a14-bb30-94c09e0d3716/image.png" width="40%">
</p>

이 비교에서 가장 중요한 관찰은 다음이다.

1. ResNeXt의 복잡도를 늘리면 error는 조금 줄지만, 개선 폭이 작다(0.1~0.23%p 수준).  
2. 반면 SKNet은 같은 수준의 비용 증가로 훨씬 큰 개선을 만든다(1.44%p).  

논문은 이를 통해 SK convolution이 단순한 용량 증가가 아니라, **구조적으로 더 효율적인 개선 방향**이라고 주장한다.

#### Table 3의 공정 비교로서의 의의
SKNet은 branch가 추가되기 때문에 baseline 대비 비용이 늘 수밖에 없다. 이때 단순히 ResNeXt-50과 SKNet-50을 비교하면, 성능 개선이 구조 때문인지 용량 증가 때문인지 반박 여지가 생긴다. Table 3은 이 반박을 줄이는 장치다. ResNeXt 쪽도 비용을 맞추기 위해 **더 넓게, 더 깊게, 더 많은** cardinality를 쓰는 변형을 만들고, 그럼에도 개선 폭이 작다는 사실을 보여준다.

즉, 논문은 성능-비용 관점에서 SK의 선택 메커니즘이 단순한 용량 증가보다 효율적이라고 주장하고, Table 3은 그 주장을 지탱하는 핵심 근거로 기능한다.

### 🔸 파라미터 효율
논문은 Fig. 2에서 파라미터 수 대비 top-1 error를 점으로 찍어, `SKNet-26/50/101`이 ResNet/ResNeXt/DenseNet/DPN/SENet 등과 비교해 파라미터를 더 효율적으로 사용한다고 주장한다. 특히 비슷한 top-1 error를 얻기 위해 `SKNet-101`이 DPN-98보다 더 적은 파라미터가 필요하다는 식의 비교를 제시한다.

이 그림의 핵심은 절대 숫자보다도, SKNet이 단지 특정 구성에서만 좋아지는 것이 아니라, 모델 규모가 달라져도 일관된 효율을 보인다는 주장에 있다.

---

## 6️⃣ 경량 모델 & CIFAR 실험

### 🔹 경량 모델에서의 효과: ShuffleNetV2 + SK
논문은 SK가 대형 모델에만 통하는 장치인지 확인하기 위해 **ShuffleNetV2에 SK를 붙인 변형**을 실험한다. Table 4는 0.5×와 1.0× 설정에서 baseline, 자체 구현, SE 추가, SK 추가를 비교하고, top-1 error, MFLOPs, 파라미터 수를 함께 보고한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/c9bda198-5fae-4d46-9320-05d00053b504/image.png" width="40%">
</p>

여기서 중요한 점은, SK가 경량 setting에서도 SE 대비 추가 이득을 만들 수 있다는 것이다. 논문은 이를 low-end 디바이스에서도 SK의 적용 가능성이 있다는 근거로 사용한다.

#### 경량 모델에서 SK 효과가 두드러지는 배경
SE는 채널 축을 재보정해 representation을 강화하지만, 공간적으로 더 넓은 문맥을 직접 만들어 주지는 않는다. 반면 SK는 여러 커널 스케일의 branch를 만들어 실제로 **문맥 크기 자체를 선택**하게 만든다. 경량 모델은 채널 폭/깊이가 제한되어 한 번의 공간 변환이 갖는 정보량이 더 중요해질 수 있고, 이때 문맥 크기를 조절하는 장치가 더 큰 체감 효과를 낼 수 있다.

또한 Table 4에서 SK의 MFLOPs 증가가 SE보다 약간 더 나타나는 구간이 있는데(예: 1.0×에서 $145.66$ vs $141.73$), 그럼에도 top-1 error가 더 내려간다는 점은 SK의 비용-효율이 경량 setting에서도 유지될 수 있음을 시사한다.

#### 경량 모델에서 스케일 선택의 중요성
모바일 네트워크는 연산 예산 때문에 채널 폭이나 깊이가 제한되고, 표현력이 병목이 되기 쉽다. 이때 SK는 새로운 큰 backbone을 만드는 대신, 제한된 연산 내에서 multi-scale branch를 만들고 선택한다. 즉, 같은 예산에서 정보를 더 효율적으로 사용하는 방향으로 개선이 나타날 수 있다.

### 🔸 CIFAR 실험: 작은 입력에서도 유지되는 개선
논문은 CIFAR-10/100에서도 SKNet이 유효한지 확인한다. 설정은 **ResNeXt-29(16×32d) 스타일 backbone** 위에 SK convolution을 넣은 `SKNet-29`를 만들고, 동일 backbone 위에 SE를 넣은 `SENet-29`와 비교한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/b0597452-6264-4480-880b-82fb509775fa/image.png" width="40%">
</p>

논문은 `SKNet-29`가 `SENet-29`보다 **더 적은 파라미터로 더 좋은 성능**을 낼 수 있다는 점을 강조한다. 즉, 스케일 선택 메커니즘이 대규모 ImageNet에서만 통하는 것이 아니라, 작은 입력/작은 데이터셋에서도 일반적으로 도움이 될 수 있다는 주장이다.

#### CIFAR 설정의 구조적 특징: 작은 입력에서의 스케일 선택
논문은 CIFAR에서의 네트워크 구성을 간단히 설명한다. $32×32$ 입력을 대상으로, 초기에 단일 3×3 conv를 두고 그 뒤에 3개의 stage를 쌓으며, 각 stage는 3개의 residual block을 갖고 그 block 내부에 SK convolution을 넣는다. 즉, ImageNet의 대형 SKNet과는 깊이/폭이 다르지만, 핵심 메커니즘(_Split–Fuse–Select_)은 동일하게 유지된다.

또한 논문은 `ResNeXt-29 16×64d`와 비교해 `SKNet-29`가 더 적은 파라미터로도 비슷하거나 더 나은 성능을 낼 수 있다고 언급한다. 이는 SK가 단순히 채널을 늘리거나 깊이를 늘리는 용량 증가와 다른 개선 방향임을 작은 데이터셋에서도 확인하려는 시도로 이해할 수 있다.

---

## 7️⃣ Ablation 연구

### 🔹 Dilation/Kernel 설정
논문은 SKNet-50의 2-branch 설정에서, 첫 번째 branch는 3×3($D=1, G=32$)로 고정하고, 두 번째 branch를 여러 방식으로 바꿔본다. Table 6은 그 결과를 정리한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/a4d2b86e-03b2-40a6-a7a1-94038507f05a/image.png" width="40%">
</p>

Table 6의 핵심은 다음이다.

- 3×3에 dilation을 키워 **RF를 키우는 방식**이 효율적으로 동작할 수 있다.  
- 최적 설정은 대체로 **5×5**(를 dilation으로 근사한 것)에 해당하는 branch가 된다.  

이 결과를 직관적으로 해석하면, 첫 번째 branch가 3×3이라면 두 번째 branch는 더 큰 RF를 제공하는 쪽(대략 5×5)이 가장 자연스럽게 보완 관계를 만들고, 그 조합이 SK의 선택 메커니즘에 의해 가장 효율적으로 사용될 수 있다는 것이다.

#### Table 6에서 D와 G 동시 고려 배경
논문은 RF를 키우는 방법으로 **dilation을 늘리는 방법과 커널 크기와 그룹 수를 같이 조절하는 방법**을 언급한다. 큰 커널은 비용이 커지므로, 그룹 수를 늘려 비용을 맞추는 전략이 사용된다. Table 6에서 5×5, 7×7 케이스는 그룹 수가 $64$, $128$로 늘어나 있는데, 이는 복잡도를 비슷하게 유지하면서 커널 스케일을 바꾸려는 의도로 해석할 수 있다.

반대로 dilation 기반 근사($3×3, D=2/3$)는 커널 크기를 키우지 않으면서 RF를 키우는 방식이라, 그룹 수를 고정해도 비용이 크게 달라지지 않는다. 논문이 dilated 3×3을 기본 선택으로 삼는 이유가 여기서 다시 확인된다.

#### 5×5 근사가 7×7 대비 유리할 수 있는 조건
RF를 더 키우면 항상 좋은 것은 아니다. 너무 큰 RF는 불필요한 배경 정보를 섞어 noise가 될 수 있고, 비용도 늘어난다. Table 6은 이 균형점이 **대략 5×5 근사**에서 가장 잘 맞았음을 보여준다. 이는 이후 Table 7에서 3-branch로 커널을 더 늘렸을 때 개선 폭이 작아지는 관찰과도 일관된다.

### 🔸 Kernal 조합과 SK Attention의 기여
Table 7은 세 가지 커널 후보를 정의하고, 어떤 조합을 사용할지, 그리고 그 조합을 단순 합으로 aggregation할지, 아니면 SK attention으로 선택할지를 비교한다.

- $K3$: 표준 3×3  
- $K5$: dilation 2의 3×3(5×5 근사)  
- $K7$: dilation 3의 3×3(7×7 근사)

논문은 Table 7에서 SK 체크가 있으면 (Fig. 1의 $\mathbf{V}$처럼) attention 가중합을 하고, 체크가 없으면 (Fig. 1의 $\mathbf{U}$처럼) 단순 합을 한다고 설명한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/b5143f79-72b9-42bb-90db-16d99e524d28/image.png" width="40%">
</p>

#### Table 7에서 논문이 도출하는 결론
논문은 Table 7을 통해 **세 가지 관찰**을 정리한다.

1. $M$이 늘어날수록(경로 수가 늘수록) 대체로 error가 내려간다.  
2. 같은 $M$이라면, 단순 합보다 SK attention이 항상 더 좋다.  
3. $M=2$에서 $M=3$으로 갈 때, SK attention 하에서는 이득이 작다(20.79 → 20.76). 그래서 효율을 위해 $M=2$를 선호한다.

이 결론은 SKNet의 성능이 단지 여러 커널을 병렬로 둬서 생긴 것이 아니라, **adaptive selection 메커니즘 자체**가 중요한 기여를 한다는 논증으로 이어진다.

#### Table 7이 주는 추가 직관
Table 7에는 $M=1$(단일 경로) 설정도 포함되며, 이때 K3/K5/K7을 단독으로 썼을 때의 성능이 함께 제시된다. 논문 수치(224× 기준)는 다음과 같다.

| 설정(M=1) | top-1 err |
|---|---:|
| K3만 사용(ResNeXt-50 baseline) | $22.23$ |
| K5만 사용 | $25.14$ |
| K7만 사용 | $25.51$ |

즉, 큰 RF를 무작정 키운 단일 커널은 오히려 성능을 크게 망칠 수 있다. SKNet의 핵심은 큰 커널이 항상 유리하다는 주장이 아니라, 입력과 표현 수준에 따라 **필요할 때 큰 스케일을 선택적으로 쓰는 것**이라는 점을 이 결과가 오히려 더 분명하게 만들어 준다.

이 관점에서 보면, SKNet이 multi-kernel을 두는 이유는 단지 표현력을 늘리기 위해서가 아니라, **작은/큰 문맥이 모두 필요한 상황에서 그 비중을 조절하기 위해서다**. 그리고 그 조절이 단순 합이 아니라 attention 기반 선택으로 구현되어, 같은 커널 조합에서도 더 나은 성능을 만든다는 것이 Table 7 전체의 논지다.

따라서 Table 7은 SKNet을 이해할 때 가장 중요한 경계선을 그어준다. 큰 커널 자체가 정답이 아니라, 큰 커널을 포함한 여러 선택지 사이에서 입력 조건부로 적절히 선택하는 메커니즘이 정답에 가깝다는 것이다. 이 점이 SKNet을 단순한 커널 스케일 확장이 아니라 선택 기반 구조로 읽어야 하는 이유다.

---

## 8️⃣ 분석과 해석

### 🔹 객체 스케일 변화에 따른 Attention의 변화
논문은 SK가 실제로 RF 크기를 적응시키는지 확인하기 위해, ImageNet validation 이미지에서 중심 객체를 $1.0×$에서 $2.0×$까지 점진적으로 확대하는 입력 변형을 만든다. 구체적으로는 중앙 크롭 후 리사이즈로 객체를 크게 만들고, 배경은 상대적으로 줄어들게 한다. 이 과정은 Fig. 3의 상단 예시로 제시된다.

그 다음 특정 SK unit에서 5×5 branch에 대한 attention 값을 채널별로 계산하고, 객체가 커질수록 attention이 어떻게 변하는지를 본다. Fig. 3a,b는 샘플 이미지 2개에 대한 결과를, Fig. 3c는 validation 전체 평균을 보여준다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/c80f2437-531e-4583-95d8-8d9dee54bbcf/image.png" width="90%">
</p>

#### Fig. 3의 핵심 관찰
논문은 대부분의 채널에서, 객체가 커질수록 5×5 branch의 attention이 증가한다고 보고한다. 이는 큰 객체일수록 더 넓은 문맥을 필요로 하거나, 더 큰 RF를 쓰는 것이 유리하다는 직관과 일치한다. 따라서 SK가 입력에 따라 RF 크기를 조절하는 방향으로 동작한다는 논증이 가능해진다.

#### Fig. 3 플롯 해석 요점
논문은 두 가지 형태의 요약을 같이 본다.

1. 특정 SK unit에서 채널별로 5×5 attention이 어떻게 변하는가(채널 축)  
2. 여러 SK unit에 대해 (큰 커널 - 작은 커널) 평균 차이가 어떻게 변하는가(깊이 축)  

채널 축의 플롯은 많은 채널을 한 번에 보여주기 어려워, 논문에서는 연속된 채널을 묶어 평균내는 형태로 시각화했다고 설명한다. 즉, 한 점이 개별 채널 하나를 의미하기보다는 일정 구간의 채널 평균을 의미한다. 이 시각화 선택은, 채널별 attention이 노이즈처럼 보이지 않고 전체적인 경향(객체 크기가 커지면 큰 커널 가중이 올라간다)이 드러나도록 만든다.

깊이 축의 플롯(오른쪽)은, 같은 객체 스케일 변화가 네트워크의 어느 stage에서 더 강하게 반영되는지를 보여준다. 이는 이후 Fig. 4의 클래스별 분석과 자연스럽게 이어진다.

### 🔸 깊이와 클래스 관점의 패턴
논문은 채널 평균의 attention 차이(큰 커널 minus 작은 커널)를 SK unit별로 계산해, 깊이에 따라 패턴이 어떻게 달라지는지 본다. Fig. 3의 오른쪽 플롯과 Fig. 4가 이 분석을 담당한다.

Fig. 4는 1,000개 클래스 각각에 대해 평균 attention 차이를 계산해 분포를 보여준다. 논문은 그 결과가 모든 클래스에 대해 일관되게 나타난다고 주장한다. 즉, 객체 스케일이 커지면 저/중층 SK unit에서 큰 커널의 중요도가 일관되게 올라간다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/50ba8ba9-ea74-4abe-a025-460bcaaba9fa/image.png" width="70%">
</p>

#### 상위 계층에서 패턴이 약화되는 원인
논문은 깊은 층에서는 스케일 정보가 이미 feature vector에 부분적으로 인코딩되어 있고, 커널 크기 선택의 중요도가 상대적으로 줄어들 수 있다고 해석한다. 즉, 저/중층에서는 공간 스케일 선택이 직접적으로 중요하지만, 고층에서는 의미 수준의 표현이 더 중요해지면서 커널 스케일 선택의 효과가 포화될 수 있다는 것이다.

이 해석은 SK가 모든 층에서 똑같이 큰 커널을 선호하는 것이 아니라 입력과 층 깊이에 따라 달라질 수 있고, SK의 핵심 이득이 저/중층에서의 적응에 더 크게 걸려 있을 수 있다는 시사점을 준다.

#### 논문 동기와의 재연결
초기/중간 stage는 텍스처, edge, 중간 수준의 패턴 등 공간적 스케일이 직접적으로 중요한 표현을 많이 다룬다. 이 구간에서는 객체가 커지면 더 넓은 문맥을 보는 것이 유리하고, 따라서 큰 커널 branch로의 attention 이동이 자연스럽게 나타난다.

반면 매우 깊은 stage에서는 이미 공간 해상도가 줄어들고, 표현은 클래스 구분을 위한 고수준 조합 특징으로 압축된다. 이때는 커널 스케일 선택이 표현의 병목이 되기보다, 채널 조합 자체(어떤 의미 특징을 쓰는가)가 더 중요해질 수 있다. 논문이 말하는 패턴 소실은, SK가 항상 동일한 방식으로 작동한다기보다, 네트워크 깊이에 따라 스케일 선택의 역할이 달라질 수 있음을 보여주는 경험적 단서로 해석할 수 있다.

이 관찰은 처음에 제시한 RF 적응 동기와도 일관된다. 유효 RF는 어디서나 무작정 커지는 것이 아니라, 입력과 표현 수준에 따라 필요한 범위만큼 조절되는 것이 바람직하다. SK의 attention이 깊이에 따라 다른 동작을 보인다는 사실 자체가, 이 메커니즘이 단순한 **정적 편향이 아니라 입력 조건부 적응**임을 뒷받침한다.

---

## 💡 해당 논문의 시사점과 한계
SKNet의 의의는 multi-scale 정보를 같은 블록 안에 두는 것을 넘어, **그중 어떤 스케일을 쓸지를 입력에 따라 선택하도록** 만든 점이다. 특히 Split–Fuse–Select로 분해된 설계는 다음과 같은 장점을 준다.

1. 커널 스케일 선택이 모델 내부에서 명시적으로 드러난다(가중치, attention 분석 가능).  
2. 전역 요약 + softmax 선택이라는 단순한 메커니즘으로도 큰 성능 개선이 가능함을 보여준다.  
3. ResNeXt/ShuffleNetV2 같은 서로 다른 규모의 backbone에 적용해 일반성을 입증하려 한다(Table 2~5).  

또한 ablation(Table 6, 7)과 분석(Fig. 3, 4)은 SK가 단지 branch 수를 늘린 효과가 아니라, **선택 메커니즘 자체**가 중요하다는 논증 구조를 구성한다. 특히 Table 7의 naive sum 대비 SK attention의 일관된 개선은 설계 타당성을 강화한다.

#### 한계와 실무적 고려
1. SK는 스케일 선택 메커니즘이지만, 모든 상황에서 더 큰 RF가 좋은 것은 아니며, 커널 조합/branch 수/효율 설계가 중요하다. 논문도 $M=3$의 추가 이득이 작다는 점을 보고하며($M=2$ 선호), 이는 무조건 branch를 늘리는 것이 정답이 아님을 보여준다.

2. 논문이 제안하는 SK의 수식은 채널별로 branch attention을 다르게 줄 수 있도록 설계되어 있는데, 실제 구현에서는 효율을 위해 채널 공유 attention으로 단순화될 가능성도 있다. 이 경우 논문이 의도한 채널별 스케일 선택이 어느 정도까지 유지되는지에 대한 추가 검증이 필요할 수 있다(이 점은 아래 Lucid 구현을 해석할 때도 중요하다).

3. Fig. 3/4의 분석은 입력 스케일 변화에 대한 설득력 있는 관찰을 제공하지만, 그것이 곧바로 모든 다운스트림 과제(예: detection/segmentation)로 일반화된다는 것을 이 논문 자체가 직접 입증하진 않는다. 다만 논문은 경량 모델까지 포함한 폭넓은 실험으로, 최소한 분류 설정에서는 규모를 가리지 않고 도움이 된다는 방향의 일반성을 확보하려 한다.

#### 후속 설계 관점에서의 의의
SKNet 이후에도 다양한 형태의 동적 convolution, attention, 그리고 입력 조건부 연산이 등장했다. 이 흐름을 지금 관점에서 보면, SKNet의 핵심 기여는 스케일 선택을 분해된 연산자(_Split–Fuse–Select_)로 정의하고, 그 선택을 softmax attention으로 명시화했으며, 그 선택이 실제로 입력 변화(객체 스케일)에 따라 움직인다는 실험적 분석까지 제공했다는 점이다.

즉, 단순히 정확도를 올린 모델이 아니라, 어떤 축에서 동적 적응을 만들 것인지(커널 스케일), 그 적응을 어떻게 구현할 것인지(softmax 기반 gating), 그리고 그 적응이 실제로 일어나는지(Fig. 3, 4)까지 하나의 논증 구조로 연결한 논문이라고 볼 수 있다.

#### 실무 적용 체크리스트
논문이 제공하는 결과/ablation을 바탕으로, 실무에서 SK를 적용할 때 고려할 포인트를 정리하면 다음과 같다.

- 목표가 RF 적응이라면, 서로 다른 스케일의 branch가 실제로 필요하다(Table 6의 결과가 그 근거다).
- Branch 수 $M$은 무작정 늘리기보다, $M=2$부터 시작하는 것이 효율적으로 유리할 수 있다(Table 7에서 $M=3$의 추가 이득이 작다).
- 큰 커널은 dilation으로 근사하는 편이 성능/효율 면에서 유리할 수 있다(Table 6의 비교).
- 선택 네트워크의 bottleneck 차원($r$, $L$)은 비용과 선택 정확도를 함께 좌우한다(Eq.(4)).
- 공정 비교를 위해서는 동일 학습 레시피/평가 프로토콜에서 baseline 대비 개선을 읽는 것이 안전하다(Table 2, 3의 구성 의도).
- 경량 모델에서는 underfitting 문제가 두드러질 수 있으므로, 학습 레시피(정규화/증강)도 함께 조정해야 한다.

---

## 👨🏻‍💻 SKNet 구현하기
이 파트에선 [`lucid`](https://github.com/ChanLumerico/lucid/tree/main)라이브러리를 이용하여 SK 모듈을 구현한 [`nn.SelectiveKernel`](https://github.com/ChanLumerico/lucid/blob/main/lucid/nn/fused.py#L155)을 먼저 소개하고, 그 다음 [`sknet.py`](https://github.com/ChanLumerico/lucid/blob/main/lucid/models/imgclf/sknet.py)가 SKNet을 ResNet 스타일 빌더로 제공하는 방식을 설명한다. SKNet은 논문에서처럼 하나의 독립 backbone이라기보다 ResNeXt/ResNet 계열 residual block 내부의 3×3 변환을 SK로 대체하는 형태이므로, Lucid 코드도 기존 ResNet 빌더를 재사용하면서 block을 SK 버전으로 바꿔 끼우는 구조를 택한다.

### 1️⃣ `nn.SelectiveKernel`: Split–Fuse–Select의 구현
Lucid의 `SelectiveKernel`은 branch conv들을 만들고(`branches`), branch 출력의 합으로 attention 입력을 만들고(`branch_outs.sum(axis=1)`), attention 네트워크로 branch별 점수를 만든 뒤(`self.attention`), softmax로 가중치를 정규화하고(`self.softmax`), branch 출력의 가중합으로 최종 출력을 만든다.

```python
class SelectiveKernel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list[int],
        stride: int = 1,
        padding: _PaddingStr | None = None,
        groups: int = 1,
        reduction: int = 16,
    ) -> None:
        super().__init__()

        branches = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=ks,
                stride=stride,
                padding=(ks // 2 if padding is None else padding),
                groups=groups,
                bias=False,
            )
            for ks in kernel_sizes
        ]
        self.branches = nn.ModuleList(branches)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                out_channels, out_channels // reduction, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(out_channels // reduction),
            nn.ReLU(),
            nn.Conv2d(
                out_channels // reduction,
                len(kernel_sizes),
                kernel_size=1,
                bias=False,
            ),
        )

        self.softmax = nn.Softmax(axis=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("Only supports 4D-tensors.")

        branch_outs = [branch(x) for branch in self.branches]
        branch_outs = lucid.stack(branch_outs, axis=1)

        att_scores = self.attention(branch_outs.sum(axis=1))
        att_weights = self.softmax(att_scores).unsqueeze(axis=2)

        out = (branch_outs * att_weights).sum(axis=1)
        return out
```

#### 논문 수식과의 대응
이 구현은 큰 흐름에서는 논문과 매우 유사하다.

- **Split**: `self.branches`가 kernel_sizes에 따라 여러 conv branch를 만든다.  
- **Fuse**: `branch_outs.sum(axis=1)`로 branch를 합친 뒤 attention 네트워크의 입력으로 쓴다(전역 요약은 `AdaptiveAvgPool2d`).  
- **Select**: `Softmax(axis=1)`로 branch 축에 대한 가중치를 만들고, branch 출력의 가중합을 만든다.  

다만 논문 Eq.(5), (6)은 채널별 가중치($a_c, b_c$)를 정의해 채널마다 다른 스케일 선택이 가능하도록 설계되어 있다. 반면 Lucid 구현은 `len(kernel_sizes)` 채널의 attention score를 만들고 이를 softmax 하므로, **branch 가중치가 채널 전체에 공유되는 형태**로 동작한다(가중치 텐서가 채널 축을 갖지 않고 브로드캐스팅으로 적용된다). 즉, Lucid 구현은 Split–Fuse–Select 패턴을 유지하되, 논문이 강조한 채널별 선택을 효율을 위해 단순화한 변형으로 해석할 수 있다.

또한 논문은 큰 커널을 dilation으로 근사하는 설정을 기본으로 소개하지만, Lucid의 `SelectiveKernel`은 `kernel_size=ks`를 직접 사용한다. 따라서 Lucid에서 `kernel_sizes=[3,5]`는 실제 3×3과 5×5 branch를 의미한다.

#### `SelectiveKernel.forward`의 텐서 형태 추적
코드가 하는 일을 텐서 shape 관점에서 정리하면 다음과 같다(배치 $N$, 채널 $C$, 공간 $H\times W$, branch 수 $M$).

1. `branch_outs = [branch(x) for branch in self.branches]`  
   - 각 branch 출력은 (N, C, H, W)
2. `branch_outs = lucid.stack(branch_outs, axis=1)`  
   - (N, M, C, H, W)로 쌓인다(새로운 branch 축이 axis=1)
3. `att_scores = self.attention(branch_outs.sum(axis=1))`  
   - `sum(axis=1)`으로 branch 축을 합치면 (N, C, H, W)
   - Attention은 GAP → 1×1 conv → BN → ReLU → 1×1 conv로 (N, M, 1, 1)을 만든다
4. `att_weights = self.softmax(att_scores).unsqueeze(axis=2)`  
   - Softmax는 axis=1(branch 축)에서 수행된다
   - `unsqueeze(axis=2)`로 (N, M, 1, 1, 1) 형태가 되고 C/H/W로 브로드캐스팅된다
5. `out = (branch_outs * att_weights).sum(axis=1)`  
   - Branch 축 가중합으로 (N, C, H, W) 출력이 된다

즉, Lucid 구현은 공간 위치별 선택이 아니라, 각 branch 전체에 대해 전역적으로 하나의 weight를 만든 뒤 그 weight로 branch를 섞는다. 논문이 말하는 채널별 선택과는 차이가 있지만, 스케일 축의 선택이라는 핵심 흐름은 유지된다.

#### Reduction과 Eq.(4)의 차이
논문은 $d = \max(C/r, L)$로 bottleneck 차원을 정의하고, 최소 차원 $L$을 둔다. 반면 Lucid 구현은 `out_channels // reduction`으로 채널을 줄이고, $L$ 같은 하한은 두지 않는다. 따라서 채널이 작은 구간에서는 bottleneck이 매우 작아질 수 있고, 이는 논문과는 다른 설계 선택이다. 리뷰에서는 이를 논문과의 완전 동일 구현으로 해석하기보다, SK 패턴을 구현한 변형으로 이해하는 것이 안전하다.

#### 논문식 채널별 선택과 Lucid 구현을 비교할 때의 주의점
논문 Eq.(5), (6)은 채널마다 다른 선택 가중치를 만들 수 있도록 설계되어 있고, 그 채널별 선택이 Fig. 3/4 분석에서 핵심 관찰 대상이 된다. Lucid 구현은 branch 축 softmax라는 큰 흐름은 동일하지만, 실제 가중치는 채널에 공유되는 형태이므로, 논문이 관찰한 채널별 미세한 선택 패턴을 그대로 재현한다고 보긴 어렵다.

그래도 이 구현은 SK의 핵심 아이디어(여러 스케일의 변환을 만들고, 전역 요약 기반의 softmax로 섞는다)를 깔끔하게 보여 준다. 따라서 Lucid 코드를 읽을 때는 스케일 선택이라는 구조적 패턴을 이해하는 데 초점을 두고, 논문과의 세부 차이(채널별/공유, dilation 근사 여부, bottleneck 하한 여부)는 구현 변형으로 받아들이는 편이 좋다.

### 2️⃣ `SKNet`: ResNet 빌더로 제공
`sknet.py`의 `SKNet` 클래스는 `ResNet`을 상속하고, block 생성 인자(`block_args`)로 kernel_sizes/base_width/cardinality를 전달하는 wrapper다.

```python
class SKNet(ResNet):
    def __init__(
        self,
        block: nn.Module,
        layers: list[int],
        num_classes: int = 1000,
        kernel_sizes: list[int] = [3, 5],
        base_width: int = 64,
        cardinality: int = 1,
        **resnet_args: Any,
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            block_args={
                "kernel_sizes": kernel_sizes,
                "base_width": base_width,
                "cardinality": cardinality,
            },
            **resnet_args,
        )
```

즉, 논문에서 말하는 SK unit의 설계는 Lucid에서는 block 클래스(`_SKResNetModule` 또는 `_SKResNetBottleneck`)로 구현되고, `SKNet`은 그 block을 ResNet 빌더에 끼워 넣는 방식으로 전체 네트워크를 만든다.

#### `_SKResNetModule`: `expansion=1` 계열 블록
`_SKResNetModule`은 1×1 conv로 width를 만든 뒤, `nn.SelectiveKernel`로 공간 변환을 수행하고, 마지막 1×1 conv로 out_channels로 돌린 뒤 residual add를 수행한다.

```python
class _SKResNetModule(nn.Module):
    expansion: ClassVar[int] = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        kernel_sizes: list[int] = [3, 5],
        cardinality: int = 1,
        base_width: int = 64,
    ) -> None:
        super().__init__()
        width = int(out_channels * (base_width / 64.0)) * cardinality

        self.conv1 = nn.ConvBNReLU2d(
            in_channels, width, kernel_size=1, stride=1, conv_bias=False
        )
        self.sk_module = nn.SelectiveKernel(
            width, width, kernel_sizes=kernel_sizes, stride=stride, groups=cardinality
        )
        self.bn2 = nn.BatchNorm2d(width)
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

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(self.bn2(self.sk_module(out)))
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
```

여기서 중요한 구현 포인트는 다음이다.

- Width 계산: `base_width`와 `cardinality`로 ResNeXt 스타일 width를 만든다.  
- SK 적용 위치: `conv1` 이후의 중간 width feature에 `sk_module`을 적용한다.  
- BN/ReLU 배치: `SelectiveKernel` 내부 branch는 conv만 수행하므로, 블록 바깥에서 `bn2`와 `relu`로 비선형을 준다.  
- Residual add: downsample이 있으면 identity를 바꾸고 더한다.  

#### 논문 SK Unit(1×1 → SK → 1×1)과의 대응
논문은 SK unit이 1×1 conv, SK convolution, 1×1 conv의 연쇄로 이루어진다고 설명한다(Section 3.2). Lucid 구현도 동일하게, `conv1`이 1×1, `sk_module`이 가운데 변환, `conv3`이 마지막 1×1 역할을 한다. 즉, ResNeXt bottleneck의 중심 3×3 변환 자리에 SK를 끼워 넣는다는 큰 그림은 그대로다.

다만 논문은 branch 변환이 grouped/depthwise conv + BN + ReLU로 구성된다고 서술하는 반면, Lucid의 `SelectiveKernel` branch는 conv만 수행하고 BN/ReLU는 블록 바깥에서 한 번만 적용된다. 이는 구현 단순화/효율화의 선택으로 볼 수 있고, 리뷰에서는 이 차이를 염두에 두고 Lucid 코드를 논문과 1:1로 동일한 구현이라고 단정하지 않는 편이 안전하다.

#### `_SKResNetBottleneck`: `expansion=4` 계열 블록
`_SKResNetBottleneck`은 기본 구조가 동일하되, `expansion=4`로 out_channels가 확장된다. 즉, ResNet-50+ 스타일 bottleneck에 해당한다.

```python
class _SKResNetBottleneck(nn.Module):
    expansion: ClassVar[int] = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        kernel_sizes: list[int] = [3, 5],
        cardinality: int = 1,
        base_width: int = 64,
    ) -> None:
        super().__init__()
        width = int(out_channels * (base_width / 64.0)) * cardinality

        self.conv1 = nn.ConvBNReLU2d(
            in_channels, width, kernel_size=1, stride=1, conv_bias=False
        )

        self.sk_module = nn.SelectiveKernel(
            width, width, kernel_sizes=kernel_sizes, stride=stride, groups=cardinality
        )
        self.bn2 = nn.BatchNorm2d(width)

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

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(self.bn2(self.sk_module(out)))
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
```

논문 관점에서는, bottleneck의 공간 변환 역할을 하는 중간 연산(원래 3×3 위치)을 SK로 대체했다는 점이 핵심이다. Lucid 구현도 동일하게, bottleneck의 가운데 변환을 `sk_module`로 두고 있다.

### 3️⃣ 모델 팩토리: `sk_resnet_*`, `sk_resnext_*`
마지막으로 `sknet.py`는 모델 엔트리들을 `@register_model`로 등록한다.

```python
@register_model
def sk_resnet_18(num_classes: int = 1000, **kwargs) -> SKNet:
    layers = [2, 2, 2, 2]
    return SKNet(_SKResNetModule, layers, num_classes, **kwargs)


@register_model
def sk_resnet_34(num_classes: int = 1000, **kwargs) -> SKNet:
    layers = [3, 4, 6, 3]
    return SKNet(_SKResNetModule, layers, num_classes, **kwargs)


@register_model
def sk_resnet_50(num_classes: int = 1000, **kwargs) -> SKNet:
    layers = [3, 4, 6, 3]
    return SKNet(_SKResNetBottleneck, layers, num_classes, **kwargs)


@register_model
def sk_resnext_50_32x4d(num_classes: int = 1000, **kwargs) -> SKNet:
    layers = [3, 4, 6, 3]
    return SKNet(
        _SKResNetBottleneck,
        layers,
        num_classes,
        cardinality=32,
        base_width=4,
        **kwargs,
    )
```

이 구성은 SENet의 Lucid 구현과 구조적으로 유사하다. 큰 틀의 ResNet 빌더는 그대로 두고, 블록을 SK 버전으로 바꿔 끼우면서, ResNeXt 설정(`cardinality/base_width`)을 인자로 넘겨 주는 방식이다. 따라서 Lucid에서 SKNet은 독립적인 신규 프레임워크가 아니라, 기존 ResNet 계열 코드에 잘 끼워 넣을 수 있는 attachment 성격의 구현으로 이해할 수 있다.

---

## ✅ 정리
**SKNet**은 입력 자극에 따라 뉴런의 유효 receptive field 크기를 적응시키는 것을 목표로, 여러 커널 스케일의 정보를 만들고 그중 어떤 스케일을 쓸지 _softmax attention으로 선택_ 하는 **SK convolution**을 제안한다. 

SK convolution은 **Split–Fuse–Select**로 구성되며, Fuse에서 전역 요약(Eq.(1)~(4))을 만들고 Select에서 채널별 attention으로 branch를 가중합(Eq.(5), (6))하는 것이 핵심이다. 논문은 이 메커니즘을 ResNeXt-style bottleneck에 통합해 SKNet을 구성하고(Table 1), ImageNet에서 기존 attention 기반 모델 대비 유사 복잡도에서 성능 우위를 주장하며(Table 2), 단순히 depth/width/cardinality를 늘린 비교보다 구조적으로 더 효율적인 개선임을 보이려 한다(Table 3). 또한 ShuffleNetV2와 CIFAR 실험으로 경량/소형 설정에서도 일반성이 유지될 수 있음을 보이고(Table 4, 5), ablation과 attention 분석을 통해 multi-kernel과 adaptive selection의 기여를 분해해 설명한다(Table 6, 7, Fig. 3, 4).

_SENet_ 이 채널 축을 재보정해 표현을 강화하는 방향이었다면, _SKNet_ 은 **공간 문맥 스케일**을 선택해 유효 RF를 조절하는 방향이다. 둘 다 전역 요약을 gating에 쓰는 점은 닮았지만, 선택이 작동하는 축이 다르기 때문에 서로 다른 종류의 개선을 제공할 수 있다. 논문은 실제로 SENet 대비 SKNet이 추가 개선을 줄 수 있다는 방향의 비교를 포함해, 스케일 선택이라는 새로운 축이 유효하다는 근거를 쌓는다.

#### 📄 출처
Li, Xiang, Wenhai Wang, Xiaolin Hu, and Jian Yang. *Selective Kernel Networks*. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, arXiv:1903.06586.
