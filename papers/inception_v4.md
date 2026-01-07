# [Inception-v4/ResNet] Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning

이 논문은 Inception 계열(GoogLeNet → Inception-v2/v3)에서 발전해 온 **병렬 멀티브랜치 설계**와, ResNet에서 제안된 **residual connection**을 한 프레임에서 비교하고 결합하는 것을 목표로 한다. 핵심 질문은 단순하다. Inception은 계산 효율이 좋고, ResNet은 학습이 빠르고 안정적인데, 두 장점을 합치면 무엇이 얻어질까?

저자들은 **Inception-ResNet**이라는 하이브리드 구조가 학습을 크게 가속한다는 실증적 결과를 제시하고, residual을 쓰지 않는 순수 Inception 계열에서도 구조를 정리해 더 깊고 균일한 **Inception-v4**를 제안한다. 또한 폭이 큰 residual Inception에서 발생하는 학습 불안정(네트워크가 죽는 현상)을 **residual scaling**으로 완화하는 실전 팁을 제공한다. 마지막으로 ImageNet(ILSVRC 2012) 분류에서 단일 모델과 멀티크롭, 앙상블 평가까지 체계적으로 비교한다.

---

## 1️⃣ Inception의 다음 단계

### 🔹 Inception과 ResNet을 함께 보려는 동기
논문은 2012년 ImageNet 이후 매우 깊은 CNN이 인식 성능 향상을 주도해 왔다는 큰 흐름에서 출발한다. 특히 Inception 계열은 좋은 정확도를 **상대적으로 낮은 연산량**으로 달성한 대표적인 예로 언급되고, **ResNet**은 ILSVRC 2015에서 residual connection을 통해 강력한 성능을 보여주었다.

이때 논문의 문제의식은 단순히 정확도 경쟁이 아니라, **어떤 구조가 학습을 더 빠르고 안정적으로 만드는가**에 가깝다. **Inception**은 모듈을 깊게 쌓아도 계산량을 비교적 통제할 수 있지만, 깊어질수록 최적화가 어렵다는 고민이 따라온다. 반대로 **ResNet**은 $y=x+F(x)$ 형태의 _skip connection_ 을 통해 gradient 경로를 짧게 만들어, 더 깊은 모델을 비교적 쉽게 학습시킨다는 인상을 남겼다. 저자들은 이 두 흐름이 2015 시점에서 비슷한 성능 수준에 도달했다고 보고, 이제는 구조적 결합이 학습 과정 자체에 어떤 영향을 주는가를 확인하려 한다.

여기서 저자들이 던지는 질문은 다음과 같다. 성능이 유사해 보이는 Inception-v3와 ResNet 계열 사이에서, residual을 Inception에 결합했을 때 어떤 이점이 있는가? Inception은 이미 매우 깊기 때문에, 깊은 네트워크 학습을 돕는 residual connection의 장점을 그대로 얻을 수 있을 것이라는 기대가 자연스럽다.

또한 이 질문은 Inception이 본질적으로 concat 기반이라 residual과 잘 안 맞는다는 선입견을 점검하는 의미도 있다. 실제로 Inception 블록은 여러 경로를 concat해 채널을 늘리며 표현력을 확보하는데, residual addition은 입력과 출력의 차원 정렬이 필요하다. 따라서 이 논문은 단순히 블록 하나를 교체하는 수준이 아니라, **Inception의 병렬 설계를 유지하면서도 add 기반 결합이 가능한 형태**로 구조를 재조합해야 한다는 과제를 함께 제시한다.

### 🔸 하이브리드 결합과 순수 구조 정리
이 논문은 탐색 방향을 두 갈래로 분명히 나눈다.

1. **Residual Inception(Inception-ResNet)**: Inception의 필터 concat 단계 주변을 residual addition으로 바꿔, 학습을 빠르게 만들 수 있는지 확인한다.  
2. **Pure Inception(Inception-v4)**: Inception-v3가 역사적 제약(분산 학습 환경에서의 파티셔닝 등) 때문에 남긴 복잡함을 걷어내고, grid size별로 **균일한 블록 선택**을 하여 더 단정한 구조를 만든다.

특히 **Inception-v4**는 Inception-v3의 아이디어를 계승하되, 더 단순하고 더 균일하게, 그리고 더 깊게라는 설계 철학으로 소개된다. 이는 논문 전체에서 여러 번 반복되는 테마인데, Inception 계열의 성능이 단순히 한두 개 트릭이 아니라 **모듈 조합과 스케일링**의 결과라는 메시지로 읽힌다.

이 구분은 실험 설계에서도 중요하다. **Inception-ResNet**은 residual을 넣는 순간 학습이 빨라진다는 최적화 관찰이 구조 변화의 직접 결과인지 보기가 쉬워진다. 반대로 Inception-v4는 residual 없이도 구조를 정리하고 깊이를 늘려 성능이 올라간다면, 이는 residual이 정확도 향상의 유일한 열쇠가 아니라는 반례가 된다. 논문 abstract에서도 두 결과를 분리해 보고하며, residual은 학습 가속을 확실히 제공하지만 최종 성능 차이는 얇을 수 있고, 반면 구조 단순화 및 스케일업은 순수 Inception에서도 성능을 끌어올릴 수 있음을 강조한다.

---

## 2️⃣ 관련 연구

### 🔹 계보 정리: AlexNet → VGG → Inception
Related Work는 전통적인 CNN 발전 흐름을 따라간다. AlexNet 이후 CNN이 객체 검출, 분할, 포즈 추정, 비디오 분류 등 다양한 비전 문제로 확장되었고, 그 과정에서 Network-in-Network, VGGNet, GoogLeNet(Inception-v1) 같은 구조적 이정표가 등장했다.

이 논문 관점에서 중요한 것은 Inception이 깊게 만들되 계산을 감당 가능한 수준으로 유지하기 위한 설계를 지속적으로 발전시켜왔다는 점이다. 또한 Batch Normalization의 도입(BN-Inception, Inception-v2)과 factorization을 포함한 Inception-v3의 개선이 자연스럽게 이어진다. 특히 Inception 계열의 개선은 표현력은 유지하고 연산은 줄인다는 반복되는 패턴을 가진다.

예를 들어,

1. 큰 커널을 작은 커널의 연쇄로 분해하는 **factorization**
2. grid reduction에서 stride를 한 경로에 몰아 넣지 않고 **병렬 경로로 분산**하는 설계
3. BN과 같은 **안정화 장치**를 통해 더 깊은 네트워크를 학습 가능하게 만드는 전략이 계속 등장한다.

이 논문은 그 연장선에서 Inception-v4를 더 균일한 모듈 반복으로 재정의하면서, 구조를 다시 단순한 언어로 설명 가능한 형태로 만들려고 한다.

### 🔸 Residual Connection의 위치
저자들은 residual connection이 매우 깊은 모델 학습에 필수라는 주장(ResNet 원 논문)을 소개하면서도, 적어도 이미지 분류에서는 residual이 없더라도 충분히 깊고 경쟁력 있는 네트워크를 학습시키는 것이 불가능하지 않다고 조심스럽게 반박한다. 다만 residual이 주는 가장 확실한 이점으로 **학습 속도 향상**을 강조한다.

이 대목이 중요한 이유는, 논문이 residual을 정확도를 올리는 마법으로 서술하지 않기 때문이다. 이 논문에서 residual은 더 깊은 모델을 가능하게 하는 하나의 도구이면서, 더 직접적으로는 optimization을 빠르게 만드는 장치로 등장한다. 실제 실험에서도 residual 변형이 더 빨리 수렴한다는 곡선이 핵심 근거로 제시된다.

이 맥락에서 Fig. 1과 Fig. 2는 residual connection의 기본 형태와 최적화된 형태(계산 보호를 위한 형태)를 상기시키는 역할을 한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/d54e4f4c-8cbf-472d-8d4d-ad8afdac7f19/image.png" width="60%">
</p>

---

## 3️⃣ 아키텍처 특징

### 🔹 Inception 블럭
이 절에서 저자들은 Inception-v4 설계의 배경을 설명한다. 과거 Inception 모델들은 분산 학습 환경에서의 파티셔닝(메모리에 맞추기 위한 sub-network 분할)을 염두에 두고 구조가 조정되었고, 그 결과 필요 이상으로 복잡해졌다는 평가가 나온다. TensorFlow 환경으로 옮겨오면서 이 제약이 완화되었고, 그래서 Inception-v4에서는 불필요한 짐을 버리고, grid size(예: 35×35, 17×17, 8×8)마다 **동일한 형태의 Inception 블록을 반복**하는 방향으로 정리한다.

여기서 말하는 균일함은 블록 종류를 줄인다는 수준을 넘어, 설계 의도를 분명히 드러내는 장치다. 예를 들어 35×35 구간은 Inception-A만 반복하고, 17×17은 Inception-B만 반복하며, 8×8은 Inception-C만 반복한다. 이렇게 하면 구조가 복잡해져도 지금은 어떤 grid에서 어떤 블록을 몇 번 반복하는지가 한 줄로 요약되고, 실험에서 어떤 변화가 성능에 영향을 줬는지 추적하기 쉬워진다. 결국 Inception-v4는 구조적 정리를 통해 확장(더 깊게/더 넓게)을 가능하게 하는 형태라고 볼 수 있다.

또 하나의 중요한 표기 규칙이 나온다. 도식에서 `V`가 붙은 convolution은 valid padding(출력 공간 크기 감소), `V`가 없는 convolution은 same padding(공간 크기 유지)을 의미한다. 즉, 다운샘플링이 어디서 일어나는지 도식만으로 추적할 수 있게 만든 규칙이다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/29c9f5a2-a6b4-474a-9a01-22437291da9b/image.png" width="60%">
</p>

### 🔸 Stem: 입력 299×299에서 35×35로 내려가는 전처리 스택
Inception-v4는 ImageNet 입력을 299×299로 두고 시작한다. Stem은 초기 단계에서 공간 해상도를 빠르게 줄이면서 채널을 확장해, 이후 Inception 모듈이 처리하기 좋은 텐서 형태(35×35×384)를 만든다.

Fig. 3의 표기를 따라가면, Stem은 다운샘플링을 한 번에 끝내는 방식이 아니라 여러 단계로 나누어 점진적으로 줄인다. 이는 단순한 편의가 아니라, 초기에는 low-level edge/texture 같은 정보를 잃지 않으면서도 연산량을 감당하기 위해 해상도를 줄여야 한다는 절충의 결과로 읽힌다. 따라서 Stem은 stride 2 conv나 maxpool을 쓰되, 매번 병렬 경로를 두어 정보 손실이 한 경로에만 쏠리지 않도록 설계한다.

논문은 Stem의 구조를 Fig. 3으로 제시한다. 주요 흐름은 다음과 같이 요약할 수 있다.

1. 3×3 stride 2 valid conv로 **299 → 149**  
2. 3×3 valid conv로 **149 → 147**  
3. 3×3 same conv로 **채널 확장**  
4. (maxpool stride 2)와 (3×3 stride 2 conv)를 병렬로 두고 concat  
5. 다시 (1×1→3×3)와 (1×1→7×1→1×7→3×3) 두 경로를 병렬로 두고 concat  
6. 마지막으로 (maxpool stride 2)와 (3×3 stride 2 conv)를 병렬로 두고 concat하여 **35×35×384**를 만든다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/778a9b3b-51f2-4c2d-9460-53d9c4cd0fa5/image.png" width="60%">
</p>

이 Stem은 Inception-v3에서 보던 **factorization**(예: 7×7을 1×7과 7×1로 분해) 아이디어를 초반부터 적극적으로 사용한다는 점이 특징이다. 단순히 빠르게 다운샘플링만 하는 것이 아니라, 병렬 경로를 통해 정보 손실을 분산시키면서 표현력을 확보한다.

추가로, Stem의 병렬 경로들은 단순히 서로 다른 커널을 쓰는 것이 아니라 **서로 다른 정보 압축 방식**을 제공한다. 예를 들어 maxpool 경로는 지역 최대값을 취해 강한 특징을 남기는 반면, stride 2 conv 경로는 학습 가능한 필터로 다운샘플링을 수행한다. 두 출력을 concat하면 같은 해상도에서 풀링 기반 요약과 학습 기반 요약이 함께 들어오므로, 이후 Inception 블록이 더 풍부한 입력을 받게 된다.

### 🔹 Inception-A: 35×35 grid에서의 기본 모듈
35×35 구간에서는 Inception-A 블록을 반복한다. Fig. 4는 Inception-A의 구성(여러 브랜치에서 1×1, 3×3, 3×3×2, avgpool 경로 등을 두고 concat)을 보여준다.

핵심은 

1. 단순 1×1 경로,
2. 1×1→3×3 경로,
3. 1×1→3×3→3×3 경로,
4. avgpool→1×1 경로를 병렬로 두고 채널 방향으로 concat하는 전형적 Inception 패턴이다.

이때 5×5 같은 큰 커널을 직접 쓰기보다는 3×3을 여러 번 쌓아 **비선형성을 더 많이 확보**하는 방향이 자연스럽게 들어간다.

이 블록을 논문 관점에서 읽으면, 각 브랜치는 _서로 다른 receptive field를 담당_ 한다. 1×1 경로는 채널 재조합(특징 선택)에 가깝고, 1×1→3×3은 작은 공간 패턴을, 1×1→3×3→3×3은 더 큰 receptive field를 담당한다. avgpool→1×1 경로는 학습되지 않는 지역 요약을 통해 주변 문맥을 안정적으로 전달하는 역할을 한다. 결국 concat은 서로 다른 스케일의 특징들을 한 텐서에 모아 다음 블록이 다시 분해·결합할 수 있게 만드는 장치다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/68b10b76-b365-4a75-bbe2-02cff7b755ac/image.png" width="40%">
</p>

### 🔸 Reduction-A: 35×35 → 17×17 Downsampling을 안전하게 수행
다운샘플링 시점은 항상 병목이 되기 쉽다. 한 경로로만 강하게 stride를 걸면 정보 손실이 한 곳에 집중되기 때문이다. Fig. 7의 **Reduction-A**는 이를 완화하기 위해 **여러 경로를 병렬로 두고** stride 2를 분산한다.

Reduction-A의 구조를 말로 풀면, maxpool로 줄이는 경로, 3×3 stride 2로 곧장 줄이는 경로, 1×1로 채널을 조정한 뒤 3×3을 거쳐 stride 2, 3×3으로 줄이는 경로의 결합이다. 즉, 같은 다운샘플링이라도 단순한 공간 요약과 학습 가능한 요약, 채널 병목을 둔 요약이 함께 들어가도록 구성한다. 이 방식은 이후 Inception-v3에서도 반복적으로 강조되던 grid reduction의 원칙과 일치한다.

Reduction-A의 중요한 점은 필터 수가 네트워크 변형(Inception-v4, Inception-ResNet-v1, v2)에 따라 달라지도록 `k, l, m, n`으로 파라미터화되어 있다는 것이다. 논문은 이 값을 Table 1에 정리한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/d0ca2872-623d-4d45-a4f3-9aab6cbd4e45/image.png" width="60%">
</p>


이 표는 Reduction-A에서 어느 경로의 채널을 얼마나 줄/늘릴지가 모델별로 조금 다르다는 것을 보여준다. Inception-v4는 비교적 균형적인 값(192,224,256,384)을 선택한다.

### 🔹 Inception-B: 17×17 Grid에서 7×7 계열 Factorization을 적극 사용
17×17 구간에서는 **Inception-B** 블록을 반복한다. 이 블록의 중심 아이디어는 7×7 계열 연산을 직접 쓰는 대신, **1×7과 7×1의 조합**으로 factorize하여 연산량을 줄이면서 표현력을 유지하는 것이다. Inception-v3에서도 등장했던 아이디어가 Inception-v4에서는 더 정돈된 형태로 반복 사용된다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/6838001c-6386-4264-8761-3f1b39e12a7d/image.png" width="40%">
</p>

Inception-B는 여러 브랜치로 구성되며, 그중 일부는 1×7과 7×1을 여러 번 교차시키는 긴 경로를 가진다. 이때 중요한 직관은 다음과 같다.

- 7×7의 receptive field를 얻고 싶지만, 7×7 conv의 비용은 크다.  
- 1×7과 7×1을 연속 적용하면 같은 receptive field 확장을 얻으면서 파라미터/연산을 줄일 수 있다.  
- conv를 여러 번 쪼개면 중간에 BN/ReLU가 들어가 비선형성이 늘어나, 단순히 근사 비용 절감이 아니라 표현력 측면에서도 유리해질 수 있다.

조금 더 계산 관점에서 보면, $7\times 7$ conv는 커널 면적이 $49$라서 연산량이 크다. 반면 $(1\times 7)$과 $(7\times 1)$을 연속으로 두면 커널 면적 합이 $14$로 줄어든다. 물론 채널 수와 중간 채널 설계에 따라 정확한 비용은 달라지지만, 큰 공간 커널을 비대칭 커널 둘로 쪼개면 비용이 크게 줄어든다는 방향은 변하지 않는다. 이 논문은 이런 factorization을 17×17 구간의 기본 블록으로 채택하여, 깊이를 늘려도 비용이 폭발하지 않도록 한다.

### 🔸 Reduction-B: 17×17 → 8×8 Grid Reduction
다음 downsampling 지점은 17×17에서 8×8로 줄어드는 구간이다. Fig. 8은 **Reduction-B**의 도식을 제공한다. 여기서도 maxpool 경로와 stride 2 conv 경로를 병렬로 둬 정보를 분산시키는 설계가 유지된다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/61c8f083-ebe8-4f00-b047-03ecbae62c69/image.png" width="40%">
</p>

Reduction-B는 Reduction-A보다 factorization이 더 적극적으로 들어가며, stride 2가 걸리는 경로들 앞에서 1×1로 채널을 정리하고 (1×7, 7×1)을 거쳐 3×3 stride 2로 마무리하는 경로가 등장한다. 이는 17×17 구간에서 이미 쓰던 비대칭 커널 설계를 downsampling 구간에도 연결해, 해상도 변화 구간에서 표현 손실을 완화하려는 의도로 해석할 수 있다.

### 🔹 Inception-C: 8×8 Grid에서의 최종 표현 학습
마지막 8×8 구간에서는 **Inception-C** 블록을 사용한다. Inception-C는 3×3 계열을 분해한 비대칭 커널(1×3, 3×1) 사용과, 브랜치를 더 세분화하여 concat하는 구조가 핵심이다.

이 모듈이 8×8처럼 작은 grid에서 쓰인다는 점이 중요하다. 공간 크기가 작은 상황에서는 채널 방향 결합과 커널 factorization이 연산량을 제어하면서도 표현을 풍부하게 만드는 핵심 수단이 된다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/bc45520c-fd2e-40ea-bc27-0c5ca1526bdd/image.png" width="40%">
</p>

또한 8×8 단계는 분류기(head) 직전이기 때문에, 여기서의 특징은 곧 클래스 분리를 결정하는 고수준 표현으로 이어진다. 따라서 이 논문은 마지막 단계에서 블록을 더 복잡하게(브랜치 내부를 좌/우로 쪼개는 등) 구성해 채널을 풍부하게 만들고, 그 뒤 global average pooling으로 공간을 접어 채널 의미를 직접적으로 로짓으로 연결한다.

### 🔸 Inception-v4 전체 구조 한 줄 요약

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/7103b087-07a6-41ca-85a3-de9b32c599b7/image.png" width="60%">
</p>

Fig. 9는 **Inception-v4**를 큰 뼈대로 요약한다.

- Stem(입력 전처리)  
- 4× Inception-A(35×35)  
- Reduction-A  
- 7× Inception-B(17×17)  
- Reduction-B  
- 3× Inception-C(8×8)  
- Average Pooling → Dropout(keep 0.8) → Softmax

이 `4-7-3` 반복 패턴은 Inception-v4가 grid size별로 균일한 모듈을 반복하도록 정리되었음을 상징적으로 보여준다.

Fig. 9에는 각 구간의 출력 채널 수도 함께 표기되어 있는데, 대략 35×35에서 384, 17×17에서 1024, 8×8에서 1536으로 확장된다. 즉 해상도는 줄고 채널은 늘어나는 전형적인 CNN 스케일링을 따르면서도, 각 구간의 블록 종류를 고정해 설계 공간을 단순화한다. 논문이 말하는 streamlined architecture는 이런 선택의 결과이며, 구조가 간결해진 만큼 블록 반복 횟수(`4-7-3`)로 용량을 키우는 방식이 자연스럽게 된다.

### 🔹 Residual Inception 블럭
Residual Inception의 핵심은 Inception 블록이 만든 출력과 입력을 더한다는 것이다. 표준 ResNet 관점에서 보면, 어떤 서브네트워크 $F(x)$를 적용한 뒤 $x$를 더해 $y = x + F(x)$ 형태를 만든다.

하지만 Inception 블록은 여러 경로 concat을 통해 채널을 늘리거나 줄이는 과정이 들어가므로, 입력 $x$와 출력 $F(x)$의 채널 수가 다를 수 있다. 논문은 이를 해결하기 위해 Inception 블록 뒤에 **filter-expansion layer**를 둔다고 설명한다. 구체적으로는 1×1 convolution을 두되, activation이 없는 `linear projection`으로 두어서 덧셈 직전에 채널 수를 맞춘다.

이를 식으로 쓰면 다음과 같이 정리할 수 있다.

$$y = x + W \cdot F(x)$$

여기서 $W$는 1×1 linear conv에 해당하고, $F(x)$는 Inception 형태의 멀티브랜치 블록 출력이다. $W$를 곱하는 이유는 채널 정렬, residual의 크기 조절(나중에 scaling과 연결)이라는 두 의미를 동시에 가진다.

이 관점에서 **Inception-ResNet**은 concat 기반의 폭 확장을 항상 채널을 늘리는 방식이 아니라, 필요한 변화량만 residual로 더하는 방식으로 바꾼 셈이다. concat은 채널 차원에서 정보를 병렬로 쌓아 다음 레이어가 다시 섞게 만들지만, add는 입력과 동일한 차원에서 변화를 누적한다. 따라서 residual 버전은 학습이 빨라질 가능성이 높지만, 동시에 폭이 매우 커질 때 불안정성이 생길 수 있고(다음 절), 그때 scaling 같은 보조 장치가 필요해진다.

논문은 Inception-ResNet-v1과 v2를 상세히 제시하며, 전체 스키마를 Fig. 15에 제시한다. v1은 Inception-v3 수준의 비용에 맞추고, v2는 Inception-v4 수준의 비용에 맞춘다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/9ee15749-e698-4be4-8649-1d3ac3570911/image.png" width="75%">
</p>

또한 residual 버전에서는 합산(summation) 위에 BN을 두지 않는 선택을 했음을 언급한다. 이유는 실무적이다. 큰 activation 크기를 가진 구간에서 BN이 GPU 메모리를 과도하게 먹어, 블록 수를 늘리는 데 제약이 되었다는 것이다. 즉, 이 논문은 구조 설계가 단순히 수식적 우아함이 아니라 **훈련 인프라와 자원 제약**에 의해 결정되는 현실을 드러낸다.

### 🔸 Residual 스케일링하기
Residual Inception을 넓게(wide) 만들면, 학습이 불안정해져 네트워크가 죽는 현상이 관찰되었다고 한다. 논문 표현을 빌리면 average pooling 직전의 마지막 레이어가 몇만 iteration 이후 **0만 출력**하는 상태가 된다는 것이다. 흥미로운 점은 learning rate를 낮추거나 BN을 추가하는 방식으로는 이 문제가 충분히 해결되지 않았다는 관찰이다.

이 현상은 residual addition이 누적되는 구조에서 자연스럽게 이해할 수 있다. 각 블록이 만들어내는 변화량 $F(x)$가 너무 커지면, $x$에 더해지는 값이 반복적으로 커지면서 activation 분포가 급격히 바뀌고, 그 결과 일부 구간이 포화되거나(특히 ReLU 이후) gradient가 사실상 사라지는 상태가 생길 수 있다. 논문은 특히 채널 수가 1000을 넘는 넓은 구성에서 이런 문제가 도드라졌다고 보고한다.

저자들이 제안한 실용적 해결책은 residual을 더하기 전에 **상수 배로 줄이는 scaling**이다(Fig. 20). 즉,

$$y = x + \alpha \cdot F(x), \quad \alpha \in [0.1, 0.3]$$

처럼 residual branch 출력에 $\alpha$를 곱한 뒤 더한다. Fig. 20의 도식은 `Inception-ResNet 모듈 → (마지막 linear conv) → scaling → add → ReLU` 구조로 설명된다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/5aeade0b-f899-4bb4-b567-0085a313cc2c/image.png" width="40%">
</p>

이 아이디어의 직관은 gradient 흐름을 지나치게 요동치게 하는 큰 residual을 억제하여, 초기 학습 단계에서 누적 activation이 폭발하거나 사라지는 현상을 완화하는 것이다. ResNet 원 논문에서 제시된 warm-up(아주 작은 learning rate로 시작하는 2-phase training)과 대비되며, 저자들은 매우 작은 learning rate(예: $10^{-5}$)조차 안정화를 보장하지 못했다고 말한다. 반면 scaling은 정확도를 크게 해치지 않으면서 안정성을 올렸다고 보고한다.

이를 구현 관점에서 정리하면, residual 블록이 반환하는 마지막 선형(activation 없는) 출력에 $\alpha$를 곱하고 더한다는 절차로 요약된다.

```text
Algorithm: Residual scaling in Inception-ResNet
Input: x, residual_block F, projection W (optional), scale α

r = F(x)
r = W(r)              # 1×1 linear projection to match channels (if needed)
y = x + α * r
return ReLU(y)
```

---

## 4️⃣ 훈련 방법론

### 🔹 분산 학습 설정과 옵티마이저
훈련은 TensorFlow 분산 학습 시스템에서 20개의 replica를 사용해 수행되었다. 각 replica는 _NVidia Kepler GPU_ 에서 실행되었다고 명시한다. 옵티마이저는 초기 실험에서는 momentum(감쇠 $0.9$)을 썼지만, 가장 좋은 모델들은 **RMSProp**을 사용했다.

RMSProp 설정은 다음과 같이 제시된다.

- decay = 0.9  
- $\epsilon = 1.0$  
- initial learning rate = 0.045  
- learning rate decay: 2 epoch마다 exponential rate 0.94로 감소

또한 모델 평가는 시간에 따른 파라미터의 running average를 사용하여 수행했다고 한다. 즉, 단일 스텝 파라미터가 아니라 EMA(지수 이동 평균에 가까운) 형태의 파라미터로 평가한 셈이다.

이 선택은 ImageNet 규모에서 흔히 보이는 실전적 판단과 맞닿아 있다. 분산 학습에서는 각 replica의 업데이트가 병렬로 합쳐지기 때문에 gradient 노이즈와 학습률 민감도가 커질 수 있고, RMSProp 같은 적응적 스케줄이 안정화에 도움을 줄 수 있다. 또한 running average(평가 시점의 파라미터 평균)를 사용하면 학습 중 진동을 줄여 validation 성능이 더 안정적으로 측정되는 경향이 있다. 논문은 이 요소들을 자세히 해설하기보다는 사용한 설정을 명시하는 방식으로 서술한다.

### 🔸 학습 레시피 의사코드
논문은 학습 레시피를 길게 늘어놓기보다는 핵심 하이퍼파라미터를 간결히 제시한다. 이를 구현 관점에서 요약하면 아래와 같은 의사코드로 정리할 수 있다.

```text
Algorithm: Training schedule (paper recipe)
Input: model f_θ, dataset D, initial lr = 0.045
Hyperparams: RMSProp(decay=0.9, eps=1.0), lr_decay_every=2 epochs, lr_decay_rate=0.94
State: running_average_params θ̄

for epoch = 1..E:
    lr = 0.045 * (0.94)^(floor((epoch-1)/2))
    for minibatch (x, y) in D:
        g = ∇_θ L(f_θ(x), y)
        θ = RMSPropUpdate(θ, g; lr, decay=0.9, eps=1.0)
        θ̄ = UpdateRunningAverage(θ̄, θ)
    Evaluate f_{θ̄} on validation
return θ̄
```

여기서 핵심은 2 epoch마다 learning rate를 한 번씩 줄이는 구조이며, RMSProp의 eps를 1.0으로 둔 점이 눈에 띈다. 일반적인 구현에서 eps가 더 작은 경우도 많지만, 논문은 이 값이 최적이었다고 간단히 보고한다.

또한 이 절의 서술은 구조가 바뀌면 레시피도 바뀐다는 식이 아니라, 비교 실험을 위해 가능한 한 레시피를 공통으로 유지하면서 구조의 영향을 보려 했다는 인상을 준다. 즉, Inception-v3/v4와 Inception-ResNet 변형들이 서로 다른 학습 속도를 보였다면, 그 원인은 학습 레시피의 차이보다 구조(특히 residual addition)의 차이에 가깝다고 해석할 여지가 생긴다.

---

## 5️⃣ 실험 결과

### 🔹 Residual 학습 곡선
실험 파트의 첫 메시지는 명확하다. Inception 구조에 residual connection을 붙이면 학습이 크게 빨라진다. Fig. 21–24는 Inception-v3/v4와 비용이 비슷한 residual 변형(Inception-ResNet-v1/v2)의 학습 곡선을 비교한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/f9d2e2b2-2d15-423b-bd1a-361484047293/image.png" width="60%">
</p>

곡선이 시사하는 바는 더 좋은 최종 정확도라기보다 더 빠른 최적화에 가깝다. 즉, residual은 목표 지점까지 도달하는 시간을 줄여주는 역할이 강하다.

논문은 이 관찰을 학습이 잘 된다는 식이 아니라, 같은 비용 수준에서 수렴 속도가 크게 달라진다는 식으로 해석한다. 즉, residual 연결은 Inception 블록의 표현 설계를 바꾸기보다, 그 표현을 학습하는 최적화 과정을 바꾸는 장치로 드러난다. 또한 Fig. 25–26은 네 모델의 학습 곡선을 함께 보여주며, residual이 빠르게 내려가지만 최종 정확도는 모델 크기(또는 용량)에 더 크게 좌우된다는 결론을 보조한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/043fc1b4-9139-420d-8042-7b16536fa664/image.png" width="60%">
</p>

### 🔸 Blacklist Subset 문제
논문은 실험 후반에 중요한 주의사항을 밝힌다. 지속적(continuous) 평가가 validation set의 일부(약 1700개 blacklist entity)를 제외한 subset에서 수행되었고, 이는 CLSLOC 벤치마크에만 적용되어야 하는데 분류 평가에도 적용된 오류였다는 것이다. 그 결과 숫자가 다른 보고들보다 다소 낙관적으로 보일 수 있으며, 차이는 top-1 약 0.3%, top-5 약 0.15%라고 한다.

다만 논문은 곡선 간 비교에는 일관되게 동일한 subset을 썼기 때문에 공정하다고 주장한다. 대신 멀티크롭/앙상블 결과는 50,000장의 전체 validation set에서 재평가했다고 한다.

이 주의사항은 숫자 자체보다, 논문이 비교를 해석하는 태도를 보여준다. 즉, 저자들은 single-crop continuous evaluation의 절대 수치가 외부 보고와 완전히 일치하지 않을 수 있음을 인정하고, 대신 같은 조건에서의 상대 비교에 초점을 맞춘다. 그리고 더 중요한 멀티크롭/앙상블 결과는 전체 validation으로 다시 측정해 신뢰도를 보완한다.

### 🔹 단일 모델, 단일 크롭 결과
Table 2는 non-blacklisted subset에서의 single crop, single model 성능이다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/bd13aab4-7f35-4eba-a92a-c2d5db096fc8/image.png" width="40%">
</p>

이 표만 보면 Inception-v4와 Inception-ResNet-v2가 **거의 비슷한 성능**을 보이고, residual이 순수 Inception보다 아주 약간 더 낫거나 비슷한 수준으로 해석된다. 하지만 차이가 매우 얇기 때문에, 이 논문이 강조하는 것은 결국 residual이 최종 성능을 크게 바꿨다기보다는 학습을 빠르게 만들었다 쪽이다.

또 다른 관찰은 Inception-v4 자체의 개선 폭이다. Inception-v3의 Top-5 error 5.6%에서 Inception-v4 5.0%로 내려가는 것은 단일 크롭 기준으로도 뚜렷한 개선이다. 반면 Inception-v4(5.0%)와 Inception-ResNet-v2(4.9%)의 차이는 0.1%p로 매우 작다. 이 대비는, 논문이 제안한 streamlined architecture(순수 Inception의 구조 정리와 스케일업)가 실제로 큰 비중을 차지하며, residual은 그 위에서 학습 과정을 빠르게 만들거나 얇은 마진을 주는 정도로 나타난다는 해석을 지지한다.

### 🔸 10/12 Crops 평가와 144 Crops 평가0
Table 3는 작은 수의 크롭(ResNet은 10 crops, Inception 계열은 12 crops)으로 평가한 결과다. Table 4는 144 crops(ResNet은 dense로 표기)라는 더 강한 평가로 비교한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/a91d4ebe-74df-4f84-990b-f86587912eb4/image.png" width="60%">
</p>

여기서도 Inception-v4와 Inception-ResNet-v2가 **거의 동급**으로 나오며, 평가가 강해질수록 전체적으로 error가 내려가지만 상대적 순위는 안정적으로 유지된다.

이 결과는 평가 전략(크롭 수)이 절대 수치를 바꾸되, 모델 간 상대적 우열은 크게 흔들지 않는다는 점을 보여준다. 즉, Inception-v4와 Inception-ResNet-v2는 single-crop에서도, 12 crops에서도, 144 crops에서도 계속 비슷한 수준으로 함께 움직인다. 이는 두 모델이 **표현력과 용량 측면에서 비슷한 지점**에 있으며, residual 여부가 최종 표현의 상한을 크게 바꾸지 않을 수 있다는 해석으로 이어진다(물론 학습 속도는 다르다).

### 🔹 앙상블과 Test-Set 확인
Table 5는 앙상블 결과를 비교한다. Inception 계열 앙상블은 Inception-v4 1개 + Inception-ResNet-v2 3개로 구성되었고, 144 crops 전략으로 평가했다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/5f85db73-24b7-4e04-91e6-1ce0b69da84c/image.png" width="40%">
</p>

또한 이 앙상블은 test set에서도 **3.08% top-5 error**를 얻었다고 보고하며, validation에 과적합하지 않았음을 확인하는 근거로 제시한다.

여기서 흥미로운 관찰은 단일 프레임 성능 향상이 앙상블 성능에 비례해 크게 반영되지는 않았다는 언급이다. 즉, 여러 모델이 이미 비슷한 방향으로 틀리는 문제(데이터 라벨 노이즈, 본질적 애매함 등)가 남아 있으면 앙상블에도 상한이 생긴다는 결론으로 이어진다.

이 점은 논문 결론에서 다시 등장하는 label noise에 아직 도달하지 않았다는 진술과 연결된다. 즉, 모델을 더 키우거나 더 빠르게 학습시키는 것만으로는 해결되지 않는 오류가 남아 있으며, 그 오류는 데이터의 애매함이나 annotation 품질과 같은 요인과 맞물릴 수 있다. Inception-v4/ResNet 계열이 이미 매우 강력해진 상황에서는, 구조 개선이 가져오는 이득이 점점 얇아지고, 평가 프로토콜(멀티크롭, 앙상블)의 한계도 더 뚜렷해진다.

---

## 6️⃣ 결론
Residual connection은 Inception 네트워크의 **학습을 크게 가속**하며, 적절한 설계(특히 scaling)를 더하면 Inception-v4와 Inception-ResNet-v2 수준의 강력한 모델을 안정적으로 학습시킬 수 있다.

여기서 핵심은 가속이다. 논문은 residual이 반드시 더 높은 최종 정확도를 보장한다고 주장하지 않고, 대신 매우 깊은 Inception 계열에서도 residual 연결이 optimization을 빠르게 만든다는 점을 실험적으로 보여준다. 또한 residual의 부작용(넓은 모델에서의 불안정성)을 scaling으로 해결할 수 있다는 메시지도 함께 담는다.

저자들은 세 가지 구조를 제시했다고 정리한다.

- **Inception-v4**: residual이 없는 순수 Inception이지만 구조가 더 단순/균일하고 깊어진 버전
- **Inception-ResNet-v1**: Inception-v3와 유사한 계산 비용의 residual 하이브리드  
- **Inception-ResNet-v2**: 더 큰 비용이지만 성능이 더 좋은 residual 하이브리드  

또한 residual scaling은 매우 넓은 모델에서 안정성을 높이는 실전적인 장치로 강조된다.

정리하면, 이 논문은 Inception 계열의 구조를 정리해서 더 키울 수 있다는 점과 residual을 결합하면 학습이 빨라진다는 점을 동시에 성립시키는 방향으로 결론을 낸다. 즉, Inception-v4는 구조적 정리를 통해 용량을 키우는 축을, Inception-ResNet은 최적화 관점의 개선을 대표한다. 실험 결과가 두 축의 효과를 분리해 보여준다는 점에서, 2016년 시점의 대규모 분류 모델 설계가 어떤 문제(연산, 메모리, 수렴 속도, 안정성)를 어떻게 다뤘는지 이해하는 데 좋은 자료가 된다.

---

## 💡 해당 논문의 시사점과 한계
이 논문이 가진 의의는 단순히 Inception-v4를 제안했다에만 있지 않다. Inception 계열이 커지면서 복잡해졌던 구조를 **grid size별로 균일한 모듈 반복**이라는 원칙으로 정리해, 아키텍처를 다시 읽기 쉬운 형태로 만들어 주었다. 또한 residual connection을 필수적인 이론적 장치로만 보지 않고, 실제로는 **최적화 관점에서 학습을 빠르게 만들어 주는 도구**로 해석하며, 그 효과를 Inception이라는 다른 설계 철학에도 적용해 확인했다.

한계도 분명하다. Inception-ResNet 변형들은 비용을 맞추기 위해 다소 _ad hoc_ 하게 선택되었음을 논문이 스스로 인정한다. 그리고 어떤 설계 선택이 정확히 성능/학습 속도에 기여했는지에 대한 정교한 ablation은 제한적이다. 마지막으로, 이 논문이 보여주는 상한(앙상블 성능이 크게 더 좋아지지 않는 현상)은 모델 구조만으로 해결되지 않는 데이터/라벨 품질 문제를 시사한다. 즉, 구조 설계의 진전이 데이터 문제를 완전히 대체하지는 못한다.

---

## 👨🏻‍💻 Inception-v4 구현하기
이 파트에서는 [`lucid`](https://github.com/ChanLumerico/lucid/tree/main) 라이브러리 내부 [`inception.py`](https://github.com/ChanLumerico/lucid/blob/main/lucid/models/imgclf/inception.py)에 구현된 Inception-v4를 논문 관점으로 읽는다. Lucid는 Inception 계열을 `inception_v1`, `inception_v3`, `inception_v4` 팩토리 함수로 등록하고, Inception-v4는 `Inception_V4` 클래스에 구현되어 있다.

논문은 Inception-v4뿐 아니라 Inception-ResNet-v1/v2까지 함께 다루지만, Lucid 구현은 파일이 분리되어 있다. Inception-v4는 `inception.py`에, Inception-ResNet은 `inception_res.py`에 구현되어 있으므로, 여기서는 먼저 Fig. 3–9(순수 Inception-v4 구성요소)와의 대응에 집중한다.

### 1️⃣ 베이스 클래스: `Inception`
Inception 계열 모델들은 공통 베이스로 `Inception(nn.Module)`을 상속한다. 여기서는 공통 설정값인 `num_classes`와 `use_aux`를 저장하고, `forward`는 하위 클래스에서 실제로 구현하도록 열어둔다.

```python
class Inception(nn.Module):
    def __init__(self, num_classes: int, use_aux: bool | None = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_aux = use_aux

    @override
    def forward(self, x: Tensor) -> Tuple[Tensor | None, ...]:
        return super().forward(x)
```

Inception-v4는 논문에서도 auxiliary classifier를 강조하지 않기 때문에, Lucid 구현 역시 v4에서는 보조 헤드를 두지 않고 최종 로짓만 반환한다. 그럼에도 베이스 클래스는 v1/v3처럼 aux가 존재할 수 있는 케이스를 포괄할 수 있게 설계되어 있다.

### 2️⃣ Stem: `_InceptionStem_V4`
논문 Fig. 3의 Stem은 299×299에서 35×35×384까지 내려가는 입력부다. Lucid의 `_InceptionStem_V4`는 그 뼈대를 그대로 따라간다.

```python
class _InceptionStem_V4(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 32, kernel_size=3, stride=2),
            nn.ConvBNReLU2d(32, 32, kernel_size=3),
            nn.ConvBNReLU2d(32, 64, kernel_size=3, padding=1),
        )

        self.branch1_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch1_conv = nn.ConvBNReLU2d(64, 96, kernel_size=3, stride=2)

        self.branch2_left = nn.Sequential(
            nn.ConvBNReLU2d(160, 64, kernel_size=1),
            nn.ConvBNReLU2d(64, 96, kernel_size=3),
        )
        self.branch2_right = nn.Sequential(
            nn.ConvBNReLU2d(160, 64, kernel_size=1),
            nn.ConvBNReLU2d(64, 64, kernel_size=(7, 1), padding=(3, 0)),
            nn.ConvBNReLU2d(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            nn.ConvBNReLU2d(64, 96, kernel_size=3),
        )

        self.branch3_conv = nn.ConvBNReLU2d(192, 192, kernel_size=3, stride=2)
        self.branch3_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)

        x = lucid.concatenate([self.branch1_pool(x), self.branch1_conv(x)], axis=1)
        x = lucid.concatenate([self.branch2_left(x), self.branch2_left(x)], axis=1)
        x = lucid.concatenate([self.branch3_pool(x), self.branch3_conv(x)], axis=1)

        return x
```

첫 `self.conv`는 (stride 2 포함) 3×3 conv 스택으로 빠르게 해상도를 낮추고 채널을 늘린다. 이후 branch1에서는 maxpool과 stride 2 conv를 병렬로 두고 concat하여, Fig. 3의 첫 번째 concat 지점을 구현한다.

다만 `forward`의 두 번째 concat에서 `self.branch2_left(x)`가 두 번 호출되고 `self.branch2_right(x)`가 호출되지 않는다. 논문 Fig. 3은 (1×1→3×3) 경로와 (1×1→7×1→1×7→3×3) 경로를 병렬로 두는 구조이므로, 의도는 `left`와 `right`를 concat하는 것이 자연스럽다. 이 부분은 구현상 오타 가능성이 있어, 실제 실행 시 논문 Stem과 정확히 동일한 구조가 되지 않는다.

### 3️⃣ 35×35 모듈: `_InceptionModule_V4A`
Inception-v4의 35×35 구간은 Inception-A를 4번 반복한다. Lucid의 `_InceptionModule_V4A`는 평균풀링 경로를 포함한 4-브랜치 concat 구조로 구현된다.

```python
class _InceptionModule_V4A(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, padding=1),
            nn.ConvBNReLU2d(in_channels, 96, kernel_size=1),
        )
        self.branch2 = nn.ConvBNReLU2d(in_channels, 96, kernel_size=1)

        self.branch3 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 64, kernel_size=1),
            nn.ConvBNReLU2d(64, 96, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 64, kernel_size=1),
            nn.ConvBNReLU2d(64, 96, kernel_size=3, padding=1),
            nn.ConvBNReLU2d(96, 96, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return lucid.concatenate(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            axis=1,
        )
```

브랜치 구성은 Fig. 4의 핵심 요소(1×1, 1×1→3×3, 1×1→3×3→3×3, pool→1×1)를 코드로 옮긴 형태다. `AvgPool2d(kernel_size=3, padding=1)`는 공간 크기를 유지하는 풀링 경로를 만든 뒤 1×1로 채널을 정리한다.

### 4️⃣ Reduction-A: `_InceptionReduce_V4A`
Reduction-A는 35×35에서 17×17로 내려가는 다운샘플링 블록이다. Lucid 구현은 maxpool 경로 + stride 2 conv 경로 + 1×1→3×3→stride 2 3×3 경로를 concat하는 형태로 구성된다.

```python
class _InceptionReduce_V4A(nn.Module):
    def __init__(self, in_channels: int, k: int, l: int, m: int, n: int) -> None:
        super().__init__()

        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch2 = nn.ConvBNReLU2d(in_channels, n, kernel_size=3, stride=2)

        self.branch3 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, k, kernel_size=1),
            nn.ConvBNReLU2d(k, l, kernel_size=3, padding=1),
            nn.ConvBNReLU2d(l, m, kernel_size=3, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return lucid.concatenate(
            [self.branch1(x), self.branch2(x), self.branch3(x)],
            axis=1,
        )
```

여기서 `(k, l, m, n)`은 논문 Table 1의 파라미터에 해당한다. 아래에서 보듯 Lucid의 `Inception_V4`는 Inception-v4 행의 값을 그대로 사용한다.

### 5️⃣ 17×17 모듈: `_InceptionModule_V4B`
17×17 구간은 Inception-B를 7번 반복한다(Fig. 9). Inception-B는 1×7/7×1 factorization을 포함한 여러 경로를 concat한다.

```python
class _InceptionModule_V4B(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, padding=1),
            nn.ConvBNReLU2d(in_channels, 128, kernel_size=1),
        )

        self.branch2 = nn.ConvBNReLU2d(in_channels, 384, kernel_size=1)
        self.branch3 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 192, kernel_size=1),
            nn.ConvBNReLU2d(192, 224, kernel_size=(1, 7), padding=(0, 3)),
            nn.ConvBNReLU2d(224, 256, kernel_size=(7, 1), padding=(3, 0)),
        )

        self.branch4 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 64, kernel_size=1),
            nn.ConvBNReLU2d(64, 192, kernel_size=(1, 7), padding=(0, 3)),
            nn.ConvBNReLU2d(192, 224, kernel_size=(7, 1), padding=(3, 0)),
            nn.ConvBNReLU2d(224, 224, kernel_size=(1, 7), padding=(0, 3)),
            nn.ConvBNReLU2d(224, 256, kernel_size=(7, 1), padding=(3, 0)),
        )

    def forward(self, x: Tensor) -> Tensor:
        return lucid.concatenate(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            axis=1,
        )
```

`branch3`와 `branch4`는 논문 Fig. 5의 7×7 factorization 경로를 코드로 옮긴 형태다. 7×7을 직접 쓰는 대신 (1×7, 7×1)을 교차 적용해 receptive field를 넓히고, 중간에 BN/ReLU가 반복되도록 구성한다.

### 6️⃣ Reduction-B: `_InceptionReduce_V4B`
Reduction-B는 17×17에서 8×8로 내려간다. Lucid 구현은 maxpool 경로, (1×1→stride 2 3×3) 경로, (1×1→1×7→7×1→stride 2 3×3) 경로를 병렬로 둔다.

```python
class _InceptionReduce_V4B(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch2 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 192, kernel_size=1),
            nn.ConvBNReLU2d(192, 192, kernel_size=3, stride=2),
        )

        self.branch3 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 256, kernel_size=1),
            nn.ConvBNReLU2d(256, 256, kernel_size=(1, 7), padding=(0, 3)),
            nn.ConvBNReLU2d(256, 320, kernel_size=(7, 1), padding=(3, 0)),
            nn.ConvBNReLU2d(320, 320, kernel_size=3, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return lucid.concatenate(
            [self.branch1(x), self.branch2(x), self.branch3(x)],
            axis=1,
        )
```

논문 Fig. 8에서 downsampling이 여러 경로로 분산되어 정보 손실이 한 지점에 몰리지 않도록 설계된 점과 대응된다.

### 7️⃣ 8×8 모듈: `_InceptionModule_V4C`
8×8 구간은 Inception-C를 3번 반복한다. Lucid 구현은 일부 경로를 좌/우로 분기하여 (1×3, 3×1) 형태의 비대칭 커널 분해를 표현한다.

```python
class _InceptionModule_V4C(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, padding=1),
            nn.ConvBNReLU2d(in_channels, 256, kernel_size=1),
        )

        self.branch2 = nn.ConvBNReLU2d(in_channels, 256, kernel_size=1)

        self.branch3 = nn.ConvBNReLU2d(in_channels, 384, kernel_size=1)
        self.branch3_left = nn.ConvBNReLU2d(
            384, 256, kernel_size=(1, 3), padding=(0, 1)
        )
        self.branch3_right = nn.ConvBNReLU2d(
            384, 256, kernel_size=(3, 1), padding=(1, 0)
        )

        self.branch4 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 384, kernel_size=1),
            nn.ConvBNReLU2d(384, 448, kernel_size=(1, 3), padding=(0, 1)),
            nn.ConvBNReLU2d(448, 512, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.branch4_left = nn.ConvBNReLU2d(
            512, 256, kernel_size=(3, 1), padding=(1, 0)
        )
        self.branch4_right = nn.ConvBNReLU2d(
            512, 256, kernel_size=(1, 3), padding=(0, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x3 = self.branch3(x)
        x3l = self.branch3_left(x3)
        x3r = self.branch3_right(x3)

        x4 = self.branch4(x)
        x4l = self.branch4_left(x4)
        x4r = self.branch4_right(x4)

        return lucid.concatenate([x1, x2, x3l, x3r, x4l, x4r], axis=1)
```

Inception-C의 특징은 브랜치 내부에서 다시 좌/우로 갈라져 concat되는 구조다. 논문 Fig. 6에서도 3×3 계열을 비대칭 커널 조합으로 분해하고, 이를 병렬로 결합해 표현을 확장하는 방향이 강조된다.

### 8️⃣ 전체 네트워크: `Inception_V4`
`Inception_V4`는 앞서 정의한 stem, Inception-A/B/C, reduction 블록들을 `nn.Sequential`로 묶어 전체 네트워크를 만든다. 논문 Fig. 9의 `4-7-3` 반복이 그대로 코드에 나타난다.

```python
class Inception_V4(Inception):
    def __init__(self, num_classes: int = 1000, use_aux: bool = True):
        super().__init__(num_classes, use_aux)
        in_channels = 3

        modules = []
        modules.append(_InceptionStem_V4(in_channels))

        for _ in range(4):
            modules.append(_InceptionModule_V4A(384))
        modules.append(_InceptionReduce_V4A(384, k=192, l=224, m=256, n=384))

        for _ in range(7):
            modules.append(_InceptionModule_V4B(1024))
        modules.append(_InceptionReduce_V4B(1024))

        for _ in range(3):
            modules.append(_InceptionModule_V4C(1536))

        self.conv = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.8)
        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
```

여기서 중요한 연결점은 다음과 같다.

- `modules.append(_InceptionStem_V4(in_channels))`는 Fig. 3 Stem에 대응한다.
- `for _ in range(4): modules.append(_InceptionModule_V4A(384))`는 Fig. 9의 4× Inception-A에 대응한다.
- `_InceptionReduce_V4A(384, k=192, l=224, m=256, n=384)`는 Table 1의 Inception-v4 행을 그대로 반영한다.
- `for _ in range(7): modules.append(_InceptionModule_V4B(1024))`는 Fig. 9의 7× Inception-B에 대응한다.
- `for _ in range(3): modules.append(_InceptionModule_V4C(1536))`는 Fig. 9의 3× Inception-C에 대응한다.
- `Dropout(p=0.8)`은 Lucid의 `Dropout` 정의상 drop 확률을 의미한다(`lucid/nn/modules/drop.py`). 따라서 이는 keep $0.8$이 아니라 keep $0.2$에 해당하며, 논문 Fig. 9의 dropout keep $0.8$과는 숫자 해석이 다르다.

### 9️⃣ 모델 등록 함수: `inception_v4`
마지막으로 Lucid의 모델 레지스트리에 등록되는 entry point는 `inception_v4`다.

```python
@register_model
def inception_v4(num_classes: int = 1000, **kwargs) -> Inception:
    return Inception_V4(num_classes, None, **kwargs)
```

이 함수는 `Inception_V4`를 생성해 반환한다. 두 번째 인자 `None`은 `Inception` 베이스의 `use_aux` 슬롯에 해당하며, Inception-v4 구현이 auxiliary classifier를 사용하지 않는다는 의도를 드러내는 신호로 볼 수 있다(실제로 `Inception_V4.forward`는 로짓 텐서 하나만 반환한다).

---

## 👨🏻‍💻 Inception-ResNet 구현하기
이제 같은 논문에서 함께 제시되는 Inception-ResNet-v1/v2를 Lucid 코드로 읽어보자. Lucid에서는 Inception-ResNet을 [`inception_res.py`](https://github.com/ChanLumerico/lucid/blob/main/lucid/models/imgclf/inception_res.py)에 구현해 두었고, `inception_resnet_v1`, `inception_resnet_v2` 두 팩토리 함수로 등록한다.

논문에서 Inception-ResNet은 Inception 블록의 concat을 residual addition으로 바꾸고, 더하기 직전에 1×1 linear conv로 채널을 맞춘 뒤 ReLU를 적용하는 형태로 제시된다(Fig. 10–13, Fig. 15, Fig. 16–19). 또한 residual scaling(Fig. 20)을 안정화 장치로 강조하지만, Lucid 구현은 scaling 계수 곱을 별도로 넣지 않고 `out + residual`을 곧바로 수행한다는 점이 차이로 남는다.

### 1️⃣ 베이스 클래스: `InceptionResNet`
Inception-ResNet 계열은 Inception-v4의 `Inception` 베이스를 공유하지 않고, 별도의 `InceptionResNet(nn.Module)` 베이스를 둔다. 여기서는 `num_classes`만 공통 속성으로 저장하고, 실제 네트워크 구성 요소는 하위 클래스가 채운다.

```python
class InceptionResNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.stem: nn.Module
        self.conv: nn.Sequential
        self.fc: nn.Sequential
```

타입 힌트로 `stem`, `conv`, `fc`가 존재할 것임을 표시해 두고, v1/v2가 각각 어떤 stem과 어떤 블록 반복을 쓰는지로 모델을 구분한다.

### 2️⃣ Stem: v1은 `_InceptionResStem`, v2는 `_InceptionStem_V4`
논문에서 Inception-ResNet-v1은 Fig. 14의 stem을, Inception-ResNet-v2는 Fig. 3/16 계열과 연결되는 구성을 사용한다. Lucid도 v1과 v2의 stem을 다르게 잡는다.

먼저 v1에서 사용하는 `_InceptionResStem`은 단일 `nn.Sequential`로 구성된 비교적 직선적인 stem이다.

```python
class _InceptionResStem(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 32, kernel_size=3, stride=2),
            nn.ConvBNReLU2d(32, 32, kernel_size=3),
            nn.ConvBNReLU2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ConvBNReLU2d(64, 80, kernel_size=1, padding=0),
            nn.ConvBNReLU2d(80, 192, kernel_size=3),
            nn.ConvBNReLU2d(192, 256, kernel_size=3, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
```

이 stem은 초기 단계에서 conv와 maxpool로 해상도를 단계적으로 줄이고, 마지막에 256 채널까지 확장해 35×35 계열 블록이 들어가기 좋은 텐서를 만든다.

반면 v2는 Inception-v4에서 사용한 `_InceptionStem_V4`를 그대로 재사용한다(`inception.py`에 구현). 따라서 v2는 Fig. 3의 병렬 concat 기반 stem을 그대로 따른다고 볼 수 있다.

### 3️⃣ 35×35 블록: `_InceptionResModule_A`
Inception-ResNet-A는 35×35 grid에서 반복되는 residual 블록이며, Lucid 구현은 `_InceptionResModule_A`에 들어 있다. 이 블록은 3개의 브랜치를 concat한 뒤, 1×1 `conv_linear`로 채널을 맞추고 residual을 더한 뒤 ReLU를 적용한다.

```python
class _InceptionResModule_A(nn.Module):
    def __init__(self, in_channels: int, version: Literal["v1", "v2"]) -> None:
        super().__init__()

        if version == "v1":
            cfg = [32, 32, 256]
        elif version == "v2":
            cfg = [48, 64, 384]
        else:
            raise ValueError("Invalid version.")

        self.branch1 = nn.ConvBNReLU2d(in_channels, 32, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 32, kernel_size=1),
            nn.ConvBNReLU2d(32, 32, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 32, kernel_size=1),
            nn.ConvBNReLU2d(32, cfg[0], kernel_size=3, padding=1),
            nn.ConvBNReLU2d(cfg[0], cfg[1], kernel_size=3, padding=1),
        )

        self.conv_linear = nn.Conv2d(32 + 32 + cfg[1], cfg[2], kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        out = lucid.concatenate([branch1, branch2, branch3], axis=1)

        out = self.conv_linear(out)
        out = out + residual
        out = self.relu(out)

        return out
```

여기서 핵심 구현 포인트는 `conv_linear`가 `nn.Conv2d`(BN/ReLU 없음)라는 점이다. 논문이 설명하는 1×1 linear projection이 코드에서 그대로 드러난다. 또한 논문이 강조한 residual scaling($\alpha$ 곱)은 이 구현에 직접 등장하지 않으며, residual은 그대로 더해진다.

### 4️⃣ Reduction-A: `_InceptionReduce_V4A`를 Table 1 값으로 재사용
Inception-ResNet-v1/v2 모두 35×35에서 17×17로 내려가는 지점에서 `_InceptionReduce_V4A`를 사용한다. 이는 v4에서 설명한 Reduction-A 구현을 그대로 재사용하는 구조다.

**v1**은 `(k, l, m, n) = (192, 192, 256, 384)`를 사용하고,

```python
modules.append(_InceptionReduce_V4A(256, k=192, l=192, m=256, n=384))
```

**v2**는 `(256, 256, 384, 384)`를 사용한다.

```python
modules.append(_InceptionReduce_V4A(384, k=256, l=256, m=384, n=384))
```

이는 논문 Table 1의 Inception-ResNet-v1/v2 행과 일치한다. 즉, Lucid 구현은 Reduction-A를 공통 코드 + 파라미터로 구성하는 방식으로 논문 설계를 반영한다.

### 5️⃣ 17×17 블록: `_InceptionResModule_B`
Inception-ResNet-B는 17×17 grid에서 반복되는 residual 블록이며, **비대칭 커널(1×7, 7×1)** 을 포함한 브랜치를 가진다.

```python
class _InceptionResModule_B(nn.Module):
    def __init__(self, in_channels: int, version: Literal["v1", "v2"]) -> None:
        super().__init__()

        if version == "v1":
            cfg = [128, 128, 128, 896]
        elif version == "v2":
            cfg = [192, 160, 192, 1152]
        else:
            raise ValueError("Invalid version.")

        self.branch1 = nn.ConvBNReLU2d(in_channels, cfg[0], kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 128, kernel_size=1),
            nn.ConvBNReLU2d(128, cfg[1], kernel_size=(1, 7), padding=(0, 3)),
            nn.ConvBNReLU2d(cfg[1], cfg[2], kernel_size=(7, 1), padding=(3, 0)),
        )

        self.conv_linear = nn.Conv2d(cfg[0] + cfg[2], cfg[3], kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        out = lucid.concatenate([branch1, branch2], axis=1)

        out = self.conv_linear(out)
        out = out + residual
        out = self.relu(out)

        return out
```

이 블록도 A와 마찬가지로 concat 이후 `conv_linear`로 채널을 맞춘 뒤 더한다. `branch2`의 (1×7, 7×1) 연쇄는 논문에서 Inception 계열이 연산 효율을 위해 사용해 온 factorization이 residual 변형에서도 유지됨을 보여준다.

### 6️⃣ 17×17 → 8×8 Reduction: `_InceptionResReduce`
논문에서 Inception-ResNet 쪽 reduction은 Fig. 12(`v1`), Fig. 18(`v2`) 계열로 제시된다. Lucid 구현은 이를 `_InceptionResReduce`로 구현하며, maxpool 경로를 포함해 4개 경로를 concat한다.

```python
class _InceptionResReduce(nn.Module):
    def __init__(self, in_channels: int, version: Literal["v1", "v2"]) -> None:
        super().__init__()

        if version == "v1":
            cfg = [256, 256, 256]
        elif version == "v2":
            cfg = [288, 288, 320]
        else:
            raise ValueError("Invalid version.")

        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.branch2 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 256, kernel_size=1),
            nn.ConvBNReLU2d(256, 384, kernel_size=3, stride=2),
        )

        self.branch3 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 256, kernel_size=1),
            nn.ConvBNReLU2d(256, cfg[0], kernel_size=3, stride=2),
        )

        self.branch4 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 256, kernel_size=1),
            nn.ConvBNReLU2d(256, cfg[1], kernel_size=3, padding=1),
            nn.ConvBNReLU2d(cfg[1], cfg[2], kernel_size=3, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return lucid.concatenate(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            axis=1,
        )
```

채널 구성만 v1/v2에서 다르게 가져가고, 구조 자체는 동일하다. 즉 downsampling 병목을 병렬 경로로 분산한다는 원칙은 Inception-v4의 reduction과 동일하고, 단지 residual 변형에 맞는 채널 수로 조정된다.

### 7️⃣ 8×8 블록: `_InceptionResModule_C`
마지막 8×8 구간의 residual 블록은 `_InceptionResModule_C`로 구현된다. 이 블록은 (1×3, 3×1) 비대칭 커널을 사용해 3×3 계열을 분해하는 구조를 포함한다.

```python
class _InceptionResModule_C(nn.Module):
    def __init__(self, in_channels: int, version: Literal["v1", "v2"]) -> None:
        super().__init__()

        if version == "v1":
            cfg = [192, 192, 1792]
        elif version == "v2":
            cfg = [224, 256, 2048]
        else:
            raise ValueError("Invalid version.")

        self.branch1 = nn.ConvBNReLU2d(in_channels, 192, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 192, kernel_size=1),
            nn.ConvBNReLU2d(192, cfg[0], kernel_size=(1, 3), padding=(0, 1)),
            nn.ConvBNReLU2d(cfg[0], cfg[1], kernel_size=(3, 1), padding=(1, 0)),
        )

        self.conv_linear = nn.Conv2d(192 + cfg[1], in_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        out = lucid.concatenate([branch1, branch2], axis=1)

        out = self.conv_linear(out)
        out = out + residual
        out = self.relu(out)

        return out
```

`conv_linear`의 출력 채널이 `in_channels`인 점이 중요하다. 즉, C 블록은 입력과 동일한 채널 수의 residual을 만들어 더하는 형태로 고정된다. 논문에서 C 블록이 최종 표현을 안정적으로 누적하는 단계로 동작한다는 감각과 잘 맞는다.

### 8️⃣ 전체 네트워크: `InceptionResNet_V1`
**Inception-ResNet-v1**은 A 블록 5개, B 블록 10개, C 블록 5개를 사용한다. Lucid 구현도 반복 횟수를 그대로 따른다.

```python
class InceptionResNet_V1(InceptionResNet):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__(num_classes)
        in_channels = 3

        self.stem = _InceptionResStem(in_channels)

        modules = []
        for _ in range(5):
            modules.append(_InceptionResModule_A(256, version="v1"))
        modules.append(_InceptionReduce_V4A(256, k=192, l=192, m=256, n=384))

        for _ in range(10):
            modules.append(_InceptionResModule_B(896, version="v1"))
        modules.append(_InceptionResReduce(896, version="v1"))

        for _ in range(5):
            modules.append(_InceptionResModule_C(1792, version="v1"))

        self.conv = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.8)
        self.fc = nn.Linear(1792, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.conv(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
```

Inception-v4와 마찬가지로 마지막은 global average pooling 뒤 dropout과 linear 분류기다. 다만 Lucid의 `Dropout(p=0.8)`은 drop 확률 0.8이라는 의미이므로, 논문 Fig. 15의 dropout keep 0.8과는 숫자 해석이 다르다. 구현을 논문과 동일하게 맞추려면 Lucid의 Dropout 정의에 맞는 확률로 재설정되어야 한다.

### 9️⃣ 전체 네트워크: `InceptionResNet_V2`
**Inception-ResNet-v2**도 동일하게 A 5개, B 10개, C 5개라는 큰 반복 구조를 가지며(Fig. 15), v2는 v4 stem을 재사용한다.

```python
class InceptionResNet_V2(InceptionResNet):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__(num_classes)
        in_channels = 3

        self.stem = _InceptionStem_V4(in_channels)

        modules = []
        for _ in range(5):
            modules.append(_InceptionResModule_A(384, version="v2"))
        modules.append(_InceptionReduce_V4A(384, k=256, l=256, m=384, n=384))

        for _ in range(10):
            modules.append(_InceptionResModule_B(1152, version="v2"))
        modules.append(_InceptionResReduce(1152, version="v2"))

        for _ in range(5):
            modules.append(_InceptionResModule_C(2144, version="v2"))

        self.conv = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.8)
        self.fc = nn.Linear(2144, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.conv(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
```

v2는 v1보다 채널 수가 더 크고(`384 → 1152 → 2144`), 따라서 논문이 말하는 넓은 residual Inception에서의 안정화 문제와 더 직접적으로 연결되는 변형이다. 하지만 앞서 말했듯 Lucid 구현은 residual scaling을 별도로 넣지 않으므로, 논문이 강조한 안정화 장치가 구현 상에서 재현되지는 않는다.

### 🔟 모델 등록 함수: `inception_resnet_v1`, `inception_resnet_v2`
Lucid의 모델 레지스트리에는 두 변형이 각각 등록된다.

```python
@register_model
def inception_resnet_v1(num_classes: int = 1000, **kwargs) -> InceptionResNet:
    return InceptionResNet_V1(num_classes, **kwargs)


@register_model
def inception_resnet_v2(num_classes: int = 1000, **kwargs) -> InceptionResNet:
    return InceptionResNet_V2(num_classes, **kwargs)
```

즉 Lucid에서 Inception-ResNet을 쓰려면 `inception_resnet_v1()` 또는 `inception_resnet_v2()`를 호출하면 된다.

---

## ✅ 정리
**Inception-v4** 논문은 Inception 계열의 설계 철학(멀티브랜치 병렬 경로로 다양한 receptive field를 효율적으로 다루고, factorization으로 계산량을 제어하는 방식)을 한 단계 더 정돈된 형태로 정리한 작업이다. Inception-v4는 grid size별로 균일한 Inception-A/B/C 블록을 반복하는 구조로 단순화되었고, Stem과 reduction 모듈에서도 병렬 경로를 통해 다운샘플링 병목을 분산시키는 원칙을 유지한다. 동시에 **Inception-ResNet**은 concat 대신 add를 쓰는 residual 결합으로 학습을 크게 가속하며, 폭이 커질수록 residual scaling이 안정성을 좌우하는 실전적 요소임을 보여준다. 전체적으로 이 논문은 모델 구조의 발전이 새로운 트릭 추가가 아니라 모듈의 반복과 스케일링, 그리고 최적화 안정성이라는 공학적 문제를 어떻게 다루는지 보여주는 좋은 사례다.

#### 📄 출처
Szegedy, Christian, et al. "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning." arXiv:1602.07261, 2016.
