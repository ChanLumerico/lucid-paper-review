# [LeNet] Gradient-Based Learning Applied to Document Recognition

이 글은 LeCun et al.(1998)의 _"Gradient-Based Learning Applied to Document Recognition"_ 을 원 논문의 섹션 전개(문제 제기 → 배경 → 수식/알고리즘 → 직관 → 실험/응용) 흐름을 최대한 그대로 따라가며 상세히 분석해 보았다. 논문이 실제로 하고 있는 이야기의 핵심은 "CNN(LeNet-5)으로 필기 숫자 인식이 잘 된다" 정도에서 끝나지 않는다.

저자들은 더 큰 관점에서,

- **고정(feature engineering) + 학습 가능한 분류기**라는 전통적 파이프라인의 한계를 짚고,
- 픽셀/신호 같은 고차원 입력에서 **미분 가능한 손실을 정의**해 **그래디언트 기반 학습(gradient-based learning)**으로 전체 시스템을 훈련하는 관점이 어떻게 확장되는지,
- 그리고 다중 모듈(분할, 인식, 언어모델, 탐색 등)로 구성된 문서 인식 시스템을 **Graph Transformer Networks (GTN)**라는 틀로 "끝단 성능을 직접 최적화"하도록 **전역 학습(global training)**하는 길까지 제시한다.

논문 초반의 메시지를 한 문장으로 옮기면, "더 나은 패턴 인식 시스템은 (가능한 한) 수작업 규칙을 줄이고, (가능한 한) 학습을 늘리는 방식으로 설계될 수 있다"이다. 다만 저자들은 "학습이 모든 것을 해결한다"가 아니라, **아키텍처/표현 설계라는 ‘최소한의 사전지식(prior knowledge)’**을 어떻게 넣어야 학습이 효과적으로 작동하는지를 굉장히 구체적으로 보여준다.

---

## 1️⃣ 학습 중심 설계의 중요성

### 🔹 전통적 패턴 인식 시스템의 두 모듈 구조

논문은 먼저 "패턴 인식 시스템을 보통 어떻게 만들었는가"를 도식으로 상기시키며 시작한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/88670caf-a4bf-4d7e-910d-f35bacbe90ec/image.png" width="70%">
</p>

이 구조의 장점은 분업이 쉽다는 것이다. 특징 추출기에는 사람이 문제별로 **"불변성이 있어야 한다(shift/scale/rotation 등)"** 같은 지식을 반영해 설계하고, 분류기는 비교적 일반적인 학습 알고리즘(선형 분류기, MLP 등)을 붙인다. 하지만 논문이 강조하는 문제는 정확도가 결국 **특징 설계의 품질**에 크게 의존한다는 점이다. 좋은 특징을 설계하는 일은 어렵고, 문제마다 다시 해야 하고, 시스템 전체를 통합 관점에서 최적화하기 어렵다.

### 🔸 90년대 후반에 학습 중심 설계가 가능해진 배경

저자들은 학습 중심 설계가 가능해진 배경을 **세 가지**로 정리한다.

1. **계산 자원의 증가**: 빠른 산술 연산과 저렴한 컴퓨팅이 "수치 최적화(numerical methods)"를 실용적으로 만들었다.  
2. **대규모 데이터셋의 등장**: 시장이 큰 문제(필기 인식 등)는 학습을 ‘받쳐줄’ 데이터가 쌓이기 시작했다.  
3. **고차원 입력을 다루는 학습 기법의 성숙**: 다층 신경망 + 역전파(backprop)가 고차원 입력에서 복잡한 결정 경계를 만들 수 있게 했다.

이 셋이 합쳐지면서, 손으로 특징을 깎아 넣기보다는 **픽셀 입력에 가까운 표현에서** 학습이 많은 일을 담당하도록 하는 쪽으로 무게가 이동한다.

### 🔹 사전 지식이 여전히 필요한 이유

여기서 논문이 중요한 균형을 잡는다. "학습이 유용하지만, 학습만으로 성공할 수 없다"는 점이다. 특히 다층 신경망을 쓰더라도, 문제 구조를 반영해 **네트워크 아키텍처를 특화(specialize)**하는 것이 핵심적인 prior knowledge 주입 방식이라고 말한다. 이 논문에서 그 대표 예가 **Convolutional Neural Networks(CNN)**이고, 이후 문서 인식으로 확장할 때는 "그래프 기반 상태 표현 + 그래프 변환 모듈"이라는 prior를 **GTN** 구조로 제공한다.

### 🔸 일반화 관점: 데이터 수, 모델 용량, 정규화

논문은 일반화 오차의 고전적 관점을 간단한 식으로 소개한다. 우선 표기부터 정리해보자면,

- $E_{test}$: 테스트 셋에서의 기대 오류율(일반화 성능).  
- $E_{train}$: 학습 셋에서의 오류율(훈련 성능).  
- $P$: 학습 샘플 수.  
- $h$: 모델의 _유효 용량(effective capacity)_ 혹은 복잡도 척도.  
- $k$: 상수, $\alpha \in [0.5, 1.0]$: 경험적 지수.

논문이 제시하는 근사 관계는
$$
E_{test} - E_{train} \approx k\left(\frac{h}{P}\right)^{\alpha}.
$$
직관은 명확하다.

- $P$가 커지면 gap이 줄어든다(데이터가 많을수록 일반화가 좋아지기 쉬움).  
- $h$가 커지면 보통 $E_{train}$은 줄지만, gap은 커질 수 있다(과적합 위험).  

따라서 "최적의 용량"이 존재하며, 학습 알고리즘은 단순히 $E_{train}$만 줄이는 것이 아니라 이 gap까지 고려해야 한다. 고전적으로는 **구조적 위험 최소화(Structural Risk Minimization)**, 혹은 경험 위험에 정규화 항을 붙인 형태로 표현한다:
$$
E_{train} + \lambda H(W).
$$
여기서 $W$는 파라미터 벡터, $H(W)$는 정규화(regularization) 함수, $\lambda$는 그 강도다.

이 관점은 뒤에서 다시 등장한다. 이 논문이 "학습이 중요하다"라고 말하면서도, 무작정 큰 모델이 아니라 **적절한 아키텍처로 용량을 통제**하고, **데이터를 늘리거나(distortion) 정규화에 준하는 효과**를 얻는 전략을 강조하기 때문이다.

### 🔹 Gradient-Based Learning의 기본 형식

논문에서 $E(W)$는 "현재 파라미터 $W$에서의 손실(loss)"로 생각하면 된다. (분류 문제면 오분류를 줄이도록, 혹은 확률/점수의 형태로 정의된 목적 함수를 최소화하도록.) 그래디언트 기반 학습의 기본 업데이트는
$$
W_{k} = W_{k-1} - \epsilon \frac{\partial E(W)}{\partial W}
$$
(여기서 $\epsilon$은 학습률 혹은 step size)이고, 샘플 단위로 노이즈가 있는 기울기를 쓰면 온라인/확률적 업데이트:
$$
W_{k} = W_{k-1} - \epsilon \frac{\partial E^{p}(W)}{\partial W},
$$
여기서 $E^p$는 샘플 $p$에 대한 즉시 손실(instantaneous loss)이다.

논문이 이 기본식을 굳이 다시 적는 이유는, 이후에 등장하는 CNN, GTN, forward/Viterbi 기반 손실이 모두 결국 "$E(W)$를 어떻게 정의하고, 그 미분을 어떻게 전파하느냐"의 문제로 귀결되기 때문이다.

### 🔸 Backpropagation의 요점

논문은 역전파를 **"다층 비선형 시스템에서 그래디언트를 효율적으로 계산하는 절차"** 로 소개한다. 핵심은 **연쇄법칙(chain rule)** 이다. 여러 층의 연산이 합성(compose)되어 있을 때,

- 최종 손실의 미분은 각 층의 국소 미분(local derivative)을 곱(또는 적절히 합)해 아래로 내려갈 수 있고,
- 이를 통해 모든 파라미터에 대한 $\partial E / \partial W$를 한 번의 forward + backward로 얻는다.

이 글에서는 논문의 목적에 맞춰, 역전파의 수학적 유도 전체를 재현하기보다는, 뒤에서 CNN의 "가중치 공유"나 GTN의 "그래프 위 동적 계획법 + 미분"이 어떻게 같은 관점으로 묶이는지를 중심으로 따라가겠다.

---

## 2️⃣ Convolutional Neural Networks: 필기 숫자 인식을 위한 설계

이 섹션에서는 저자들이 "필기 숫자"라는 단일 객체 인식(single object recognition) 문제에서, **특징 추출을 손으로 만드는 대신 아키텍처로 불변성과 구조를 넣고 학습으로 해결**하는 대표 사례로 CNN(LeNet-5)을 제시한다.

### 🔹 Convolutional Network의 3가지 설계 아이디어
논문은 CNN이 이동/스케일/왜곡 불변성을 어느 정도 갖도록 하는 세 가지 아이디어를 정리한다.

1. **Local receptive fields(국소 수용장)**: 한 유닛이 이전 층의 "작은 지역"만 본다. 이는 (시각 피질의 국소 민감 뉴런에 대한 생물학적 영감과 함께) 엣지/코너/끝점 같은 국소 특징을 잡는 데 유리하다.  
2. **Shared weights(가중치 공유, weight replication)**: 이미지의 서로 다른 위치에서 "같은 종류의 특징"을 검출할 때 같은 필터(가중치)를 쓰도록 강제한다. 이렇게 하면 파라미터 수가 줄어들고, 특징이 위치 이동에 강해진다. 공유되는 유닛들의 출력 집합을 **feature map**이라고 부른다.  
3. **Subsampling(공간/시간 서브샘플링)**: 풀링/서브샘플링으로 공간 해상도를 줄이며, 작은 이동에 대한 불변성을 더 얻고 계산량을 줄인다.

이 셋이 결합되면 "이미지 전체에서 같은 필터가 슬라이딩하며 특징을 뽑고, 중간중간 다운샘플링으로 변형에 둔감해지는" 구조가 된다. 이후 LeNet-5의 층 구성`(C1/S2/C3/S4/C5/F6/Output)`은 이 아이디어를 구체화한 사례다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/b938ebe0-da44-4a6e-a6e2-709c8c31749c/image.png" width="80%">
</p>

### 🔸 LeNet-5 아키텍처: 층별 구성과 의도

논문이 설명하는 LeNet-5는 입력을 $32\times 32$로 맞춘 필기 숫자 이미지로 시작한다. (원래 숫자들은 $28\times 28$ 정도지만, $32\times 32$로 확장하는 이유는 "가장 높은 층의 수용장 중심에 판별적인 스트로크 특징(끝점, 코너 등)이 들어올 수 있도록" 여유를 두기 위해서라고 설명한다.)

입력 픽셀은 배경(흰색)을 약 $-0.1$, 전경(검은색)을 $1.175$로 정규화해 평균을 $0$에 가깝게, 분산을 $1$에 가깝게 하여 학습을 빠르게 한다고 적는다.

논문 표기대로, 합성곱 층은 $C_x$, 서브샘플링 층은 $S_x$, 완전연결 층은 $F_x$로 쓴다.

#### C1: 6개의 feature map, $5\times 5$ 합성곱
- 입력: $32\times 32$  
- 커널: $5\times 5$ (패딩 없이 적용하면 공간 크기 $28\times 28$)  
- 맵 수: 6  

각 유닛은 입력의 $5\times 5$ 지역을 보고(국소 수용장), 같은 map 내부에서는 동일한 필터를 공유한다(가중치 공유). 따라서 "같은 특징 검출기"가 위치만 달리하며 등장한다.

#### S2: 6개의 feature map, $2\times 2$ 서브샘플링 + 학습 가능한 스케일/bias
논문에서 S2는 단순 평균풀링이 아니라, $2\times 2$ 윈도우를 더한 뒤 학습 가능한 계수와 바이어스를 적용하고 비선형 함수를 통과시키는 형태로 설명된다. 즉 각 map마다 (혹은 각 유닛 타입마다) 학습 가능한 파라미터가 있다.

핵심은 해상도를 절반으로 줄여 $14\times 14$가 되며, 작은 이동에 대해 더 둔감해진다는 점이다.

#### C3: 16개의 feature map, 부분 연결(partial connectivity) $5\times 5$ 합성곱
C3는 16개의 feature map을 갖는다. 흥미로운 점은 "모든 S2 map을 모든 C3 map에 완전 연결"하지 않고, **부분 연결**을 쓴다는 것이다. 논문은 그 이유를 두 가지로 설명한다.

1. 연결 수를 합리적 범위로 유지(파라미터/연산량 억제).  
2. 더 중요하게, **대칭성 파괴(symmetry breaking)**: 모든 map이 동일한 입력을 보면 비슷한 특징을 학습할 수 있는데, 입력 조합을 다르게 주면 각 map이 서로 다른(상보적인) 특징을 더 잘 분화해 학습하도록 유도한다.

<p>
  <img src="https://velog.velcdn.com/images/lumerico284/post/96119695-0673-4204-b63c-34c22439c38e/image.png" width="70%">
</p>

#### S4: 16개의 feature map, 다시 $2\times 2$ 서브샘플링
S4는 C3의 각 map을 $5\times 5$로 줄인다(해상도 절반). 구성은 S2와 유사한 "합 → 스케일/바이어스 → 비선형" 형태다.

#### C5: 120개의 feature map, 사실상 "전결합"에 가까운 합성곱
논문에서는 C5를 합성곱 층으로 부르지만, S4의 크기가 $5\times 5$이기 때문에 $5\times 5$ 커널을 쓰면 출력 map 크기가 $1\times 1$이 된다. 즉 각 C5 유닛은 S4의 전체 공간을 본다. 이 때문에 C5는 결과적으로 **"S4 전체에 대한 완전 연결"**처럼 동작한다.

다만 논문이 C5를 굳이 합성곱 층으로 부르는 이유도 적는다. 입력 이미지가 더 커지면(예: 더 긴 단어 이미지 등) 같은 구조를 유지할 때 C5의 출력이 $1\times 1$이 아니라 더 큰 map이 되며, 이때 C5는 다시 "공간적으로 슬라이딩하는 합성곱"이 되기 때문이다. 이 관점은 뒤에서 SDNN(공간 복제)과 연결된다.

#### F6: 84 유닛의 완전 연결 층
F6는 $84$ 유닛이며 C5와 완전 연결된다. 논문은 "$84$라는 숫자"가 출력층 설계(분산 코드, ASCII용 코드북)와 연결된다는 점을 뒤에서 설명한다.

#### 비선형 함수: 스케일된 tanh
논문은 각 유닛의 가중합 $a_i$에 대해 상태 $x_i$를
$$
x_i = f(a_i)
$$
로 두고, squashing function으로
$$
f(a) = A \tanh(Sa)
$$
를 사용한다고 쓴다. 여기서 $A$는 출력 범위(진폭), $S$는 원점에서의 기울기를 조절한다. 논문 본문에는 $A=1.7159$가 언급되며, 구체적 선택 이유는 Appendix A에서 설명한다고 한다.

이 선택의 요지는 **"포화(saturation)"** 를 피하고(포화되면 기울기가 작아져 학습이 느려짐), 유닛이 비선형성이 잘 살아있는 구간에서 동작하도록 유도하는 것이다.

### 🔹 출력층: Euclidean RBF와 분산 코드(distributed code)
LeNet-5의 출력층은 흔히 떠올리는 softmax가 아니라, **Euclidean Radial Basis Function(RBF)** 유닛으로 설명된다. 클래스마다 하나의 RBF 유닛이 있고(숫자라면 10개), 각 RBF는 F6의 84차원 벡터를 입력으로 받는다.

RBF 유닛 $i$의 출력(논문에서는 "penalty"로 해석)은, 입력 $x \in \mathbb{R}^{84}$와 그 클래스의 파라미터 벡터 $w_i \in \mathbb{R}^{84}$ 사이의 유클리드 거리(제곱합)로 표현된다:
$$
y_i = \sum_{j}(x_j - w_{ij})^2.
$$
즉 $x$가 $w_i$에 가까울수록 $y_i$가 작아지고(낮은 penalty), 멀수록 커진다.

여기서 중요한 설계가 "$w_i$를 어떻게 두느냐"인데, 논문은 초기에 $w_i$를 **수작업으로 만든 분산 코드**(예: 7×12 비트맵을 펼친 84차원)로 두고, 각 성분을 $\pm 1$로 설정했다고 설명한다. 이 선택이 주는 직관은 다음과 같다.

- 서로 헷갈리기 쉬운 문자들(예: O/0, 1/l/I 등)은 코드가 비슷해지도록 만들면, 출력 RBF penalty도 비슷해져 "후단 언어모델/문맥 처리"가 교정할 여지를 가진다.  
- "1-of-N"처럼 대부분의 출력이 항상 0에 가까워야 하는 코드는 시그모이드 유닛과 상성이 나쁘고(항상 꺼져 있어야 하므로), 거절(rejection) 같은 문제에도 불리할 수 있다.  
- $\pm 1$은 비선형 유닛이 포화되지 않도록 하며(본문에서는 "곡률이 큰 구간"에 놓인다고 설명), 학습 안정성에 기여한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/2735dc3f-bc8d-496a-8357-29f56fa4a0b8/image.png" width="70%">
</p>

### 🔸 손실 함수: MSE(MLE)와 MAP 기준
이제 이 출력 구조 위에서 $E(W)$를 어떻게 정의하는지가 나온다. (여기서 $W$에는 CNN 가중치뿐 아니라 필요하면 RBF 파라미터도 포함될 수 있다.)

#### 가장 단순한 기준: MSE(=MLE) 관점
논문은 가장 단순한 출력 손실을 "Maximum Likelihood Estimation(MLE)"로 부르며, 여기서는 "Minimum Mean Squared Error(MSE)"와 동치라고 말한다. 직관적으로는 "정답 클래스의 RBF 출력(거리) $y_{D^p}(Z^p, W)$를 작게 만들자"는 것이다. 학습 샘플 $p=1..P$에 대해,
$$
E(W) = \frac{1}{P}\sum_{p=1}^{P} y_{D^p}(Z^p, W).
$$
여기서
- $Z^p$: $p$번째 입력(이미지),
- $D^p$: 그 입력의 정답 클래스 라벨,
- $y_{D^p}(Z^p, W)$: 정답 클래스 RBF 유닛의 penalty(거리).

즉 "정답 클래스 중심에 F6 표현이 가까워지도록" 밀어 넣는 학습이다.

#### 이 기준의 문제점
논문이 말하는 첫 번째 문제는, RBF 파라미터까지 학습 가능하게 두면 **붕괴(collapse)**하는 "사소하지만 최악의 해"가 생긴다는 점이다. 모든 RBF 중심이 같아지고, F6 출력도 상수로 고정되면 입력을 무시해도 $y_i$가 모두 0이 되는 형태의 해가 가능해진다(즉 모델이 "아무것도 보지 않는다"). 이는 분류기로서 의미가 없다.

두 번째 문제는 "정답을 낮추는 것만으로는" 다른 클래스와의 **경쟁(competition)**이 약하다는 것이다. 분류 문제에서는 정답 점수를 올리는(또는 penalty를 내리는) 동시에 오답 점수는 내리거나(또는 penalty를 올리는) 상대적 분리가 필요하다.

#### 더 판별적인 기준: MAP(최대 사후확률) 형태
그래서 논문은 더 판별적인(discriminative) 기준을 제시한다. 이름은 **MAP(maximum a posteriori)** 로 불린다.

핵심 아이디어는 _"정답 penalty는 내리고, 오답 penalty는 올리되, 이미 충분히 큰 오답까지 무한정 밀어올리지는 않도록"_ 로그-합 형태로 경쟁을 주는 것이다. 논문 식(9)의 형태를 정리하면,

$$
E(W)=\frac{1}{P}\sum_{p=1}^{P}
 \left[
 y_{D^p}(Z^p,W)
 + \log\!\left(
e^{-j} + \sum_{i\neq D^p} e^{-y_i(Z^p,W)}
\right)
 \right].
$$
여기서 $j>0$는 상수로, "이미 penalty가 큰 클래스는 더 올려도 의미가 적다"는 점을 반영해 로그 항을 안정화한다.

직관적으로 해석하면:

- 첫 항 $y_{D^p}$는 정답을 당기는(pull) 역할.  
- 두 번째 항은 오답들의 $e^{-y_i}$가 커지면(즉 오답 penalty가 작아져 위험해지면) 커지므로, 오답들을 밀어내는(push) 역할.  
- 로그-합이기 때문에 "가장 위험한 오답(작은 penalty)"이 주로 영향을 주지만, 여러 오답이 동시에 위험한 경우도 반영한다.

이 손실은 **collapse를 억제하는 효과**가 있다고 논문은 설명한다. 오답과의 경쟁이 명시되면, 모든 중심이 같아지는 방향이 손실을 낮추기 어렵기 때문이다.

### 🔹 핵심 알고리즘: LeNet-5 forward 의사코드
논문은 층별 정의를 텍스트로 풀어 쓰지만, 이해를 돕기 위해 "논문 아키텍처 그대로"를 전방 계산 의사코드로 정리하면 다음과 같다. (여기서는 논문 구조를 유지하기 위해 RBF 출력까지 포함한다.)

```pseudocode
Algorithm 1: LeNet-5 Forward (paper structure)
Input: image X (32×32), parameters of conv/subsample layers, F6 weights, RBF centers {w_i}
Output: penalties {y_i} for each class i

1:  C1 ← Conv5×5(X; shared weights)            # 6 maps, 28×28
2:  C1 ← f(C1)                                 # scaled tanh
3:  S2 ← Subsample2×2(C1; per-map scale/bias)  # 6 maps, 14×14
4:  S2 ← f(S2)
5:  C3 ← Conv5×5_partial(S2; partial connectivity)  # 16 maps, 10×10
6:  C3 ← f(C3)
7:  S4 ← Subsample2×2(C3; per-map scale/bias)  # 16 maps, 5×5
8:  S4 ← f(S4)
9:  C5 ← Conv5×5(S4; 120 maps)                 # 120 maps, 1×1
10: C5 ← f(C5)
11: F6 ← FullyConnected(C5; 84 units)
12: F6 ← f(F6)
13: for each class i do
14:     y_i ← Σ_j (F6_j − w_{ij})^2            # Euclidean RBF penalty
15: end for
16: return {y_i}
```

이 알고리즘이 중요한 이유는, CNN은 단순한 필터들의 묶음이 아니라, **손실 $E(W)$가 정의되면** 그 손실이 출력의 penalty를 통해 다시 F6, C5, $\ldots$, 입력 쪽으로 기울기를 전파해 **필터(가중치) 자체가 학습되는 특징 추출기**가 된다는 점을 명확히 보여주기 때문이다.

---

## 3️⃣ MNIST에서의 성능과 비교

세 번째 섹션에서는 LeNet-5를 "문서 인식의 한 사례"로서 먼저 정량적으로 검증하는 파트다. 데이터셋, 학습 설정, 그리고 다양한 분류기들과의 비교를 다룬다.

### 🔸 데이터셋: Modified NIST(MNIST)와 전처리
논문은 NIST의 Special Database 3(SD-3)을 기반으로 한 MNIST(Modified NIST) 데이터셋을 소개한다. 핵심은 "학습/평가가 표준화된 필기 숫자 벤치마크"라는 점이다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/2386c10d-42a6-4a29-8670-8699297a4e4f/image.png" width="60%">
</p>

### 🔹 학습 설정: 반복 횟수, 학습률 스케줄, 2차 정보 활용
논문에서 LeNet-5는 전체 훈련 데이터를 20번(pass) 도는 식으로 학습한다. 학습률 $\eta$는 일정하지 않고 패스에 따라 감소한다. 텍스트로 제시된 스케줄은 대략:

- 처음 2 pass: 0.0005  
- 다음 3 pass: 0.0002  
- 다음 3 pass: 0.0001  
- 다음 4 pass: 0.00005  
- 이후: 0.00001

여기서 흥미로운 점은, 논문이 단순 SGD만 쓰지 않고, 각 pass 전 **대각 헤시안(diagonal Hessian) 근사**를 일부 샘플(예: 500개)로 재평가하여 고정한 뒤 그 pass 동안 사용한다는 설명이다. 지금 단계에서는 핵심만 잡자면,

- 파라미터마다 "스텝 크기"가 같지 않을 수 있고,
- 2차 정보(곡률)를 대각 근사로 이용해 step size를 조정하면 학습이 더 안정/가속될 수 있다

정도로 이해하면 된다.

### 🔸 결과(정량): test error 0.95% → distortion으로 0.8%
논문은 학습 곡선을 제시하며, 왜 "over-training(과훈련)"이 흔히 관찰되지만 여기서는 명확히 관찰되지 않았는지 논의한다.

- 60,000개 학습 데이터에서 약 10 pass 이후 test error가 **0.95%** 수준으로 안정화.  
- train error는 19 pass에서 **0.35%** 수준까지 내려간다고 보고한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/2b61e217-2777-4423-b72a-d96d2fea89c9/image.png" width="60%">
</p>

또한 학습 데이터 크기를 15k/30k/60k로 바꿔가며 성능 변화를 보고하고, 더 많은 데이터가 도움이 됨을 보여준다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/8e569fd5-dffc-4dec-84f0-e40b2cb1a722/image.png" width="60%">
</p>

이를 검증하기 위해, 원본 60,000개에 더해 랜덤 왜곡(random distortions)으로 **추가 540,000개**를 생성하여 총 600,000개로 학습하는 실험을 한다. 왜곡은 평면 아핀 변환(이동, 스케일, 스퀴즈, 시어링 등)의 조합으로 생성한다.

그 결과 test error는 **0.8%**로 더 내려간다고 보고한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/51b76268-4ff6-4f38-afbb-94918fd3ea27/image.png" width="80%">
</p>

이 결과가 논문 전체에서 갖는 의미는 단순히 "숫자 인식 잘 됨"이 아니라, 다음 두 가지다.

1. **아키텍처 기반 prior(CNN)** + **충분한 데이터** + **적절한 최적화**가 결합되면, 수작업 특징 없이도 높은 정확도를 얻을 수 있다.  
2. 데이터 증강(distortion)은 단순 트릭이 아니라, (논문 초반의 일반화 관점에서) 사실상 "유효 데이터 수 $P$ 증가" 혹은 "불변성 prior 주입"과 비슷한 효과를 주며, 그 결과가 수치로 확인된다.

### 🔹 다른 분류기들과의 비교

논문은 다양한 분류기(선형 분류기, k-NN, PCA+quadratic, SVM 계열, MLP 등)를 동일 데이터에서 비교한다. 이 글에서는 **논문 본문에서 수치로 직접 언급된 비교점**을 중심으로 정리한다.

- 단순 선형 분류기는 정규 데이터에서 test error가 **12%** 수준이라고 언급한다(픽셀의 가중합 기반).  
- pairwise linear(클래스 쌍별 분리 유닛을 두는 방식)은 **7.6%**까지 개선된다고 보고한다.  

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/ccdc478b-ce22-40db-8772-a3277b9062ae/image.png" width="70%">
</p>

비교의 결론은 논문이 이미 초반에 주장한 바와 맞물린다. "고정 특징 + 단순 분류기"가 아니라, **아키텍처가 불변성과 구조를 내장하고 학습이 이를 데이터로 구체화**하는 방식이 특히 2D 형태(필기)에서 강력하다는 것이다.

---

## 💡 해당 논문의 시사점과 한계

이 논문이 (LeNet-5 단독으로 유명해진 측면과 별개로) 지금 읽어도 인상적인 지점은 "학습"을 모델 내부뿐 아니라 **시스템 설계 전반**으로 확장하는 관점이다. 개인적으로 시사점을 세 가지로 정리하면 다음과 같다.

1. **아키텍처는 prior knowledge를 담는 그릇이다.**  
   LeNet-5는 단순히 합성곱을 쓴 것이 아니라, local receptive fields/weight sharing/subsampling이라는 prior를 통해 "필기 이미지가 갖는 변형"을 다루도록 설계했다. 이게 없으면 데이터가 많아도 학습이 비효율적일 수 있다.

2. **손실 함수는 ‘무엇을 잘하고 싶은지’를 정확히 반영해야 한다.**  
   RBF+MSE만으로는 경쟁이 약하거나 collapse 문제가 생길 수 있고, 그래서 MAP 형태나 discriminative forward 같은 손실이 나온다.

한계와 주의점도 있다.
  
- 논문이 선호하는 "로컬 정규화를 미루는 방식"은 거절/garbage 처리에는 장점이 있지만, 반대로 모듈 내부 값의 확률적 해석이 어려워질 수 있고, 다른 시스템과 결합할 때 교정(calibration)이 필요할 수 있다.  
- LeNet-5의 출력층(RBF + 분산 코드)은 당시 문맥(ASCII/후단 언어모델 결합)을 반영한 설계라, 오늘날의 표준 softmax 분류기와는 다르다. 그러나 이 차이는 "아이디어가 틀렸다"기보다 "문제/시스템 목표가 달랐다"는 맥락에서 이해해야 한다.

---

## 👨🏻‍💻 LeNet 구현하기

[`lucid`](https://github.com/ChanLumerico/lucid/tree/main)를 이용해 직접 LeNet 계열 모델들을 구현해보고, 논문과 대응시켜 단계적으로 해설하겠다.

### 0️⃣ Lucid에서 LeNet이 구현된 위치와 범위
Lucid의 LeNet 구현은 [`lenet.py`](https://github.com/ChanLumerico/lucid/blob/main/lucid/models/imgclf/lenet.py)에 있다. 이 파일은 논문 전체(특히 GTN, 전역 학습, 그래프 변환기)를 구현한 것이 아니라, 논문의 Section II–III에서 다룬 "CNN 기반 필기 분류기(LeNet 계열)"의 **아키텍처 뼈대**를 제공한다.

즉, 이 Lucid 구현은 다음을 목표로 한다고 볼 수 있다.

- 논문의 LeNet-5처럼 **Conv → 비선형(Tanh) → AvgPool**을 반복하는 특성을 반영하고,
- 이후 **분류기(FC layers)** 를 붙여 숫자/이미지 분류 문제에 쓸 수 있게 한다.

반면 논문에서 강조한 출력층의 Euclidean RBF, 분산 코드북, MAP 손실 등은 이 파일에 직접 구현되어 있지는 않다(아래에서 논문과의 대응/차이를 명확히 짚겠다).

### 1️⃣ 코드 구조: `LeNet` 클래스와 `lenet` 팩토리

Lucid의 `LeNet`은 하나의 클래스가 "Conv 블록 2개 + 동적 FC 스택"을 구성하도록 만들어져 있고, `lenet_1`, `lenet_4`, `lenet_5`는 채널 수/FC 폭을 달리한 변형을 반환한다.

```python
# lucid/models/imgclf/lenet.py
class LeNet(nn.Module):
    def __init__(self, conv_layers, clf_layers, clf_in_features, _base_activation=nn.Tanh):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv_layers[0]["out_channels"], kernel_size=5),
            _base_activation(),
            nn.AvgPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                conv_layers[0]["out_channels"],
                conv_layers[1]["out_channels"],
                kernel_size=5,
            ),
            _base_activation(),
            nn.AvgPool2d(2, 2),
        )

        in_features = clf_in_features
        for idx, units in enumerate(clf_layers, start=1):
            self.add_module(f"fc{idx}", nn.Linear(in_features, units))
            self.add_module(f"tanh{idx + 2}", _base_activation())
            in_features = units

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)

        idx = 1
        while hasattr(self, f"fc{idx}"):
            x = getattr(self, f"fc{idx}")(x)
            x = getattr(self, f"tanh{idx + 2}")(x)
            idx += 1

        return x
```

논문에서도 Fig. 9에서 LeNet-1, LeNet-4, LeNet-5 같은 변형들을 비교하고 있기 때문에, Lucid의 구성 방식(하나의 공통 구현 + 변형 함수들)은 논문 전개와 자연스럽게 맞닿는다.

### 2️⃣ 첫 번째 Conv 블록
Lucid의 첫 번째 블록은 다음과 같다.

```python
self.conv1 = nn.Sequential(
    nn.Conv2d(1, conv_layers[0]["out_channels"], kernel_size=5),
    _base_activation(),
    nn.AvgPool2d(2, 2),
)
```

논문과의 대응을 해석하면:

- `Conv2d(1, out_channels, kernel_size=5)` 형태는 **C1의 $5\times 5$ 합성곱**과 대응된다.  
- `_base_activation()`은 논문이 사용한 비선형 함수(스케일된 tanh)의 "형태적 대응"이다. Lucid는 기본값으로 `nn.Tanh`를 쓴다.  
- `AvgPool2d(2,2)`는 논문의 **S2(2×2 서브샘플링)**에 대응한다.

중요한 구현적 포인트는 **입력 채널이 1**이라는 점이다. 이는 MNIST 같은 흑백 입력(채널 1)을 전제로 한다.

또 하나의 포인트는 **패딩이 없다는 점**이다. 논문에서도 C1에서 $32→28$로 공간이 줄어드는 것을 전제로 설명하며, Lucid도 동일하게 패딩 없는 $5\times 5$ 합성곱을 사용한다.

### 3️⃣ 두 번째 Conv 블록
두 번째 블록은 다음과 같다.

```python
self.conv2 = nn.Sequential(
    nn.Conv2d(conv_layers[0]["out_channels"], conv_layers[1]["out_channels"], kernel_size=5),
    _base_activation(),
    nn.AvgPool2d(2, 2),
)
```

논문에서 C3는 "부분 연결"이 특징이지만, Lucid 구현은 표준 `Conv2d`로 **완전 연결된 채널 합성곱**을 사용한다. 즉 C3의 "partial connectivity로 대칭성 파괴"라는 세부 아이디어는 Lucid 구현에 직접 반영되어 있지 않다.

그럼에도 구조적 대응은 명확하다.

- $5\times 5$ 합성곱 + 비선형 + $2\times 2$ 평균 풀링이라는 반복 패턴은 C3/S4 흐름과 동일하고,
- 공간 크기 변화도 논문과 맞는다(예: $32→28→14→10→5$).

### 4️⃣ Flatten과 FC 스택

Lucid의 `forward`는 합성곱 블록을 거친 뒤 flatten하고, `fc1`, `fc2`, $\ldots$로 등록된 선형층을 순차적으로 적용한다.

먼저 생성자에서 분류기(FC) 파트를 구성하는 코드가 핵심이다. `clf_layers`에 지정된 유닛 수 리스트를 읽어, `fc1`, `fc2`, `fc3`…처럼 FC 레이어들을 동적으로 추가하고, 각 FC 뒤에 적용할 `tanh3`, `tanh4`, `tanh5`… 활성화 모듈도 함께 등록한다.

```python
in_features = clf_in_features
for idx, units in enumerate(clf_layers, start=1):
    self.add_module(f"fc{idx}", nn.Linear(in_features, units))
    self.add_module(f"tanh{idx + 2}", _base_activation())
    in_features = units
```

여기서 포인트는 두 가지다.

1. `clf_layers`가 곧 FC 스택의 설계도다. 예를 들어 `lenet_5`는 `[120, 84, 10]`이므로 `fc1: 400→120`, `fc2: 120→84`, `fc3: 84→10`이 생성된다.  
2. `tanh{idx+2}`처럼 활성화 모듈에 일련번호를 붙여 함께 등록해두고, `forward`에서는 `fc{idx}`와 `tanh{idx+2}`를 짝지어 순차 적용한다. (즉 마지막 `fc3` 뒤에도 `tanh5`가 적용되므로, Lucid의 LeNet은 최종 출력(10차원)도 `tanh`를 통과한 값이 된다.)

```python
def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.reshape(x.shape[0], -1)

    idx = 1
    while hasattr(self, f"fc{idx}"):
        x = getattr(self, f"fc{idx}")(x)
        x = getattr(self, f"tanh{idx + 2}")(x)
        idx += 1

    return x
```

여기서 `clf_in_features`가 중요한데, 이는 "conv2 출력의 flatten 차원"과 정확히 맞아야 한다.

Lucid의 `lenet_5` 설정은

```python
def lenet_5(**kwargs):
    return LeNet(
        conv_layers=[{"out_channels": 6}, {"out_channels": 16}],
        clf_layers=[120, 84, 10],
        clf_in_features=16 * 5 * 5,
        **kwargs,
    )
```

이다. 이 숫자들은 논문 LeNet-5의 차원 흐름과 거의 1:1로 대응된다.

입력을 $32\times 32$로 두면,
- Conv5(패딩 없음): $32→28$  
- AvgPool2: $28→14$  
- Conv5: $14→10$  
- AvgPool2: $10→5$  
- 채널 수는 16이므로 flatten은 $16\times 5\times 5$  

즉 `clf_in_features=16*5*5`는 논문이 설명한 "S4의 16 maps, 5×5"에서 C5(120)로 가는 흐름을 **선형층으로 등가 구현**한 것으로 볼 수 있다. 논문에서는 C5가 "5×5 합성곱으로 1×1"이 되어 사실상 전결합처럼 동작한다고 했는데, Lucid는 그 아이디어를 그대로 받아 "flatten 후 Linear(120)"로 구현한 셈이다.

그 다음 `clf_layers=[120, 84, 10]`는

- 120: 논문의 C5(120 maps)  
- 84: 논문의 F6(84 units)  
- 10: MNIST 숫자 클래스 수(논문은 숫자 인식에서는 10 클래스 실험을 수행)

이라는 숫자 대응을 그대로 가진다.

### 5️⃣ Lucid 구현과 논문 구현의 중요한 차이
여기서 "논문을 읽는 관점"에서, Lucid 구현이 논문과 다른 지점을 명확히 구분해두는 것이 중요하다.

1. **출력층이 RBF가 아니다.**  
   논문 LeNet-5는 출력층을 Euclidean RBF로 두고 penalty를 거리로 정의했다. Lucid는 마지막을 `nn.Linear(in_features, 10)`처럼 선형층으로 두고, 그 뒤에도 `Tanh`를 적용한다. 즉 "거리 기반 penalty" 구조나 "분산 코드북"은 구현되어 있지 않다.  

2. **스케일된 tanh(A$\cdot$tanh(Sa))가 아니라 표준 tanh다.**  
   Lucid는 기본 활성화로 `nn.Tanh`를 쓴다. 논문이 강조한 $A=1.7159$ 등의 스케일링은 이 코드에 나타나지 않는다.

이 차이는 "Lucid 구현이 틀렸다"는 의미라기보다, Lucid가 제공하는 LeNet이 "논문 LeNet-5의 핵심적인 구조적 아이디어(Conv/Tanh/AvgPool의 반복과 차원 설계)"를 중심으로 한 **실용적/간결한 변형**이라는 뜻에 가깝다.

즉 논문의 수식/출력 설계를 그대로 구현한 것이 아니라, **아키텍처 수준의 대응**이 중심이다.

---

## ✅ 정리

이 논문은 LeNet-5를 통해 **"픽셀에서 바로 학습하는 CNN"** 이 필기 숫자 인식에서 얼마나 강력한지를 보여주면서도, 거기서 멈추지 않고 "문서 인식 시스템 전체"로 논의를 확장한다.

전통적 방식이 고정 특징 추출기를 사람 손으로 설계하는 데 많은 노력을 들였다면, 저자들은 아키텍처(CNN)와 전역 학습(GTN)을 통해 그 노력을 "학습 가능한 구조"로 옮겨간다. 이는 오늘날의 end-to-end 학습 관점과 닿아 있으면서도, 그래프/동적 계획법을 중심에 두고 미분을 설계한다는 점에서 여전히 교과서적인 가치가 있다.

한편 Lucid의 `lenet.py` 구현은 논문 LeNet-5의 "Conv/Tanh/AvgPool 반복과 차원 흐름"을 비교적 충실히 따라가되, RBF 출력/분산 코드/스케일된 tanh 같은 세부는 생략한 간결한 변형이다. 논문을 이해한 뒤 이 코드를 다시 읽어보면, 각 숫자(6,16,120,84,10)와 공간 크기(32→28→14→10→5)가 단순한 구현 선택이 아니라, 논문이 설명한 설계 의도가 코드의 shape/레이어 구성으로 그대로 번역되어 있음을 확인하게 된다.

#### 📄 출처
LeCun, Yann, Léon Bottou, Yoshua Bengio, and Patrick Haffner. "Gradient-Based Learning Applied to Document Recognition." *Proceedings of the IEEE*, vol. 86, no. 11, 1998, pp. 2278–2324. doi:10.1109/5.726791. http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf.