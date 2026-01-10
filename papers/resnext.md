# [ResNeXt] Aggregated Residual Transformations for Deep Neural Networks
ResNeXt 논문은 ResNet 이후의 네트워크 설계가 마주한 문제를 한 문장으로 요약한다. 더 깊게, 더 넓게 키우는 방식은 성능을 올리긴 하지만, 모듈이 복잡해질수록 하이퍼파라미터 선택지가 폭발하고(필터 수, 경로 수, 커널 조합, stage별 커스터마이징 등), 모델 설계를 데이터셋/태스크에 과적합시키기 쉬워진다. Inception 계열의 split-transform-merge가 대표적으로 높은 정확도를 보여줬지만, 그 성과가 동시에 설계 복잡도의 증가를 동반했다는 점을 저자들은 문제로 삼는다.

이 논문이 제안하는 핵심은 **cardinality**라는 새로운 축이다. ResNeXt 블록은 동일한 형태의 변환 $T_i(\cdot)$를 여러 개 준비해 두고, 그 결과를 합산(sum)하는 방식으로 residual function을 만든다. 이때 변환들의 개수 $C$가 cardinality이며, 깊이(depth)와 너비(width)와는 다른 독립적 설계 축으로 취급된다. 중요한 점은, 이 다중 경로가 Inception처럼 경로마다 다른 필터를 쓰는 구조가 아니라 **모든 경로가 동일한 topology**를 갖도록 강제된다는 것이다. 그 결과 설계가 더 단순해지고, cardinality만 늘려도(복잡도/파라미터를 유지한 상태에서) 정확도가 좋아지는 현상을 ImageNet-1K/5K, CIFAR, COCO detection에서 실험으로 보여준다.

## 1️⃣ 논문 배경

### 🔹 Feature Engineering에서 Network Engineering으로
논문은 컴퓨터 비전이 _feature engineering에서 network engineering으로 이동_ 해 왔다고 말한다. 전통적 수작업 특징(SIFT, HOG 등)은 사람이 설계하고, 분류기/파이프라인을 맞추는 방식이 중심이었다. 반면 대규모 데이터와 신경망 기반 학습은 학습 과정에서 사람이 관여할 요소를 줄이고, 학습된 표현을 여러 태스크로 전이하는 방향을 강화해 왔다. 이 변화는 인간의 노동을 줄인 것이 아니라, 노동의 위치를 **네트워크 설계**로 옮겨 놓았다.

여기서 문제가 되는 것은 설계 변수의 증가다. 네트워크가 깊어질수록 조정할 항목이 많아지고, 너비(width), 커널 크기, stride, 병렬 경로 구조, stage별 구성 등 **선택지가 늘어난다**. 논문은 VGG/ResNet 계열이 성공한 이유 중 하나로, 모듈을 같은 형태로 반복해 설계 자유도를 크게 줄였다는 점을 든다. 즉 깊이(depth)가 핵심 축으로 전면에 등장한 것은, 네트워크를 무작정 깊게 만들었기 때문이 아니라, 반복 규칙이 설계 공간을 단순화해 준 덕분이라는 해석이다.

반대로 Inception 계열은 _split-transform-merge_ 를 통해 낮은 이론적 복잡도에서 높은 정확도를 보여줬지만, 그 과정에서 경로별 필터 수와 커널 조합이 stage마다 커스터마이즈되는 경향이 강했다. 결과적으로 그 설계를 다른 데이터셋/태스크로 옮길 때 무엇을 어떻게 조정해야 하는지가 불명확해지고, 설계가 복잡해진다는 문제가 생긴다. 저자들은 여기에서, **정확도와 복잡도 간 트레이드오프**뿐 아니라 **정확도와 설계 복잡도 간 트레이드오프**를 본격적으로 다루고자 한다.

#### 논문의 목표
논문이 설정하는 목표는 다음처럼 요약할 수 있다.

1. Inception처럼 multi-branch 구조를 쓰되  
2. VGG/ResNet처럼 반복 가능한 단순 규칙으로 만들고  
3. 그 과정에서 조정해야 할 하이퍼파라미터 수를 최소화한다.

이 목표를 만족하려면, 경로마다 서로 **다른 설계를 허용하면 안 된다**. ResNeXt는 이 지점에서 한 가지 강한 제약을 둔다. **모든 branch(변환) $T_i$가 같은 topology**를 갖게 하고, output을 sum으로 합친다. 이 단순한 제약 덕분에, 경로 수를 늘리는 것(cardinality 증가)을 독립적 변수로 분리해 실험적으로 평가할 수 있게 된다.

### 🔸 Cardinality라는 새 축
논문에서 **cardinality** $C$는 aggregated transformation의 개수(경로 수)다. 겉으로 보면 단순히 branch를 늘린 multi-branch 모델처럼 보이지만, ResNeXt에서 중요한 점은 **그 경로들이 서로 다른 일을 하도록 설계하지 않는다**는 것이다. 각 경로는 같은 형태의 bottleneck 변환을 수행하고, 그 결과를 더한다.

이렇게 보면 cardinality는 Inception의 가지 수(branch 수)와 비슷해 보이기도 한다. 하지만 Inception에서는 경로마다 커널 크기와 필터 수가 달라지고, merge가 concat으로 이뤄지며, stage마다 모듈이 변형되는 경우가 많다. 반면 ResNeXt는 경로마다 topology를 동일하게 유지해 설계를 단순화하고, 경로 수만을 독립적으로 올릴 수 있게 만든다. 그 결과 depth/width를 늘리지 않고도 표현력을 키우는 방식으로 cardinality가 작동한다는 것이 논문 전체의 주장이다.

#### Fig. 1의 핵심 메시지
Fig. 1은 ResNet bottleneck block과 ResNeXt block(예: $C=32$)을 나란히 보여주면서, FLOPs/파라미터 수가 비슷한 조건에서도 ResNeXt의 구조가 더 강한 표현을 만들 수 있음을 암시한다. ResNeXt 블록은 여러 경로의 bottleneck을 병렬로 두고 더하는 형태인데, 그 구조를 동일 복잡도 조건에서 유지하도록 bottleneck width를 조정하는 것이 이후 논문 내용의 핵심으로 연결된다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/48ef07fc-3622-43be-9058-e9d6eade7f93/image.png" width="40%">
</p>

## 2️⃣ 관련 연구

### 🔹 Multi-Branch 네트워크와 잔차 연결의 계보
논문은 multi-branch 구조를 가진 대표적 계열로 _Inception_ 을 언급한다. Inception은 입력을 여러 경로로 분기한 뒤, 각 경로에서 다른 변환을 수행하고, concat으로 합친다. 이 구조는 큰 커널을 직접 쓰는 dense layer의 표현력을 부분적으로 근사하면서도 계산량을 줄이려는 목적과 결합되어 있다. 하지만 경로별 설계와 stage별 커스터마이징이 필요하다는 점에서, **설계 공간이 커지고 하이퍼파라미터가 증가하는 단점**이 있다.

_ResNet_ 은 multi-branch 관점에서 보면 2-branch 네트워크로 해석될 수 있다. 한 branch는 identity shortcut이고, 다른 branch가 residual function $F(x)$를 계산한다. 여기서 중요한 점은, ResNet의 성공이 어떤 복잡한 branch 조합 규칙 때문이 아니라, **항상 열려 있는 identity 경로**가 최적화를 안정화한다는 구조적 메시지였다는 것이다. **ResNeXt**는 그 위에서, residual branch 내부를 더 풍부하게 만드는 방향으로 multi-branch를 도입하되, 설계를 복잡하게 만들지 않는 방식을 찾는다.

### 🔸 Grouped Convolution과 앙상블(Ensemble)
ResNeXt는 구현 관점에서 _grouped convolution_ 과 강하게 연결된다. 논문은 Krizhevsky 등이 소개한 grouped convolution을 Fig. 3(c)의 동치 변형으로 끌어온다. 여기서 중요한 점은, grouped convolution이 원래는 엔지니어링 상의 절충(계산/메모리 제약)으로 쓰였던 맥락이 있다는 것이다. ResNeXt는 이를 단지 가속을 위한 트릭으로 쓰는 것이 아니라, **동일 topology의 여러 변환을 병렬로 수행**하는 구조를 명료하게 구현하는 도구로 사용한다.

또한 논문은 압축(compression) 연구들과의 차이를 분명히 한다. grouped convolution이나 channel-wise(depthwise) convolution은 종종 redundancy 제거, 모델 경량화의 수단으로 등장해 왔다. 반면 ResNeXt는 압축을 목표로 하는 것이 아니라, 동일한 복잡도/파라미터 조건에서 **표현력이 더 강한 아키텍처**를 만들 수 있음을 보여주려 한다. 즉, 같은 계산 예산에서 더 나은 정확도를 얻는 것이 핵심이다.

마지막으로 ResNeXt는 외형상 ensemble과 비슷해 보일 수 있다. 여러 경로의 출력을 더하는 구조는 여러 모델을 평균내는 것과 유사해 보이기 때문이다. 하지만 논문은 이를 ensemble로 보는 것은 **부정확**하다고 말한다. Ensemble은 독립적으로 학습된 모델들을 평균내는 것이고, ResNeXt의 여러 변환 $T_i$는 **하나의 네트워크 안에서 joint하게 학습**된다. 따라서 ResNeXt의 효과를 단순히 ensemble 효과로 환원하기보다는, residual function 내부의 표현력 증가로 해석해야 한다.

## 3️⃣ 방법론

### 🔹 반복 규칙으로 설계 공간을 줄이기
논문은 VGG/ResNet의 장점(모듈 반복)을 유지하기 위해, 네트워크 전체를 residual block의 stack으로 구성하고, block이 따라야 할 두 가지 규칙을 제시한다.

1. **같은 spatial map size**를 만드는 블록들은 같은 하이퍼파라미터(필터 크기, width 등)를 공유한다.
2. spatial map이 2배 downsample될 때마다 block의 width는 2배가 된다. 

두 번째 규칙의 목적은 층당 FLOPs가 _대략 일정하게 유지_ 되도록 하는 것이다. 해상도가 절반으로 줄면 픽셀 수가 $1/4$가 되므로, 채널 수를 2배로 늘려도 전체 계산량 증가가 완만해지고, stage별로 계산 예산이 균형을 이룬다. 이 규칙은 ResNet-50/101 같은 표준 stage 설계($56→28→14→7$)와도 자연스럽게 맞물린다.

#### ResNet-50과 ResNeXt-50(32×4d)의 템플릿 비교
논문 Table 1은 동일한 stage 반복 수(`[3,4,6,3]`)를 가지는 ResNet-50과 ResNeXt-50(32×4d)을 비교한다. 핵심은, ResNeXt-50이 ResNet-50과 비슷한 파라미터 수/연산량을 유지하면서도, 3×3 convolution이 **grouped convolution**으로 바뀐다는 점이다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/6f5df7cd-55c6-4496-a027-72f677fc50ce/image.png" width="40%">
</p>

논문은 ResNet-50이 대략 25.5M parameters, 4.1B FLOPs이고, ResNeXt-50(32×4d)이 25.0M parameters, 4.2B FLOPs로 유사한 수준이라고 적는다. 즉, Table 1은 ResNeXt가 단지 모델을 키운 것이 아니라, 비슷한 계산 예산 안에서 구조를 바꿨다는 점을 강조한다.

#### ResNeXt-50(32×4d) 표기의 의미
논문은 템플릿으로 아키텍처를 지칭하는 표기법을 쓴다. ResNeXt-50(32×4d)에서

- `32`는 cardinality $C$  
- `4d`는 bottleneck width $d$가 4임을 의미

로 이해할 수 있다. 중요한 것은, `32×4d`가 단순히 채널 수 $128$을 의미하는 것이 아니라, grouped convolution 관점에서 **32개의 group이 있고 각 group의 width가 4**라는 구조적 의미를 가진다는 점이다.

### 🔸 내적을 Split-Transform-Aggregate로 재해석하기
논문은 aggregated transformation의 직관을 만들기 위해, 가장 단순한 연산인 inner product(가중합)를 재해석한다. 입력 벡터 $x=[x_1,\dots,x_D]$에 대한 inner product는

$$
\sum_{i=1}^{D} w_i x_i \tag{1}
$$

로 쓸 수 있다. 이 연산은 완전히 선형이지만, 논문은 이를 다음 세 단계로 쪼갠다.

1. **Splitting**: 입력 $x$를 저차원 embedding으로 분해한다(이 경우 각 $x_i$는 1차원 subspace).  
2. **Transforming**: 각 subspace를 변환한다(이 경우 $w_i x_i$라는 단순 스케일링).  
3. **Aggregating**: 변환 결과를 합산한다($\sum_i$).

이 관점에서 Eq.(1)의 $D$는 매우 많은 단순 변환들의 집합을 합산하는 형태다. 논문은 여기에서 한 발 더 나아가, 단순 변환 $w_i x_i$ 대신 **더 복잡한 변환 함수**를 놓고, 그것을 여러 개 합산하는 방향으로 일반화한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/cdaaf166-e00e-4515-be94-e87cfe4fd6e5/image.png" width="40%">
</p>

### 🔹 Cardinality의 수학적 정의
논문은 Eq.(1)의 관점을 바탕으로, 변환 함수 $T_i(\cdot)$를 도입해 aggregated transformation을 다음처럼 정의한다.

$$
F(x)=\sum_{i=1}^{C} T_i(x) \tag{2}
$$

여기서 $C$가 **cardinality**다. 논문은 $C$가 Eq.(1)의 $D$와 비슷한 위치에 있지만, 반드시 같을 필요가 없고 임의의 값이 될 수 있다고 강조한다. width가 단순 변환(채널 수)의 개념과 연결된다면, cardinality는 복잡한 변환 $T_i$의 개수라는 점에서 다른 축이다.

ResNeXt는 ResNet의 residual 연결 위에서 Eq.(2)를 residual function으로 사용한다. 즉, residual block의 출력은

$$
y = x + \sum_{i=1}^{C} T_i(x) \tag{3}
$$

로 정의된다. 여기서 핵심은, 여러 변환의 합이 **residual branch**를 구성한다는 점이다. 즉 ResNeXt는 ResNet의 $F(x)$를 더 풍부한 형태로 확장한 셈이며, 그 확장 방식이 cardinality라는 하나의 스칼라로 제어된다.

#### 모든 $T_i$의 topology를 같게 설계
Eq.(2)는 $T_i$를 임의 함수로 둘 수 있지만, 논문은 실험에서 모든 $T_i$가 같은 topology를 가지게 한다. 이 제약은 두 가지 이유에서 중요하다.

1. VGG/ResNet 스타일의 반복 규칙과 잘 맞는다(모듈이 _homogeneous_).  
2. cardinality 증가가 다른 변수(경로별 커스터마이징)와 얽히지 않도록 분리해 준다.

논문은 각 $T_i$를 bottleneck 형태로 두고(예: 1×1으로 차원 축소 → 3×3 → 1×1으로 복원), 첫 1×1이 _low-dimensional embedding_ 을 만든다고 설명한다. 이때 $C$가 늘면 병렬 bottleneck 경로가 늘어나며, 합산으로 aggregation이 수행된다.

#### ResNeXt 블록의 여러 동치들
Fig. 3은 ResNeXt 블록의 동치 형태를 보여준다.

- (a) **Aggregated Residual Transformations**: 여러 경로의 출력을 sum으로 합침(직접 구현).  
- (b) **Early Concatenation**: 중간 표현을 concat한 뒤, 마지막 1×1으로 섞어 sum과 동치가 되게 만듦.  
- (c) **Grouped Convolution**: (b)의 concat 구조를 grouped conv 표기로 더 succinct하게 구현.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/56a8c0c3-903b-4e14-aeae-3da133c01a38/image.png" width="70%">
</p>

논문은 이 동치가 단순한 대수학으로 확인 가능하다고 설명한다. 예를 들어 두 경로($C=2$)만 두면, sum 형태 $A_1B_1 + A_2B_2$가 concat 형태의 행렬 곱 $[A_1, A_2][B_1,B_2]$로 재표현된다는 식의 직관이다. 핵심은, sum으로 합친다는 것은 결국 마지막 선형 결합으로 흡수될 수 있고, 이때 concat된 표현을 처리하는 방식이 grouped conv의 형태로 이어진다는 점이다.

#### 너무 얕은 Depth의 비효율성
논문은 중요한 주의점을 하나 둔다. Fig. 3(b)(c) 같은 재표현은 block depth가 충분히 클 때(대략 `depth>=3`) **nontrivial**해진다. ResNet의 basic block처럼 `depth=2`인 블록에서는 이런 재표현이 결국 단순히 wide하고 dense한 모듈로 수렴해 버린다. 즉, cardinality가 의미 있는 새로운 축이 되려면, $T_i$가 단순한 1-layer 변환이 아니라, **내부에 구조가 있는 변환**이어야 한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/db5b47f5-9b91-4fa5-ae02-2069b553e7e4/image.png" width="40%">
</p>


### 🔸 복잡도 보존을 위한 Width 조정
논문은 ablation에서 cardinality를 바꿀 때, 다른 하이퍼파라미터를 최대한 유지하면서 복잡도를 보존하고자 한다. 이때 핵심 전략은 bottleneck 내부 폭 $d$를 조정하는 것이다. $d$는 입력/출력 채널과 분리된 내부 폭이므로, $C$와의 trade-off를 만들기 좋다.

논문은 ResNet bottleneck(예: `conv2` stage)을 기준으로 파라미터 수를 근사 계산한다. ResNet bottleneck은

- `1×1`: $256 → 64$  
- `3×3`: $64 → 64$  
- `1×1`: $64 → 256$

형태이며, 파라미터 수는 대략

$$
256\cdot 64 + 3\cdot 3\cdot 64\cdot 64 + 64\cdot 256 \approx 70\text{k}
$$

로 설명된다. 반면 ResNeXt 템플릿에서 bottleneck width가 $d$이고 cardinality가 $C$이면, 각 경로가 비슷한 구조를 가지므로 전체 파라미터는

$$
C\cdot(256\cdot d + 3\cdot 3\cdot d\cdot d + d\cdot 256) \tag{4}
$$

로 근사할 수 있다. 이 식에서 관찰할 점은 다음과 같다.

- $C$는 선형으로 곱해진다(경로 수가 늘면 병렬 변환이 늘어남).  
- $d$는 3×3 항에서 $d^2$로 들어가므로, width를 늘리는 효과가 빠르게 커진다.  

따라서 복잡도를 유지하려면, $C$를 늘릴수록 $d$를 줄이는 trade-off가 필요하다. 논문은 $C=32, d=4$이면 Eq.(4)가 약 70k로 맞춰진다고 설명하며, 이를 Table 2로 정리한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/1e844ee9-ebd2-42d6-83fb-f34bf2270979/image.png" width="40%">
</p>

여기서 group conv width는 전체 3×3 층의 채널 폭으로, 대략 $C\cdot d$에 대응한다. 예컨대 $C=32, d=4$이면 128이 된다. Fig. 3(c)에서 grouped convolution의 입력/출력이 128채널이고 group=32이면, 각 group이 4채널을 처리하는 구조가 된다. 이 구조가 바로 ResNeXt 표기 32×4d의 의미를 구현 관점으로 고정해 준다.

## 4️⃣ 구현 세부사항

### 🔹 ImageNet 학습/평가 프로토콜
논문은 구현을 ResNet의 공개 구현을 따르는 것으로 두고, ablation에서는 평가 프로토콜을 엄격히 통제한다. 핵심은 구조의 차이를 비교할 때, data augmentation이나 테스트 크롭 방식이 혼입되어 결론이 흔들리지 않게 하는 것이다.

텍스트 추출본이 명시하는 **ImageNet 구현 요점**은 다음과 같다.

- **입력**: 224×224 crop을 랜덤으로 추출(리사이즈 + scale/aspect ratio augmentation)  
- **Shortcut**: 차원 증가 블록만 projection(type B), 그 외는 identity  
- **Downsampling**: conv3/4/5의 첫 블록에서 stride-2를 적용하되, *첫 블록의 3×3 층에서 stride-2*를 준다  
- **최적화**: SGD, 8 GPUs에서 batch size 256(=GPU당 32), weight decay 0.0001, momentum 0.9  
- **Learning Rate**: 0.1에서 시작해 스케줄에 따라 10배씩 3회 감소  
- **초기화**: 특정 초기화 방식(논문이 인용)을 사용  
- **Ablation 평가**: 짧은 변 256으로 리사이즈 후 224×224 center crop 단일 평가

이 설정은 ResNet-50/101과 ResNeXt-50/101의 비교에서 동일하게 적용된다. 특히 ablation에서는 multi-crop이나 multi-scale이 아니라 단일 center crop으로 평가한다는 점이 중요하다. Table 3/4 같은 비교표의 수치가 모델 구조에 의한 차이를 더 직접적으로 반영하기 때문이다.

### 🔸 Grouped Convolution을 선택한 이유
논문은 Fig. 3의 세 형태가 BN/ReLU 배치를 적절히 처리하면 엄밀히 동치이며, 실제로 세 형태를 모두 학습해 같은 결과를 얻었다고 말한다. 그럼에도 구현은 Fig. 3(c)를 택한다. 이유는 간단하다.

- **(a)**: 직접 합산은 경로 수가 늘수록 구현이 복잡해질 수 있다.  
- **(b)**: concat 구현은 텐서 조작과 마지막 결합을 관리해야 한다.  
- **(c)**: grouped conv는 프레임워크의 group conv primitive로 표현되므로 코드가 짧고 빠르다.

## 5️⃣ 모델 실험

### 🔹 ImageNet-1K
논문은 먼저 Table 2의 trade-off 조건에서 cardinality와 bottleneck width를 바꿨을 때의 성능을 비교한다. 이때 핵심 조건은 FLOPs를 거의 유지하는 것이다(ResNet-50/101 대비 약 4.1B/7.8B FLOPs 수준). 결과는 Table 3으로 요약된다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/be678945-0008-459f-b3de-afcf117293cb/image.png" width="70%">
  <img src="https://velog.velcdn.com/images/lumerico284/post/d4635b8a-b3fd-471f-a3e3-43dba69ed747/image.png" width="40%">
</p>

이 결과가 논문 메시지를 구성하는 방식은 다음과 같다.

1. 복잡도를 유지한 상태에서도, $C$를 늘릴수록 error가 지속적으로 감소한다.  
2. 특히 32×4d는 ResNet baseline보다 더 낮은 validation error를 준다(50-layer에서 23.9 → 22.2).  
3. Fig. 5는 training error 측면에서도 ResNeXt가 더 낮게 내려감을 보여주며, 논문은 이를 단순 정규화 효과가 아니라 **더 강한 표현**의 근거로 해석한다.

#### 50-Layer에서 개선 폭이 더 크게 보이는 이유
Table 3만 보면, ResNeXt-50의 개선 폭(1.7%p)이 ResNeXt-101의 개선 폭(0.8%p)보다 더 크다. 논문은 이 차이를 단순히 깊이에 따른 diminishing returns로만 말하지 않고, training error 곡선을 함께 보고 해석한다. ResNeXt-101은 training error 개선 폭이 **매우 큰데**(예: ResNet-101 20% vs ResNeXt-101 16% 수준), validation의 개선은 데이터 규모와도 연결될 수 있다고 본다. 즉, 더 많은 데이터에서 _generalization gap_ 이 더 크게 벌어질 수 있고, 이를 ImageNet-5K에서 확인하겠다는 흐름으로 이어진다.

#### 너무 작은 Bottleneck Width로 인한 Saturation
Cardinality를 늘리기 위해 $d$를 계속 줄이면, 어느 순간 폭이 너무 작아져 **표현이 병목을 겪을 수 있다**. 논문은 Table 3의 추세를 근거로, bottleneck width가 너무 작아지는 경우에는 정확도 개선이 _saturate_ 할 수 있다고 말하며, 이후 비교에서는 $d\ge 4$ 수준을 채택한다. 이는 cardinality가 만능 축이 아니라, **width와의 trade-off**가 존재한다는 현실적인 메시지다.

### 🔸 Depth/Width vs. Cardinality
다음으로 논문은 복잡도를 2배로 늘렸을 때, 성능 향상이 어느 축에서 더 효율적으로 나오는지 비교한다. Table 4는 ResNet-101 baseline의 2× FLOPs(∼15B FLOPs) 수준에서 다음을 비교한다.

- _going deeper_: ResNet-200  
- _going wider_: wider ResNet-101(1×100d)  
- _increasing cardinality_: 2×64d ResNeXt-101, 64×4d ResNeXt-101

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/b3bc58d4-4e5e-4e57-be99-6aaa0ad48428/image.png" width="40%">
</p>

논문이 강조하는 결론은 명확하다.

1. 복잡도를 늘리면 대부분 error가 줄어들긴 한다.  
2. 하지만 depth를 늘리는 ResNet-200(0.%3p 개선)이나 width를 늘리는 wider ResNet-101(0.7%p 개선)보다, cardinality를 늘리는 방식(예: 2×64d, 64×4d)이 더 큰 개선(1.3%p, 0.8%p)을 준다.  
3. 특히 32×4d ResNeXt-101(21.2)은 ResNet-200보다도 더 좋으면서, 복잡도는 대략 절반 수준이라고 논문은 강조한다.

이 비교가 의미 있는 이유는, 단지 ResNeXt가 좋다는 사실을 넘어서, **설계 축의 효율성**을 보여주기 때문이다. 같은 계산 예산을 어디에 투자할지(깊이, 너비, cardinality)에 대한 실험적 근거가 Table 4로 제공된다.

#### Residual Connection의 역할
논문은 ResNeXt가 **잔차 연결을 전제로 한 구조**임을 다시 확인하기 위해, shortcut을 제거한 비교를 포함한다. 텍스트 추출본의 표를 그대로 쓰면 다음과 같다.

| setting | w/ residual | w/o residual |
|---|---:|---:|
| ResNet-50 (1×64d) | 23.9 | 31.2 |
| ResNeXt-50 (32×4d) | 22.2 | 26.1 |

shortcut을 없애면 ResNeXt도 성능이 크게 나빠진다($22.2 → 26.1$). 하지만 ResNeXt는 shortcut이 있든 없든 대응되는 ResNet보다 _일관되게 더 좋다_. 논문은 이를 다음처럼 해석한다.

- Residual connection은 최적화를 돕는다(없으면 둘 다 크게 악화).  
- Aggregated transformation은 residual 유무와 무관하게 더 강한 표현을 제공한다(항상 더 좋음).  

즉, ResNeXt는 ResNet의 **최적화 장점을 그대로 가져가면서**, residual branch의 표현력을 cardinality로 키우는 방식이라고 이해하는 것이 자연스럽다.

#### Grouped Convolutoin 구현의 현실
논문은 grouped convolution이 이론적으로 FLOPs가 비슷해도, 구현에 따라 학습 속도 오버헤드가 생길 수 있음을 보고한다. Torch의 기본 group conv 구현을 썼을 때, 8×M40에서

- 32×4d ResNeXt-101: `0.95s/mini-batch`
- ResNet-101 baseline: `0.70s/mini-batch`

로 차이가 있었다고 적는다. 논문은 이를 reasonable overhead로 보고, CUDA 등 하위 최적화가 들어가면 줄어들 것이라고 기대한다. 이 지점은 ResNeXt의 아이디어가 모델 설계 차원에서는 단순하지만, 하드웨어/커널 최적화까지 고려하면 실무적인 고려가 남는다는 사실을 보여준다.

### 🔹 SOTA 비교: 단일 Crop과 ILSVRC 2016
논문은 Table 5에서 ImageNet-1K validation에서 single-crop 결과를 여러 SOTA와 비교한다. 여기서 ResNeXt-101(64×4d)은 $320×320$ crop(ResNet 계열의 테스트 크기)에서 top-5 error $4.4$를 보고한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/dcd7d214-25ff-4529-b4a8-7991a7bbda68/image.png" width="40%">
</p>

이 표는 ResNeXt의 메시지를 두 층위에서 강화한다.

1. 단일 crop에서도 경쟁력이 높다(특히 top-5에서 $4.4$).  
2. Inception 계열 대비 구조가 단순하며, 경로별 커스터마이징을 거의 요구하지 않는다.

논문은 또한 _multi-scale dense testing_ 을 쓰면 단일 모델 top-1/top-5가 $17.7$/$3.7$ 수준으로 내려가고, ensemble top-5가 $3.03$이라고 언급한다. 이 수치는 경쟁에서 multi-crop/multi-scale이 saturation을 만드는 현실도 함께 보여준다. 즉, 모델 구조 자체의 비교는 single-crop/통제된 프로토콜에서 더 직접적으로 읽는 것이 좋고, 경쟁용 제출은 별도의 시스템적 요소가 섞인다는 점을 염두에 둬야 한다.

### 🔸 ImageNet-5K
논문은 ImageNet-1K에서 성능이 saturate해 보일 수 있지만, 그것이 모델의 한계가 아니라 **데이터의 복잡도 때문**일 수 있다고 주장한다. 그래서 ImageNet-22K의 subset으로 5000-class 데이터셋을 구성해 실험한다.

여기서 중요한 실험 설계는 평가 방식이다. 5K 데이터는 공식 valid split이 없으므로, 원래 ImageNet-1K validation set에서 평가한다. 그리고 평가를 두 방식으로 나눈다.

1. **5K-Way 분류**: 5000 classes 전체에서 softmax, 4K classes로 예측되면 자동 오류  
2. **1K-Way 분류**: softmax를 1K classes에만 적용(같은 val set에서 1K task로 평가)

또한 학습은 5K 데이터로 scratch부터 하되, 1K 학습과 같은 mini-batch 수로 학습한다(즉 epoch 수는 $1/5$ 수준). 이 조건은 학습 시간 증가 없이 더 많은 데이터를 경험하게 하는 방식이다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/6fd7e4e1-539c-4b6a-81b0-6f51118d427e/image.png" width="70%">
</p>

논문은 여기서 ResNeXt-50이 5K top-1을 3.2%p 개선하고($45.5 → 42.3$), ResNeXt-101이 2.3%p 개선한다고($42.4 → 40.1$) 해석한다. 즉, 더 많은 데이터에서는 **validation gap이 더 커질 수 있다**는 앞선 추측이 어느 정도 확인된다.

또 하나의 흥미로운 관찰은, 5K로 학습한 ResNeXt-101의 1K-way error($22.2$/$5.7$)가 1K로 학습한 ResNeXt-101(32×4d)의 1K-way error($21.2$/$5.6$)와 경쟁력 있다는 점이다. 논문은 이를 학습 시간 증가 없이 더 어려운 태스크를 학습했음에도 성능이 유지되는 결과로 해석하며, **더 큰 데이터에서 ResNeXt의 표현력이 더 강해질 가능성**을 강조한다.

### 🔹 CIFAR: 작은 이미지에서도 Cardinality가 더 효율적인가
논문은 CIFAR-10/100에서도 실험을 수행한다. 여기서는 ImageNet의 50/101-layer와 달리, ResNet 논문의 CIFAR 설정을 따라 29-layer 네트워크(각 stage 3 blocks)를 사용한다. 논문은 basic block 대신 bottleneck 템플릿

$$
\begin{bmatrix} 1\times 1, 64 \ 3\times 3, 64 \ 1\times 1, 256 \end{bmatrix}
$$

을 사용한다고 명시한다. 네트워크는 첫 3×3 conv(64 filters)로 시작하고, map size $32/16/8$인 3 stages에 각 3 residual blocks, global average pooling + fc로 끝난다.

이 조건에서 논문은 **두 가지 방식**으로 모델 크기를 늘려 비교한다.

1. Cardinality를 늘리고 **width는 고정**  
2. **Cardinality=1**을 유지하고 width를 늘림

Fig. 7은 CIFAR-10에서 test error vs 파라미터 수의 곡선을 보여주며, 동일 모델 크기에서 **cardinality 증가가 더 효과적임**을 시각화한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/f3e98a7b-af84-439e-a901-6029374db750/image.png" width="40%">
</p>

Table 7은 대표 설정의 성능을 정리한다.

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/014100ee-1a20-48a7-a7a9-49ab122d9d96/image.png" width="40%">
</p>

논문은 34.4M 규모에서 Wide ResNet보다 더 좋은 결과를 보고하며, 더 큰 모델에서는 CIFAR-10 $3.58%$, CIFAR-100 $17.31%$로 **당시 SOTA 수준**임을 주장한다(유사한 data augmentation 조건에서).

## 💡 해당 논문의 시사점과 한계
ResNeXt의 가장 큰 의의는 네트워크 설계를 새로 복잡하게 만드는 대신, **단순한 반복 규칙 + 단일 축(cardinality)** 로 multi-branch의 장점을 끌어왔다는 점이다. 논문은 Inception이 보여준 split-transform-merge의 표현력 향상을 인정하면서도, 경로별 커스터마이징과 stage별 설계가 가져오는 복잡도를 문제로 삼는다. ResNeXt는 모든 경로의 topology를 동일하게 유지해 설계를 단순화하고, 경로 수 $C$를 독립 변수로 만들었다.

이 단순성이 강점이 되는 이유는 실험 논증이 깔끔해지기 때문이다. Table 2/3/4는 동일 복잡도(또는 복잡도 증가가 명시된 조건)에서 depth/width/cardinality의 효과를 비교하도록 설계되어 있고, 그 비교에서 cardinality가 일관되게 효율적인 축으로 나타난다. 특히 Table 4는 동일한 2× FLOPs 조건에서 cardinality 증가가 depth/width 증가보다 더 큰 error 감소를 준다는 것을 보여주며, ResNeXt가 단지 한 모델이 아니라 설계 원리(어떤 축이 더 효율적인가)를 제시했다고 볼 수 있다.

또한 ResNeXt는 구현 관점에서 Fig. 3(c)의 **grouped convolution**으로 정리될 수 있다는 점이 중요하다. 즉, multi-branch 구조를 프레임워크 primitive로 명료하게 표현할 수 있어 확장성이 좋고, 설계가 단순하다는 장점이 실제 코드 구조에서도 유지된다.

#### 한계와 현실적 고려
한편 ResNeXt에도 현실적인 제약이 있다.

- Group conv의 실제 속도는 **구현/하드웨어에 영향을 받는다**. 논문도 기본 구현에서는 학습 시간 오버헤드가 있다고 보고한다. 즉 FLOPs가 비슷하다고 해서 항상 같은 속도를 보장하지는 않는다.
- Cardinality와 width 사이에는 **trade-off**가 있으며, width를 너무 줄이면 saturation이 나타날 수 있다(Table 3 해석).
- 논문은 topology를 homogeneous하게 고정해 설계를 단순화했지만, 이 제약이 최적의 표현을 **항상 보장하는 것은 아니다**. 즉, 설계 단순성과 최고 정확도 간의 트레이드오프는 여전히 남는다.

그럼에도 불구하고 ResNeXt의 영향은 컸다. 이후 많은 backbone 설계에서 group conv, split-transform-merge, 그리고 cardinality에 해당하는 축이 **반복적으로 등장**했고, ResNeXt는 그 축을 비교적 단순한 형태로 정식화해 실험으로 설득한 대표 사례로 남았다.

---

## 👨🏻‍💻 ResNeXt 구현하기
이 파트에서는 [`lucid`](https://github.com/ChanLumerico/lucid/tree/main) 라이브러리의 [`resnext.py`](https://github.com/ChanLumerico/lucid/blob/main/lucid/models/imgclf/resnext.py)를 중심으로 ResNeXt가 Lucid에서 어떻게 구현되는지 논문 관점으로 읽는다. ResNeXt 논문이 Fig. 3(c)의 grouped convolution 형태를 구현 선택으로 삼았듯, Lucid 구현도 bottleneck 블록에서 `groups=cardinality`를 주는 방식으로 ResNeXt의 핵심을 직접 반영한다.

Lucid에서의 핵심 대응 관계를 먼저 요약하면 다음과 같다.

- 논문 Eq.(2)(3)의 cardinality $C$ → `ResNeXt` 생성자의 `cardinality`, `_Bottleneck` 생성자의 `cardinality`  
- 논문 Fig. 3(c)의 grouped convolution → `_Bottleneck`의 `conv2`에서 `groups=cardinality`  
- 논문에서 말하는 bottleneck width $d$ → `base_width`로 제어(폭 계산을 통해 group당 채널 수가 결정됨)  
- 논문 Table 1의 stage 반복 수($[3,4,6,3]$, $[3,4,23,3]$) → `resnext_*` 팩토리 함수의 `layers` 

### 0️⃣ 파일 구성과 모델 엔트리
ResNeXt 전용 베이스 클래스는 매우 얇다. 실제로는 ResNet 빌더를 상속해, ResNeXt에 필요한 두 하이퍼파라미터(cardinality, base_width)를 `block_args`로 전달하는 역할만 수행한다.

```python
class ResNeXt(ResNet):
    def __init__(
        self,
        block: nn.Module,
        layers: list[int],
        cardinality: int,
        base_width: int,
        num_classes: int = 1000,
    ) -> None:
        block_args = {"cardinality": cardinality, "base_width": base_width}
        super().__init__(block, layers, num_classes, block_args=block_args)


@register_model
def resnext_50_32x4d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    layers = [3, 4, 6, 3]
    return ResNeXt(_Bottleneck, layers, 32, 4, num_classes, **kwargs)


@register_model
def resnext_101_32x4d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    layers = [3, 4, 23, 3]
    return ResNeXt(_Bottleneck, layers, 32, 4, num_classes, **kwargs)


@register_model
def resnext_101_32x8d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    layers = [3, 4, 23, 3]
    return ResNeXt(_Bottleneck, layers, 32, 8, num_classes, **kwargs)


@register_model
def resnext_101_32x16d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    layers = [3, 4, 23, 3]
    return ResNeXt(_Bottleneck, layers, 32, 16, num_classes, **kwargs)


@register_model
def resnext_101_32x32d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    layers = [3, 4, 23, 3]
    return ResNeXt(_Bottleneck, layers, 32, 32, num_classes, **kwargs)


@register_model
def resnext_101_64x4d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    layers = [3, 4, 23, 3]
    return ResNeXt(_Bottleneck, layers, 64, 4, num_classes, **kwargs)
```

여기서 핵심은 `block_args` 딕셔너리다. ResNet 빌더는 블록을 생성할 때 `**block_args`를 전달하도록 설계되어 있으므로, ResNeXt는 그 통로를 이용해 `_Bottleneck`에 `cardinality`, `base_width`를 흘려보낸다. 즉, ResNeXt는 ResNet과 같은 stage 구조를 유지하되, bottleneck 블록의 내부 연산을 group conv로 바꾸는 방식으로 구현된다.

`layers`는 논문 Table 1의 stage 반복 수와 동일하며, `32x4d` 같은 표기는 `cardinality=32`, `base_width=4`라는 두 숫자로 그대로 구현된다.

### 1️⃣ ResNet 빌더의 핵심: `block_args`가 Stage 전체로 전파되는 방식
ResNeXt는 ResNet을 상속하지만, 실제로는 ResNet이 가진 일반화된 조립 로직을 그대로 사용한다. `ResNet`은 `block`, `layers`, `block_args`를 받아 stage를 조립하며, 핵심은 `_make_layer`가 블록을 생성할 때마다 `**block_args`를 전달한다는 점이다.

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

        if deep_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, stem_width, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(),
                nn.Conv2d(stem_width, stem_width, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(),
                nn.Conv2d(stem_width, self.in_channels, 3, padding=1, bias=False),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_channels, self.in_channels, 7, stride=2, padding=3, bias=False
                ),
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(),
            )

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

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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

        layers = [
            block(self.in_channels, out_channels, stride, downsample, **block_args)
        ]
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1, **block_args))

        return nn.Sequential(*layers)
```

처럼 첫 블록과 이후 반복 블록 모두에 `block_args`를 전달한다. 이 구조는 논문에서 말한 두 가지 반복 규칙(같은 stage에서는 같은 블록 형태가 반복됨)과 매우 잘 맞는다. 즉, ResNeXt에서 cardinality는 stage 전체에 걸쳐 동일하게 적용되며, stage 경계에서 해상도/채널이 바뀔 때만 downsample이 들어간다.

### 2️⃣ `_Bottleneck` 내부: Grouped Convolution으로 ResNeXt 구현
ResNeXt 논문에서 가장 중요한 구현 대응은 Fig. 3(c)이다. Lucid의 `_Bottleneck`은 바로 그 형태를 `groups=cardinality`로 구현한다.

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

        self.conv1 = nn.ConvBNReLU2d(in_channels, width, kernel_size=1, stride=1, conv_bias=False)
        self.conv2 = nn.ConvBNReLU2d(
            width, width, kernel_size=3, stride=stride, padding=1,
            groups=cardinality, conv_bias=False,
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

        self.se = nn.SEModule(out_channels * self.expansion, **se_args) if se else None
        self.relu = nn.ReLU()
        self.downsample = downsample
```

여기서 논문과의 대응을 핵심만 잡으면 다음과 같다.

#### Width 계산식이 논문 표기(32×4d)를 고정
논문 표기에서 `32×4d`는 group 수가 $32$, group당 채널 폭이 $4$라는 의미였다. Lucid는 이를 다음 공식으로 강제한다.

- `out_channels`는 stage의 기준 폭(ResNet 채널 규칙: $64/128/256/512$)  
- `base_width`는 group당 폭을 결정하는 스케일(논문에서의 $d$에 대응)  
- `cardinality`는 group 수(논문에서의 $C$)

그리고 `width = floor(out_channels*(base_width/64))*cardinality`로 전체 폭을 만든다. 예를 들어 ResNeXt-50(32×4d)의 `conv2` stage에서 `out_channels=64`, `cardinality=32`, `base_width=4`이면

- `out_channels*(base_width/64) = 64*(4/64) = 4`  
- `width = 4*32 = 128`

이 된다. 즉, `conv1(1×1)`이 $128$ 채널로 확장하고, `conv2(3×3)`가 $128$ 채널을 `groups=32`로 처리하므로, 각 group은 4채널을 담당한다. 이는 논문 Table 1의 ResNeXt-50 `conv2` 블록($[1×1, 128; 3×3, 128, C=32; 1×1, 256]$)과 정확히 일치한다.

#### `groups=cardinality`의 중요성
`conv2`에서 `groups=cardinality`를 주면, 입력 채널과 출력 채널이 `cardinality`개의 그룹으로 나뉘어 **그룹별로 독립적인 convolution**이 수행된다. 위 예시에서 128 채널을 32개 그룹으로 나누면 그룹당 4채널이 되고, 이는 논문 Fig. 3(c)의 grouped convolution(각 group이 저차원 embedding에서 3×3을 수행)과 대응된다.

즉, Lucid 구현은 논문에서의 aggregated transformations을 코드에서 **별도의 루프로 구현하지 않는다**. 대신 grouped convolution primitive가 그 aggregation을 내장된 방식으로 수행하도록 만든다. 논문이 Fig. 3(c)를 구현 선택으로 삼은 이유와 동일한 설계다.

### 3️⃣ Residual Add와 Projection Shortcut
ResNeXt의 블록이 복잡해 보이더라도, residual add 자체는 _ResNet과 동일_ 하다. `_Bottleneck.forward`의 마지막은 다음처럼 끝난다.

```python
def forward(self, x: Tensor) -> Tensor:
    identity = x
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu(out)
    return out
```

`out += identity`는 논문 Eq.(3)에서의 $y = x + \sum_i T_i(x)$에서 마지막 덧셈에 해당한다. ResNeXt의 차이는 $\sum_i T_i(x)$가 블록 내부에서 group conv로 구현된다는 점이지, residual 연결의 형태가 바뀌는 것은 아니다.

Projection shortcut은 `ResNet._make_layer`에서 stride/채널 불일치일 때 만들어진다.

```python
if stride != 1 or self.in_channels != out_channels * block.expansion:
    downsample = nn.Sequential(
        nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels * block.expansion),
    )
```

논문 구현 설명에서도, 차원 증가 구간만 projection(type B)이고 나머지는 identity shortcut이라고 되어 있다. Lucid에서도 `avg_down=False` 기본값이면 stride가 걸린 1×1 conv projection이 사용되므로, 논문 설정과 가장 직접적으로 대응된다.

### 4️⃣ 전체 Forward 로직
ResNeXt는 ResNet의 골격을 그대로 상속하므로, forward 흐름도 **동일**하다.

- stem(7×7 conv) → maxpool
- stage `layer1`$\ldots$`layer4`
- global average pooling → fc

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

따라서 Lucid에서 ResNeXt를 이해하는 가장 좋은 방법은, ResNet을 이미 알고 있다고 가정하고 블록 내부만 **ResNeXt 스타일로 바뀐다고 보는 것**이다. 논문도 ResNet-50/101에서 블록을 단순히 대체한다고 표현하며, 그 대체가 group conv와 cardinality로 실현된다는 점을 강조한다.

---

## ✅ 정리
**ResNeXt**는 ResNet 이후의 설계 난점(복잡해지는 하이퍼파라미터 공간)을 정면으로 다루면서도, 해결책은 의외로 단순한 형태로 제시한다. Residual branch를 여러 개의 동형 변환 $T_i$로 분해해 합산하는 Eq.(2)(3)는 multi-branch 구조를 제공하지만, 경로별 커스터마이징을 허용하지 않음으로써 설계를 단순화한다. 이때 경로 수 $C$를 cardinality로 정의해 깊이/너비와 다른 설계 축으로 분리했고, 복잡도 보존 조건(Table 2)에서 $C$를 늘리면 정확도가 좋아진다는 것을 Table 3/4로 설득한다.

특히 동일 2× FLOPs 조건에서 depth/width 증가보다 cardinality 증가가 더 효율적으로 error를 줄인다는 결과는, ResNeXt를 단지 하나의 모델이 아니라 설계 원리로 이해하게 만든다. 또한 ImageNet-5K, CIFAR, COCO detection으로 확장해 결과를 제시함으로써, ResNeXt가 backbone으로서 표현 학습을 강화하고 전이 성능에도 기여한다는 메시지를 강화한다.

- **문제 정의**: 설계 공간이 커질수록 하이퍼파라미터가 폭발하고 과적합 위험이 커진다.
- **핵심 아이디어**: 동일 topology의 변환을 $C$개 합산하는 aggregated transformation(Eq.(2)).
- **핵심 수식**: residual 출력은 $y=x+\sum_{i=1}^{C}T_i(x)$(Eq.(3)).
- **복잡도 보존**: Eq.(4)로 $C$–$d$ trade-off를 만들고(Table 2), 같은 FLOPs에서 정확도 개선을 비교한다.
- **실험 결론**: depth/width보다 cardinality 증가가 더 효율적인 축으로 나타난다(Table 4).
- **해석 관점**: ResNeXt는 multi-branch를 설계 복잡도 없이 도입하는, 모듈화된 backbone 설계 문법에 가깝다.

#### 📄 출처
Xie, Saining, et al. "Aggregated Residual Transformations for Deep Neural Networks." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, arXiv:1611.05431.
