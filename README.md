# capstone_elevator_algorithm



## Project schedule

### 기간 2021년 9월 16일 목요일 ~
|주차|목표|설명|
|------|---|---|
|3주차|주제설정, 스케줄설정|논문 찾기|
|4주차|문제상황 확인|torch 이해하기, ppo 논문 공부|
|5주차|알고리즘 구현|코드 구현하기, ppo 알고리즘 정리|
|6주차|테스트2|테스트3|
|7주차|테스트2|테스트3|
|8주차|테스트2|테스트3|
|9주차|테스트2|테스트3|
|10주차|테스트2|테스트3|

## Proximal Policy Optimization Algorithms 리뷰

## 1 Introduction

### Idea

"현재 얻은 데이터로 가능한 큰 step만큼 update를 하고싶은데, 

그렇다고 너무 멀리 update를 해서 성능을 떨어뜨리고 싶지도 않을 때"

- 조금만 update를 할 때, 학습이 느리다.
- 가지고 있는 데이터를 최대한 full로 사용하기 위해서는 업데이트를 큰 범위로 해야한다.
- 하지만 너무 많이 업데이트를 한다면 성능이 떨어진다.
- 이 때, 얼만큼 업데이트를 해야 안전한가?

### PPO 알고리즘의 특징

- 데이터를 업데이트에 사용 후 버리는 것이 아니라,
- 이미 만든 데이터를 여러 번 사용할 수 있다.

#### 샘플을 재사용 하기 위해 → IS(importance sampling)

- 확률분포 p의 기댓값을 구하는 것과 밀접한 연관이 있다.
- 효율적으로 기댓값 추정하기 위해 고안되었다.
- 확률분포 p(x)의 확률 밀도 함수는 알고있지만, p에서 샘플을 생성하기가 어려울 때,
- 비교적 샘플을 생성하기가 쉬운 q(x)에서 샘플을 생성하며 p의 기댓값을 계산한다.

<img src = "https://user-images.githubusercontent.com/78775910/135992074-6f57f4a1-463f-4eeb-b065-0111cc60d2a3.png" width="40%" height="height 20%">


## 2 Background

### 2-1 Policy Gradient Methods

#### Policy gradient

<img src = "https://user-images.githubusercontent.com/78775910/135992255-91de7963-c50f-499c-8ff4-8d899a53cc04.png" width="30%" height="height 15%">

#### Policy gradient로 미분할 수 있는 loss함수

<img src = "https://user-images.githubusercontent.com/78775910/135992296-83af68a7-eb8f-4f4e-93f0-d5e656494af5.png" width="30%" height="height 15%">

- 자동으로 gradient를 계산해주는 auto diff library를 사용한다고 한다면,
- 첫 번째 식은 미분이 된 결과이기 때문에, 미분되기 전에 loss 함수가 있어야 된다.
- gradient log pi가 log pi를 theta로 미분한다. 이 때 A는 상수이기 때문에 theta와 관련이 없다.
- auto diff library를 이용해서 함수를 불러오면 gradient계산 + 업데이트를 할 수 있다.

### "But don't want to optimize it too far"

- A가 양수인 경우 → maximize하기위해 log pi가 최대한 커져야 된다. → 확률에서 가장 큰 값은 1이다. → maximize 하기위해서 필요한 값은 log 1
- A가 음수인 경우 → pi를 0으로 만들어 -∞으로 만들 수 있다. → 그렇게 큰 값으로 optimize를 할 수 없다. → 그럼 얼만큼 업데이트를 하는게 적합한가?

### 2-2 Trust Region Methods - TRPO 설명

<img src = "https://user-images.githubusercontent.com/78775910/135992412-1d876ca9-2f09-4e9b-9f82-35392c209a6c.png" width="40%" height="height 15%">

<img src = "https://user-images.githubusercontent.com/78775910/135992493-ba4ea7c9-39f7-4871-8f70-3d11d2d2855b.png" width="40%" height="height 15%">

## 3 Clipped Surrogate Objective

<img src = "https://user-images.githubusercontent.com/78775910/135992555-0de654ad-af6e-4b0a-bedb-925ac7212850.png" width="40%" height="height 15%">

<img src = "https://user-images.githubusercontent.com/78775910/135992607-3926ee39-2127-490a-b60f-6a1b925206e2.png" width="40%" height="height 15%">

- clip 함수 - clip(a,b,c)가 있으면 결과값이 b와 c 사이에 있도록 유도하는 함수이다.
1. b < a < c 이면 clip(a,b,c) = a
2. a < b 이면 clip(a,b,c) = b
3. c < a 이면 clip(a,b,c) = c

<img src = "https://user-images.githubusercontent.com/78775910/136148016-62f8a44a-946c-4262-8104-e875475165f2.png" width="40%" height="height 15%">

- 가중치인 A가 양수 = 좋은 sample 즉, 발생할 확률(r)을 증가시키려 함. 하지만 일정 이상으로는 증가시키지 못하게 한다.
- 가중치인 A가 음수 = 나쁜 sample 즉, 발생할 확률(r)을 감소시키려 함. 하지만 일정 이하로 감소시키지 못하게 한다.
- 이러한 과정을 통해 좋은 sample은 많이 재활용, 나쁜 sample은 적게 재활용한다.
- 가중치인 A를 최적화 시키면 되는데, 이를 반복해서 업데이트(큰 폭으로 변화하지 않기 때문에)
- 논문에 따르면 실험 결과로는 뒤에 나오는 Adaptive KL Penalty Coefficient보다 결과가 좋다고 한다.

## 4 Adaptive KL Penalty Coefficient

<img src = "https://user-images.githubusercontent.com/78775910/135992658-e300246e-6dd4-4869-b162-43406e267462.png" width="40%" height="height 15%">

- 패널티인 d가 특정한 수인 d(target)보다 작으면 베타를 감소시킨다. d가 작았다는 것은 변동이 작다는 것이기 때문에 베타를 감소시켜 변동 폭을 증가시킨다.
- 패널티인 d가 특정한 수인 d(target)보다 크면 베타를 증가시킨다. d가 컸다는 것은 그만큼 변동이 크다는 것이기 때문에 베타를 증가시켜 변동 폭을 감소시킨다.
- 위의 수식은 기존 TRPO 모형에서 실질적인 부분을 발전시킨 모형이다. TRPO는 2차 근사까지 사용하여 학문적인 정확도는 증가할지 몰라도 복잡한 수식이기 때문에 실제 상황에 적용하기 어렵고 처리 속도가 늦어진다.
