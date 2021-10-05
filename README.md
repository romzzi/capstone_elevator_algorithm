# capstone_elevator_algorithm

# 1 Introduction

## Idea

"현재 얻은 데이터로 가능한 큰 step만큼 update를 하고싶은데, 

그렇다고 너무 멀리 update를 해서 성능을 떨어뜨리고 싶지도 않을 때"

- 조금만 update를 할 때, 학습이 느리다.
- 가지고 있는 데이터를 최대한 full로 사용하기 위해서는 업데이트를 큰 범위로 해야한다.
- 하지만 너무 많이 업데이트를 한다면 성능이 떨어진다.
- 이 때, 얼만큼 업데이트를 해야 안전한가?

## PPO 알고리즘의 특징

- 데이터를 업데이트에 사용 후 버리는 것이 아니라,
- 이미 만든 데이터를 여러 번 사용할 수 있다.

### 샘플을 재사용 하기 위해 → IS(importance sampling)

- 확률분포 p의 기댓값을 구하는 것과 밀접한 연관이 있다.
- 효율적으로 기댓값 추정하기 위해 고안되었다.
- 확률분포 p(x)의 확률 밀도 함수는 알고있지만, p에서 샘플을 생성하기가 어려울 때,
- 비교적 샘플을 생성하기가 쉬운 q(x)에서 샘플을 생성하며 p의 기댓값을 계산한다.

![importance sampling](https://user-images.githubusercontent.com/78775910/135992074-6f57f4a1-463f-4eeb-b065-0111cc60d2a3.png){: width="50%" height="20%"}


# 2 Background

## 2-1 Policy Gradient Methods

### Policy gradient

![Policy gradient](https://user-images.githubusercontent.com/78775910/135992255-91de7963-c50f-499c-8ff4-8d899a53cc04.png)

### Policy gradient로 미분할 수 있는 loss함수

![loss function](https://user-images.githubusercontent.com/78775910/135992296-83af68a7-eb8f-4f4e-93f0-d5e656494af5.png)

- 자동으로 gradient를 계산해주는 auto diff library를 사용한다고 한다면,
- 첫 번째 식은 미분이 된 결과이기 때문에, 미분되기 전에 loss 함수가 있어야 된다.
- gradient log pi가 log pi를 theta로 미분한다. 이 때 A는 상수이기 때문에 theta와 관련이 없다.
- auto diff library를 이용해서 함수를 불러오면 gradient계산 + 업데이트를 할 수 있다.

### "But don't want to optimize it too far"

- A가 양수인 경우 → maximize하기위해 log pi가 최대한 커져야 된다. → 확률에서 가장 큰 값은 1이다. → maximize 하기위해서 필요한 값은 log 1
- A가 음수인 경우 → pi를 0으로 만들어 -∞으로 만들 수 있다. → 그렇게 큰 값으로 optimize를 할 수 없다. → 그럼 얼만큼 업데이트를 하는게 적합한가?

## 2-2 Trust Region Methods - TRPO 설명임

![Trust Region Methods1](https://user-images.githubusercontent.com/78775910/135992412-1d876ca9-2f09-4e9b-9f82-35392c209a6c.png)

![Trust Region Methods2](https://user-images.githubusercontent.com/78775910/135992493-ba4ea7c9-39f7-4871-8f70-3d11d2d2855b.png)

# 3 Clipped Surrogate Objective

![Clipped Surrogate Objective1](https://user-images.githubusercontent.com/78775910/135992555-0de654ad-af6e-4b0a-bedb-925ac7212850.png)

![Clipped Surrogate Objective2](https://user-images.githubusercontent.com/78775910/135992607-3926ee39-2127-490a-b60f-6a1b925206e2.png)

# 4 Adaptive KL Penalty Coefficient

![Adaptive KL Penalty Coefficient](https://user-images.githubusercontent.com/78775910/135992658-e300246e-6dd4-4869-b162-43406e267462.png)
