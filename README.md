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

![importance sampling](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5cc54a2f-cf56-4ed2-8008-112825712894/Untitled.png)

# 2 Background

## 2-1 Policy Gradient Methods

### Policy gradient

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/98704999-4c9a-435c-a7b5-a4438f61a6c5/Untitled.png)

### Policy gradient로 미분할 수 있는 loss함수

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/19846dc9-ead3-4526-93b8-e54861955451/Untitled.png)

- 자동으로 gradient를 계산해주는 auto diff library를 사용한다고 한다면,
- 첫 번째 식은 미분이 된 결과이기 때문에, 미분되기 전에 loss 함수가 있어야 된다.
- gradient log pi가 log pi를 theta로 미분한다. 이 때 A는 상수이기 때문에 theta와 관련이 없다.
- auto diff library를 이용해서 함수를 불러오면 gradient계산 + 업데이트를 할 수 있다.

### "But don't want to optimize it too far"

- A가 양수인 경우 → maximize하기위해 log pi가 최대한 커져야 된다. → 확률에서 가장 큰 값은 1이다. → maximize 하기위해서 필요한 값은 log 1
- A가 음수인 경우 → pi를 0으로 만들어 -∞으로 만들 수 있다. → 그렇게 큰 값으로 optimize를 할 수 없다. → 그럼 얼만큼 업데이트를 하는게 적합한가?

## 2-2 Trust Region Methods - TRPO 설명임

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a0fafa3f-197e-4b4c-ae57-8aaf6e3e0cbb/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/183ba7c7-378e-4a75-86b8-6194d7a809bc/Untitled.png)

# 3 Clipped Surrogate Objective

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/29fe5fa2-bda6-463b-8096-4e56410d2422/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/576badd8-b8e4-4737-9376-89fe9bfd4619/Untitled.png)

# 4 Adaptive KL Penalty Coefficient

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fbe36871-4232-4acf-9ccf-d662265db869/Untitled.png)
