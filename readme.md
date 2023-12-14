# atari taxi-v3 게임의 강화학습 approach

1. Q-learning 학습 및 실행방법
    q-learning은 학습이 끝나면 자동으로 실행되게 구성하였다.
    실행방법은 다음과 같다.

	`
	python3 q_learning.py
	`

2. DQN 학습 방법
DQN은 학습과 검증을 따로 실행하여야 한다.
	`
	python3 train.py
	`

3. DQN 검증 방법
DQN 실행시 repository에 제공된 pt를 기본으로 사용하도록 설정되어 있다.
다른 checkpoint를 사용하고 싶다면 test.py의 load_model() 에서 경로를 변경하면 된다.
	`
	python3 test.py
	`
