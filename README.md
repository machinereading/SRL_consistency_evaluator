# SRL_consistency_evaluator

## 개요
SRL 데이터의 일관성 검증을 위한 모듈입니다. 

## 필요조건
* `python 3`
* `transformers` ([Link](https://github.com/huggingface/transformers))

## 설치방법
먼저 `transformers` 를 설치합니다 (필요시)
```
pip3 install transformers # if needed
```
본 레포지토리도 클론합니다.
```
git clone https://github.com/machinereading/SRL_consistency_evaluator.git
```

## 사용법

### 학습

학습은 다음과 같이 수행합니다.

예를들어, 학습하고자 하는 데이터(`json`파일들)가 있는 폴더가 `/data/`라고 해 봅시다. 이 경우 다음과 같이 학습합니다.
```
python3 training.py --train /data
```

다음의 argument 들을 지정할 수 있습니다.
- `--epoch`: 학습 횟수 (default=3)
- `--model`: 모델이 저장될 폴더 (default='./model/')
- `--batch`: 학습에 사용할 배치사이즈 파라미터 (default=3)
- `--split`: 학습데이터 중 몇 퍼센트를 사용할 지 지정 (default=100)
- `--n_split`: 학습데이터를 순차적으로 몇 개로 나누어서 학습할지 지정 (default=False)

#### 기본학습
```
python3 training.py --train /data --model ./model
```
이 경우 학습데이터 전체를 사용해서 모델을 학습합니다. 모델은 `./model` 폴더에 저장됩니다.

#### 학습데이터를 N개로 분리하여 N개 모델 학습
```
python3 training.py --train /data --model ./model --n_split 10
```
이 경우 학습데이터를 10개로 나누어서 각각을 사용하여 10개의 모델을 학습합니다. 각 모델들은 `./model/`폴더 아래에 `split_0`, `split_1`, ... 등의 폴더에 저장됩니다. 

학습데이터를 나누는 기준은 문서id 기준(즉 파일명)으로 N개로 분할합니다. 각 분할에 해당하는 학습데이터의 목록은 `splited_documents.txt` 파일에 저장됩니다.

#### 학습데이터 중 임의의 n 퍼센트만 사용하여 모델 학습
```
python3 training.py --train /data --model ./model --split 10
```
이 경우 학습데이터 중 임의의 10%를 사용하여 1개의 모델을 학습합니다. 모델은 `./model` 폴더에 저장됩니다.


### 평가
TODO (0318)





## Licenses
* `CC BY-NC-SA` [Attribution-NonCommercial-ShareAlike](https://creativecommons.org/licenses/by-nc-sa/2.0/)

## Publisher
[Machine Reading Lab](http://mrlab.kaist.ac.kr/) @ KAIST

## Contact to author
Younggyun Hahm. `hahmyg@kaist.ac.kr`, `hahmyg@gmail.com`
