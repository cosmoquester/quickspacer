# QuickSpacer

[![codecov](https://codecov.io/gh/psj8252/quickspacer/branch/master/graph/badge.svg)](https://codecov.io/gh/psj8252/quickspacer)


- 빠른 속도와 준수한 정확도를 목표로하는 한국어 띄어쓰기 교정 모델입니다.

# Demo

데모는 TFJS로 만들어져 있으며 https://psj8252.github.io/quickspacer/ 에서 사용해보실 수 있습니다.

# Install & Usage

```bash
pip3 install git+https://github.com/psj8252/quickspacer.git
```
위 명령어로 설치하실 수 있습니다.

```python
from quickspacer import Spacer

spacer = Spacer()
spacer.space(["띄어쓰기를안한나쁜말", "또는 띄어쓰기가 잘 되어있는 좋은 말"])
```
이런식으로 사용하실 수 있습니다. spacer.space() 함수는 띄어쓰기가 된 리스트를 반환합니다.


만약 모델을 따로 학습시키셨다면, `spacer = Spacer("somewhere/my_custom_savedmodel_dir")`로 인자를 넘겨 직접학습한 모델을 사용할 수 있습니다.

# Train

## Make Vocab

```bash
python -m scripts.make_vocab \
    --input-dir [corpus_directory_path] \
    --vocab-path [vocab_file_path]
```
기본 vocab은 resources/vocab.txt에 존재하지만 따로 문자열이 필요하거나 다른 언어의 띄어쓰기 모델을 만들 예정이라면 위 명령어를 통해 새로 vocab 파일을 만들 수 있습니다.

## Model Train

```bash
python -m scripts.train_quickspacer \
    --model-name [model_name] \
    --dataset-file-path [dataset_paths] \
    --output-path [output_dir_path]
```
모델은 위 명령어로 학습할 수 있습니다.
- 현재 레포지토리에 존재하는 모델은 [ConvSpacer1, ConvSpacer2, ConvSpacer3] 세 종류입니다. model-name에는 이 세 가지 중 하나를 입력합니다.
- 각 모델마다 사용하는 파라미터가 있는데 configs 디렉토리에 기본 설정파일들이 있으며 이를 수정해서 사용하면 됩니다.
- dataset은 UTF-8로 인코딩 된 텍스트파일 형태입니다. 띄어쓰기가 올바르게 되어있는 문장이라고 가정하고 학습합니다. 여러 파일을 입력할 수 있습니다. ex) "corpus_*.txt"

```bash
python -m scripts.train_quickspacer --help
```
를 보면 여러 학습 parameter을 입력할 수 있습니다.

# Deploy

배포는 SavedModel을 이용한 배포와 Tensorflowjs를 이용한 배포가 가능합니다.

## Deploy using SavedModel

### Make SavedModel

```bash
python -m scripts.convert_to_savedmodel \
    --model-weight-path [model_weight_path] \
    --output-path [saved_model_path]
```
위 명령어를 통해 모델을 TF SavedModel 형식으로 변환할 수 있습니다.
- model-weight-path는 train을 통해서 models에 저장된 경로를 입력하면 되는데 "spacer-XXepoch-xxx.index" 이런 식으로 파일이 존재하는데 "spacer-XXepoch-xxx" 까지만 입력합니다.
- SavedModel은 그대로 Tensorflow로 Load해서 그대로 사용해도 되고, Tensorflow serving 등을 통해 API 서버로 Deploy할 수 있습니다.

### SavedModel Additional Description

convert_to_savedmodel로 변환한 모델은 두 개의 signature function을 가지고 있습니다. 아래 명령들은 SavedModel을 만드는 것과는 무관하며 추가적인 설명을 위한 것입니다.
```bash
$ saved_model_cli show --dir saved_spacer_model/1 \
    --tag_set serve \
    --signature_def serving_default
2020-11-15 17:02:30.861944: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
The given SavedModel SignatureDef contains the following input(s):
  inputs['texts'] tensor_info:
      dtype: DT_STRING
      shape: (-1)
      name: serving_default_texts:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['spaced_sentences'] tensor_info:
      dtype: DT_STRING
      shape: unknown_rank
      name: StatefulPartitionedCall_1:0
Method name is: tensorflow/serving/predict
```
- default는 text를 입력받고 띄어쓰기가 완료된 문장을 반환하도록 되어있습니다. 위의 saved_model_cli를 통해 살펴보면 DT_STRING이 입출력인 것을 알 수 있습니다.

```bash
$ saved_model_cli run --dir saved_spacer_model/1 \
    --tag_set serve \
    --signature_def serving_default \
    --input_exprs 'texts=["근데이것좀띄워주시겠어요?", "싫은데영ㅎㅎ"]'
2020-11-15 17:06:27.735637: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2020-11-15 17:06:28.659347: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
[각종 TF log들 ...]
Result for output key spaced_sentences:
[b'\xea\xb7\xbc\xeb\x8d\xb0 \xec\x9d\xb4\xea\xb2\x83 \xec\xa2\x80 \xeb\x9d\x84\xec\x9b\x8c \xec\xa3\xbc\xec\x8b\x9c\xea\xb2\xa0\xec\x96\xb4\xec\x9a\x94?'
 b'\xec\x8b\xab\xec\x9d\x80\xeb\x8d\xb0 \xec\x98\x81\xe3\x85\x8e\xe3\x85\x8e']
```
- 이런 식으로 saved_model이 잘 저장되었고 제대로 동작하는지 확인할 수 있습니다.
- Unicode 바이너리로 나와서 조금 불편한데 한글로 바꿔보면 ["근데 이것 좀 띄워 주시겠어요?","싫은데 영ㅎㅎ"] 으로 띄어쓰기를 해주었습니다.

```bash
$ saved_model_cli show --dir saved_spacer_model/1 \
    --tag_set serve \
    --signature_def model_inference
2020-11-15 17:03:19.988061: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
The given SavedModel SignatureDef contains the following input(s):
  inputs['tokens'] tensor_info:
      dtype: DT_INT32
      shape: (-1, -1)
      name: model_inference_tokens:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['output_0'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, -1)
      name: StatefulPartitionedCall:0
Method name is: tensorflow/serving/predict
```
- 또 하나의 Signature 함수는 문장을 글자를 다 잘라서 Vocab을 이용해 숫자로 변환된 입력을 받고, 각 자리를 띄워야할 확률을 알려줍니다.
- 아까와 같은 문장을 숫자로 변환하여 테스트해보겠습니다.

```bash
$ saved_model_cli run --dir saved_spacer_model/1 \
    --tag_set serve \
    --signature_def model_inference \
    --input_exprs 'tokens=[[88,26,4,100,112,1241,93,64,38,56, 6,19,15],[216,33,26,202,67,67,0,0,0,0,0,0,0]]'
[각종 Tensorflow log...]
Result for output key output_0:
[[2.5608379e-03 9.8520654e-01 1.2721949e-03 9.7731644e-01 9.9997485e-01
  1.6742251e-07 5.0763595e-01 2.1732522e-03 1.0649196e-03 2.6994228e-04
  1.1066308e-04 1.4717710e-03 2.8897190e-01]
 [3.6909140e-04 1.2601367e-02 8.4685940e-01 7.0986725e-06 1.3404434e-05
  5.6068022e-10 2.8169470e-12 1.1617506e-17 2.8605146e-17 2.8605146e-17
  2.8605146e-17 5.3611558e-16 2.1768996e-07]]
```
- 두 문장의 길이가 다른 경우에는 위 예시처럼 0으로 패딩을 해줘야합니다.
- 결과를 보면 각 위치마다 띄어야할 확률이 나왔습니다.

### Deploy using Tensorflow/serving docker

```bash
$ docker run  --rm --name test -p 8500:8500 -p 8501:8501 \
    --mount type=bind,source=`pwd`/saved_spacer_model,target=/models/spacer \
    -e MODEL_NAME=spacer \
    -t tensorflow/serving:latest
```
간단히 docker로 tensorflow serving 서버를 여는 명령입니다. 현재 모델 파일은 실제로는 `pwd`/saved_1pacer_model/1 에 저장되어 있습니다.

```bash
$ curl -X POST localhost:8501/v1/models/spacer:predict \
    -H "Content-Type: application/json" \
    -d '{"instances":["근데이것좀띄워주시겠어요!", "싫은데영ㅎㅎ"]}'
{
    "predictions": ["근데 이것 좀 띄워 주시겠어요!", "싫은데 영ㅎㅎ"
    ]
}
```
이제 curl을 이용해 테스트해보면 정상적으로 띄어쓰기가 완료된 문장을 반환하는 것을 볼 수 있습니다. signature function을 지정하면 model_inference만 하는 것도 가능합니다.

## Deploy using Tensorflowjs

다음은 tensorflowjs를 이용해서 서버가 아닌 클라이언트의 브라우저에서 추론하도록할 수 있습니다. 데모페이지에 있는 것도 TFJS를 이용한 것입니다.

### Make TFJS Graph Model

```bash
python -m scripts.convert_to_tfjsmodel \
    --saved-model-path [saved_model_path] \
    --output-path [output_dir_path]
```
이 명령어를 통해 TFJS 모델로 변환할 수 있습니다.
TFJS에서도 문장을 넣고 문장이 나오도록 만들고 싶었지만 Vocab을 포함하고 있는 signature function은 tfjs로 변환하는데 에러가 발생하여 tfjs에선 모델 추론만 하도록 했습니다.
위 파이썬 스크립트를 이용하지 않더라도 `tensorflowjs_wizard`나 `tensorflowjs_converter` 명령어를 바로 사용해도 됩니다.

### Use TFJS Graph Model

- https://js.tensorflow.org/api/latest/ 를 참고하면 js로 사용할 수 있는 API가 정리되어 있습니다.
- tfjs를 사용하려면 https://www.tensorflow.org/js/tutorials/setup에 나와있는 것처럼 tensorflow js 파일을 넣어줘야합니다.

우리가 위에서 만든 건 tf.GraphModel이므로 `tf.loadGraphModel` 함수로 불러와서 사용하면 됩니다.
