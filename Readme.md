## 프로젝트 구성
* __DataSets__
* __Matlab__: Preprocessing Matlab 코드 ([NPMK Toolkit](https://github.com/BlackrockNeurotech/NPMK) 포함)
* __Py__: Classification python 코드

## 사전 준비
1. __Matlab 설치__, 버전: 24.2.0.2863752 (R2024b) Update 5
2. __EEGLAB 설치(Matlab Addon)__, 버전: 2025.0.0
3. __Python 환경 설정__
* __개발환경 설정__(pycharm, vscode)
* __가상환경 설정__(.venv, conda, etc...)
* __패키지 설치__ 
    > pip install -r requirements.txt

4. __DataSet(Raw) 준비__

    다운로드: [구글 드라이브](https://drive.google.com/file/d/14yxq0ZfrMdkKPkI47Cw0h3SvUIXl-wvX/view?usp=drive_link)

    압축해제: /DataSets/raw

## 프로젝트 실행
### 데이터 전처리
1. Matlab/main.m 스크립트 실행,

    저장경로: /DataSets/convert

### 분류 모델 수행
1. Py/run_classifier.py 실행

