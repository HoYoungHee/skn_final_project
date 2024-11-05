# SKN 2기_Final Project_1팀

AI 기반 모의 면접 시스템으로, 지원자의 이력서를 바탕으로 맞춤형 질문을 생성하고 면접을 진행하며, 면접 종료 여부를 파악하는 기능을 제공

---

## 프로젝트 개요
- **프로젝트명**: 지원자의 이력서를 기반으로 지원 기업 맞춤 정보를 활용한 실전 대비 AI 모의면접 시스템
- **목적**: AI 기술을 활용하여 지원자에게 실전 대비 모의 면접 경험을 제공
- **주요 기능**:
  - **맞춤형 질문 생성**: 사용자의 이력서를 분석하여 개인 맞춤형 질문 제공
  - **면접 종료 판단**: AI 에이전트를 통해 면접 종료 여부를 판단

---

## 폴더 구조
skn_final_project-main/ │ ├── .gitignore ├── api/ │ ├── api/ # 메인 API 코드 │ └── interview_end_chacker_lamma # 면접 종료 판단 에이전트 설정 파일 ├── readme.md # 프로젝트 설명 파일


- **api**: 메인 API와 면접 종료 판단 에이전트 관련 파일이 포함된 폴더
- **interview_end_chacker_lamma**: 면접 종료 여부를 판단하는 설정 스크립트

---

## 설치 및 실행

### 1. 환경 설정
- 프로젝트를 로컬에 복제 후 필요한 패키지를 설치합니다.

### 2. Docker 설정
- `interview_end_chacker_lamma` 설정 파일을 통해 LLaMA 모델을 실행합니다.
- **예시**:
  ```plaintext
  FROM llama3.1

  # 창의성 조정
  PARAMETER temperature 1

  # 에이전트 역할 설명
  SYSTEM """
  ## 역할: 당신은 면접이 종료됐는지 파악하는 에이전트입니다.
  입력된 문장을 분석하여 면접 종료 여부를 판단합니다.
  종료라면 '종료'

## 폴더 구조
skn_final_project-main/
│
├── .gitignore               # Git 무시 파일
├── api/                     # API 코드 및 설정 폴더
│   ├── api/                 # 메인 API
│   └── interview_end_chacker_lamma  # 면접 종료 판단 설정 파일
├── readme.md                # 프로젝트 설명

## 주요 파일 설명
- **api**
api: AI 모델을 사용하여 면접 질문을 생성하고 사용자 응답을 처리하는 메인 API 코드가 포함됩니다.
- **interview_end_chacker_lamma**
역할: 면접 종료 여부를 판단하는 에이전트 설정
설정: LLaMA 모델을 사용하며, 창의성 및 응답 방식 조정 가능

