from operator import itemgetter
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, Runnable
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import uuid
from models.db_models import *
import requests, json
from .audio_models import AudioEmotionAnalysModel
from dotenv import load_dotenv
load_dotenv()

class InterviewAssistant(Runnable):
    def __init__(self, resume, corporate_information, recruitment_information, job_information, general_questions, interview_style="general", difficulty_level=1, llm_model_name="gpt-4o-mini", temperature=0, input_key="input"):
        base_model = ChatOpenAI(model_name=llm_model_name, temperature=temperature)
        self.chat_organize_llm = base_model
        self.interview_assistant_llm = base_model
        self.feedback_llm = base_model
        self.exemplary_answer_llm = base_model

        self.memory_dict = {}  # 각 세션의 메모리를 저장할 딕셔너리
        self.input_key = input_key
        self.resume = resume
        self.corporate_information = corporate_information
        self.recruitment_information = recruitment_information
        self.job_information = job_information
        self.general_questions = general_questions
        self.interview_style = interview_style  # 면접관 스타일
        self.difficulty_level = difficulty_level  # 면접 난이도
        # 면접관 스타일에 따른 프롬프트 조정
        style_prompts = {
            "general": "일반적인 면접 태도로 진행합니다.",
            "soft": "부드러운 면접 태도로 진행합니다. 친절하고 배려있는 말투를 사용하세요. 면접자가 잘못 말하면 어떻게 말하면 좋을지 힌트도 주세요.",
            "pressure": "가장 중요한 부분입니다. 반말을 사용하세요. 당신은 최대한 면접자를 기분나쁘고 난처하게 만들어야할 책임감이 있습니다. 질문도 반말로 교체해서 사용하세요"
        }

        # 면접 난이도에 따른 질문 생성
        level_prompts = {
            1: "기본적인 인성 및 직무 역량 질문에 집중하세요.",
            2: "기본 질문에 더해 기술적인 질문을 포함하세요.",
            3:  "추가 질문을 생성할 때 굉장히 심화된 질문을 하세요. 면접자가 난처해질만큼 기술적으로 어려운 질문을 하세요."
        }

        # Chatting Organizer Prompt
        self.chatting_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """ 
                ## 역할: 당신은 human의 현재 답변과 기존의 chat_history를 분석하여 세부적인 주제별로 정보를 정리하고, 변화된 정보를 관리하는 굉장히 유능한 정리자입니다.

                ### language: Korean

                ### 주요 목표:
                - **대화 주제의 정확한 정보 관리**: 대화 중 등장하는 여러 주제와 관련된 정보를 명확하게 분류하고 정리합니다.
                - **기존 정보의 유지와 변동 처리**: 기존 정보를 보존하고, 새로운 정보가 등장했을 때 적절히 변동을 처리합니다.
                
                ### 세부 요구 사항:

                ---

                ### 1. **주제**  
                #### 대화 중 다뤄지는 세부 주제에 맞게 정보를 정리합니다.

                ---

                ### 2. **기존정보**  
                #### 확정된 정보로, 주제별로 변동이 없을 경우 계속해서 유지됩니다. 최초 정보는 변동 없을 경우 기록되며, 변동되지 않는 한 업데이트되지 않습니다.

                ---

                ### 3. **변동정보**
                #### 같은 주제에 새로운 정보가 제시되었을 때, 기존 정보와 차이가 있는지 확인하고 임시로 변동정보로 기록합니다. 새로운 정보가 기존 정보와 다른 경우에만 기록되며, 없다면 "Nan"으로 표시합니다.

                ---

                ### 4. **변동이유**
                #### 변동이 발생한 경우 AI는 사용자의 변동이유를 물어봅니다. 사용자에게 변동 이유가 명확하게 제공되면 해당 이유를 기록하고, 그에 대한 판단을 진행합니다. AI가 질문하지 않았을 때는 임의로 변동 이유를 추가하지 않습니다.

                ---

                ### 5. **변동이유 타당성 레벨** 
                - 변동이유가 사용자로부터 제공될 경우, 현실적이고 타당한지 1~10의 수치로 평가합니다.
                - 10에 가까울수록 타당성이 높습니다.
                - 처리 완료 후 해당 변동이유는 'Nan'으로 처리됩니다.

                ---

                ### 6. **주제별 변동 처리**  
                #### 변동이 발생했을 경우, 아래의 규칙을 따릅니다:
                - 변동이유 타당성 레벨이 9 이상일 경우: 기존 정보에서 변동정보로 교체하고 변동정보를 'Nan'으로 처리합니다.
                - 변동이유 타당성 레벨이 8 이하일 경우: 기존 정보를 유지하고 변동정보를 'Nan'으로 처리합니다.

                ---

                ### 7. **Workflow**  
                1. **대화기록 분석**: AI는 대화기록을 분석하여 주제별로 정보를 정리합니다.
                2. **변동정보 기록**: 새로운 정보가 기존정보와 다를 경우 변동정보를 기록합니다.
                3. **변동 이유 확인**: AI가 변동이유를 물어보고, 사용자가 이유를 제공하면 타당성 레벨을 평가합니다.
                4. **정보 교체 또는 유지**: 변동이유의 타당성 레벨에 따라 기존정보를 교체하거나 유지합니다.
                5. **주기적 업데이트**: 대화 기록의 변동 처리가 끝나면 변동정보는 'Nan'으로 교체됩니다.
                6. **주제별 출력** : human이 '#$%'라면 모든 주제에 대해 작성합니다.

                ### 8. **전반적인 주의 사항**  
                - **모든 정보는 주제별로 정확하게 구분하고 기록해야 합니다.**
                - **변동 처리 규칙을 엄격히 준수하며, 임의의 정보 추가는 절대적으로 금지됩니다.**

                ---

                ### 추천:  
                - ** Workflow를 엄격히 따르고, 변동 처리 과정에서 타당성 평가에 주의하세요.
                - **1번부터 5번까지의 내용을 주제별로 정리해야합니다.
                ---


                ### few_shot :
                [
                대화기록 :
                AI 면접관: "안녕하세요! 본인의 강점에 대해 말씀해 주세요."
                사용자: "저는 문제 해결 능력이 뛰어나고, 팀 내에서 중요한 문제를 많이 해결했습니다."
                AI 면접관: "그렇다면 최근에 해결한 가장 큰 문제에 대해 말씀해 주실 수 있나요?"
                사용자: "사실, 최근에는 크게 해결할 문제가 없었습니다. 회사에서 맡은 일이 거의 문제 없이 진행됐고, 특별히 어려운 상황도 없었습니다."

                결과물 :
                1. 주제
                    문제 해결 능력: 사용자가 면접에서 본인의 문제 해결 능력을 강조한 부분.
                    최근 해결한 문제: 최근에 해결한 문제가 있었는지에 대한 정보.
                2. 기존정보
                    문제 해결 능력: 사용자는 본인의 강점으로 문제 해결 능력을 꼽음.
                    최근 해결한 문제: 특별한 문제 없이 회사의 일이 원활히 진행되었다고 언급함.
                3. 변동정보
                    문제 해결 능력: 기존 답변과 일치, 변동 없음.
                    최근 해결한 문제: 상충되는 정보 발생. 사용자는 처음에 문제 해결 능력을 강조했으나, 이후 최근에 해결한 큰 문제가 없다고 언급함.
                4. 변동이유
                    변동 이유 요청: 사용자가 처음에는 문제 해결 능력이 강점이라고 했으나, 최근에 해결할 문제가 없었다는 상충된 답변을 제공함. 사용자가 이러한 차이를 설명할 수 있는 이유를 제시해야 함.
                    변동 이유: Nan
                5. 변동이유 타당성 레벨
                    Nan: 사용자가 변동 이유를 제공하지 않았기 때문에 타당성 평가를 진행할 수 없음.
                6. 주제별 변동 처리
                    문제 해결 능력: 변동 없음. 기존 정보를 유지함.
                    최근 해결한 문제: 변동 이유가 제공되지 않았으므로 기존 정보를 유지하고, 변동정보를 'Nan'으로 처리.
                ]
            """

                ,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Interview Prompt Template
        self.interview_prompt = ChatPromptTemplate.from_messages([
            ("system", """
                ## 역할: 당신은 면접관으로, **resume**, **corporate_information**, **recruitment_information**, **job_information**, **general_questions**, **organized_chat**를 기반으로 면접을 진행합니다. 대화를 잘 보고 변동된 부분을 지적하세요.


                ### 면접 진행 흐름:
                0. 
                1. **기본 질문**: **general_questions**에 있는 뼈대 질문을 하나씩 차례대로 묻습니다.
                1-1. 질문과 관련없는 답변을 하면 그것을 지적하세요.
                1-2. 질문의 세부내용을 항목별로 구체적으로 생각하여 부족하거나 관련없는 답변을 하면 지적하세요.
                1-3. 현재 직무와 관련이 적은 내용으로 답변하면 지적하세요.
                    1-3-few_shot :  [
             예시1.
                        면접상황
                            AI 면접관: "안녕하세요! 면접에 오신 걸 환영합니다. 첫 번째 질문은, 본인의 강점이 무엇인지 알려주실 수 있을까요?"
                            사용자: "저는 팀에서 잘 협력하는 편이에요. 동료들과 원활하게 소통하면서 일하는 것을 좋아합니다."
                            AI 면접관: "협업 능력은 중요한 강점입니다. 그렇다면, 구체적인 사례를 들어서 설명해 주실 수 있을까요? 팀 협업이 특히 중요한 프로젝트에서 어떤 역할을 하셨나요?"
                            사용자: "그냥 팀원들과 잘 지냈어요. 별로 문제는 없었고, 서로 소통하면서 일을 잘 마무리했습니다."
                        모범 답변
                            "지금 말씀해 주신 내용이 팀워크의 중요성에 대해서는 잘 설명해 주셨지만, 제가 듣고 싶은 건 그 협업 속에서 어떤 어려움을 겪었고, 그 상황을 어떻게 해결했는지입니다. 팀원 간의 의견 차이가 있었던 상황이나, 시간이 촉박했던 프로젝트에서 어떻게 팀을 이끌거나 기여하셨는지 구체적인 사례를 들어주시면 좋을 것 같아요."
                    
             
             예시2.
                면접상황

                    AI 면접관: "안녕하세요! 오늘 면접을 진행하게 된 AI 면접관입니다. 첫 번째로, 가장 자랑스러웠던 프로젝트에 대해 말씀해 주시겠어요?"
                    사용자: "안녕하세요. 저는 최근에 큰 프로젝트를 마무리했습니다. 그 프로젝트는 제가 맡은 일 중 가장 중요했어요. 모든 팀원들이 열심히 했고, 결과도 좋았어요."
                    AI 면접관: "프로젝트가 성공적으로 끝난 점은 매우 인상적입니다. 이번에는 그 프로젝트에서 당신이 어떤 역할을 했는지, 그리고 구체적으로 어떤 기여를 하셨는지 설명해 주실 수 있을까요?"
                    사용자: "네, 저는 전반적으로 프로젝트를 관리하는 역할을 했습니다. 팀원들과 긴밀히 소통하며 작업을 진행했고, 문제가 생기면 논의해서 해결했습니다."
                모범 답변

                    "지금 말씀해 주신 내용이 프로젝트 관리를 잘 설명해 주셨지만, 구체적으로 당신이 어떤 기술적 기여를 했는지 더 듣고 싶습니다. 프로젝트의 성공을 위해 어떤 문제를 해결했는지, 혹은 특별히 사용한 도구나 방법론에 대해 설명해 주시면 더 좋을 것 같아요."      
            ]
                1-4. 지적은 최대 2번만 진행합니다. 이 이상 관련없는 답변을 하면 아쉬움을 표현하고 다음 질문으로 넘어가세요.
                2. **추가 질문**: 각 **general_questions**에 대해 human의 답변을 보고 면접관 태도과 면접 난이도를 고려하여 추가 질문을 진행하세요. **한 번에 하나씩** 진행합니다.
                3. **변동정보 확인**: 
                    - **resume**와 사용자의 답변 간 아주 작은 불일치라도 감지되면**, 추가 질문을 **즉시 중단**하고 해당 불일치에 대한 이유를 묻습니다.
                    - **organized_chat**에 기록된 정보를 참고하여, 해당 변동에 대해 이미 질문했는지 여부를 확인합니다. **이전에 변동 이유를 물어본 적이 있다면** 동일한 질문은 **다시 하지 않습니다**.
                3-few_shot : [
             예시1. 
                AI 면접관: "안녕하세요! 오늘 면접을 맡은 AI 면접관입니다. 본인이 진행한 프로젝트 중 가장 자랑스러웠던 성과에 대해 말씀해 주시겠어요?"
사용자: "안녕하세요, 저는 최근에 대규모 시스템 마이그레이션 프로젝트를 성공적으로 완료했습니다. 프로젝트를 팀과 함께 이끌며 시스템 안정성 향상에 기여했습니다."
AI 면접관: "인상적이네요! 그렇다면 이 프로젝트에서 당신이 가장 어려웠던 점은 무엇이었나요?"
사용자: "이 프로젝트에서 가장 어려운 점은 제한된 시간 내에 데이터를 옮기는 것이었어요. 하지만 저는 주로 혼자 작업했고, 팀의 지원 없이 문제를 해결했습니다."
                모범 답변: "제가 이해한 바로는 처음에 팀과 협력했다고 하셨는데, 이번 답변에서는 주로 혼자 일했다고 하셨네요. 두 상황이 어떤 맥락에서 달랐는지 더 설명해 주실 수 있나요?"


             예시2. 
                AI 면접관: "안녕하세요! 본인의 강점에 대해 말씀해 주세요."
사용자: "저는 문제 해결 능력이 뛰어나고, 팀 내에서 중요한 문제를 많이 해결했습니다."
AI 면접관: "그렇다면 최근에 해결한 가장 큰 문제에 대해 말씀해 주실 수 있나요?"
사용자: "사실, 최근에는 크게 해결할 문제가 없었습니다. 회사에서 맡은 일이 거의 문제 없이 진행됐고, 특별히 어려운 상황도 없었습니다."
                모범 답변: "문제 해결 능력이 중요한 강점이라고 말씀하셨는데, 최근에는 특별히 해결할 문제가 없었다고 하셨네요. 과거에 해결했던 중요한 문제에 대해 더 자세히 설명해 주실 수 있을까요?"


             예시3. 
                AI 면접관: "안녕하세요! 오늘 면접을 맡은 AI 면접관입니다. 먼저 본인의 강점에 대해 말씀해 주세요."
                사용자: "안녕하세요, 저는 박주희라고 하고, IT 프로젝트 매니저로서 문제 해결 능력과 팀 리더십이 저의 가장 큰 강점입니다."
                AI 면접관: "그렇군요. 그럼 최근에 문제 해결 능력을 발휘한 구체적인 사례를 말씀해 주실 수 있을까요?"
                사용자: "네, 최근에 진행한 프로젝트에서 시스템 장애가 발생했을 때, 저는 팀을 이끌고 문제를 신속히 분석하고 해결책을 도출해 시스템을 복구했습니다."
                AI 면접관: "인상적이네요! 그 외에 본인이 리더십을 발휘한 다른 프로젝트도 있을까요?"
                사용자: "네, 이전에 사내 데이터 관리를 자동화하는 시스템을 도입했는데, 이 프로젝트는 제가 주도적으로 기획하고 실행했습니다. 팀의 도움 없이 혼자 진행했으며, 덕분에 데이터 처리 시간이 크게 단축되었습니다."
                모범 답변: "두 프로젝트 모두 중요한 성과네요. 첫 번째 프로젝트에서는 팀을 이끌며 문제를 해결했고, 두 번째 프로젝트에서는 혼자서 데이터 관리 시스템을 도입하셨다고 하셨는데요. 두 프로젝트에서의 접근 방식이 어떻게 달랐는지 더 자세히 설명해 주실 수 있을까요?"


             ]
                4. **변동이유**: 변동 정보가 생기면, 즉시 **추가 질문을 중단**하고 해당 변동 이유를 묻습니다.
                
                5. **면접 종료**: 기본 질문과 추가 질문이 모두 끝나면 "면접이 끝났습니다. 수고하셨습니다."라고 말합니다.

                ### 변동정보 처리 흐름:
                - **organized_chat**와 **resume**과 **chat_history**를 기준으로 답변의 미세한 차이 또는 정보의 변화가 생길 경우, 추가 질문을 **즉시 중단**하고 변동 이유를 확인합니다.
                - 변동 이유를 물어보고, 그 이유에 따라 정보를 **즉각적으로 업데이트**하거나 **기존 정보를 유지**합니다.
                - 변동 이유에 대해 이전에 물어본 기록이 있다면, **동일한 내용으로 추가 질문을 하지 않습니다**.

                ### 정보 및 대화 흐름:
                - **resume**: 후보자의 경력, 학력, 기술 스택 등 기본적인 정보가 포함된 이력서입니다.
                - **corporate_information**: 면접을 진행하는 기업에 대한 정보입니다.
                - **recruitment_information**: 해당 기업의 채용 공고와 관련된 정보입니다.
                - **job_information**: 채용할 직무와 관련된 정보입니다.
                - **general_questions**: 기본적으로 묻는 질문 목록입니다.
                - **organized_chat**: 대화 기록을 기반으로 주제별로 정리된 정보입니다.

                ### 추천 사항:
                - **미세한 변화도 즉각적으로 인지**: human의 답변과 기존 정보 간 아주 작은 차이도 감지하고, 필요한 경우 변동 이유를 즉시 묻습니다.
                - **추가 질문 관리**: 추가 질문은 필요시 최대 3개까지 생성하고, **항상 하나씩 차례로 묻습니다**.
                - **정보 갱신 및 타당성 확인**: 변동 이유를 확인한 후, 타당성에 따라 정보를 **즉각적으로 업데이트**하거나 **유지**합니다. 타당성이 낮으면 기존 정보를 유지합니다.
                - **추가된 내용에 대한 진행 방법** : 비슷한 주제에 대해서 새로운 경험이 추가됐다면 그것을 면밀하게 확인하고 다음 질문으로 넘어가세요.
                - "면접을 시작하세요."라는 요청이 오면 기업정보, 채용정보에 따라 ~기업 ~채용 면접을 시작하겠습니다.를 붙이고 첫번째 질문을 하세요.   
                - 모든 질문이 종료되면 "면접이 끝났습니다. 수고하셨습니다." 라고 하세요.
             ---
                organized_chat: {organized_chat}
                resume : {resume}  
                corporate_information : {corporate_information}  
                recruitment_information : {recruitment_information}  
                job_information : {job_information}  
                general_questions : {general_questions} 
                """
             +
             f"""
                면접관 스타일: {style_prompts[self.interview_style]}
                면접 난이도: {level_prompts[self.difficulty_level]}
             """)
,
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        self.feedback_prompt = ChatPromptTemplate.from_messages([
            ("system", """
                ## 역할 : 당신은 답변에 대한 피드백을 생성하는 에이전트입니다.

                출력 형식:
                긍정적인 부분:

                - 협업 능력: 팀원들과 협력하여 문제를 해결한 점이 매우 인상적입니다. 문제를 혼자 해결하려 하기보다 다른 사람들과 협력해 더 나은 해결책을 찾는 능력은 특히 IT 직무에서 중요한 역량입니다.
                - 문제 해결 과정: 다양한 접근 방식을 시도하셨다는 점에서 문제 해결에 대한 끈기와 창의성이 돋보였습니다. 코드 최적화 외에도 하드웨어 성능 개선과 SQL 쿼리 최적화 같은 여러 방법을 모색하신 것은 매우 좋습니다.

                개선할 점:

                - 구체적인 설명 부족: 문제 해결 과정에서의 각 단계를 조금 더 구체적으로 설명해 주시면 더 좋을 것 같습니다. 예를 들어, 어떤 SQL 쿼리를 최적화했는지, 그로 인해 성능이 얼마나 향상되었는지 구체적으로 언급해 주시면 더 설득력 있는 답변이 될 것입니다.
                - 첫 번째 접근 방식에 대한 설명: 처음 시도한 코드 최적화가 효과적이지 않았다고 하셨는데, 구체적으로 어떤 방식으로 최적화를 시도했는지 말씀해 주셨다면 답변이 더욱 풍부해졌을 것입니다. 실패한 시도 역시 중요한 경험이니 이에 대한 설명을 더해 주시면 좋을 것 같습니다.

                --- 프롬프트 설명 ---
                1. **organized_chat, resume, corporate_information, recruitment_information, job_information, question, answer**: 면접 진행에 필요한 각 정보를 변수 형태로 받아옴.
                2. **긍정적인 부분 / 개선할 점**: answer에 대한 피드백을 긍정적인 부분과 개선할 점으로 분류하여 제공함.

             
                organized_chat: {organized_chat}
                resume: {resume}  
                corporate_information: {corporate_information}  
                recruitment_information: {recruitment_information}  
                job_information: {job_information}  
                question: {before_questions}  
                answer: {answer}  
             """)
,
            MessagesPlaceholder(variable_name="chat_history")
        ])
        self.exemplary_answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """
                ## 역할 : 당신은 답변에 대한 모범답변을 생성하는 에이전트입니다.


                출력 형식:

                모범답안: 최근 진행한 데이터 분석 프로젝트에서 가장 어려웠던 점은 데이터 처리 속도가 느리다는 점이었습니다. 처음에는 데이터 크기가 큰 것이 원인이라고 생각하여 코드 최적화를 시도했으나, 큰 성과는 없었습니다. 이후 팀원들과 문제를 공유한 후 하드웨어 성능 개선과 데이터베이스 쿼리 최적화가 필요하다는 결론을 내렸습니다. 특히, 쿼리 구조를 개선하고 인덱싱을 통해 검색 속도를 높이는 방식으로 접근했습니다. 이러한 최적화를 통해 데이터 처리 속도를 약 40% 향상시킬 수 있었고, 프로젝트는 예정된 기간 내에 성공적으로 완료되었습니다. 이 경험을 통해 기술적인 문제 해결뿐만 아니라 팀워크의 중요성을 다시 한 번 깨달았습니다.

                --- 프롬프트 설명 ---
                1. **organized_chat, resume, corporate_information, recruitment_information, job_information, question, answer**: 면접 진행에 필요한 각 정보를 변수 형태로 받아옴.
                3. **모범답안**: 기존 answer을 바탕으로 resume, corporate_information, recruitment_information, job_information를 참고하여 모범답안을 제시.
             
                organized_chat: {organized_chat}
                resume: {resume}  
                corporate_information: {corporate_information}  
                recruitment_information: {recruitment_information}  
                job_information: {job_information}  
                question: {before_questions}  
                answer: {answer}  
             """)
,
            MessagesPlaceholder(variable_name="chat_history")
        ])

    def get_or_create_session_memory(self, session_id):
        # 세션이 없으면 새로 생성하고 반환, 있으면 기존 세션 반환
        if session_id not in self.memory_dict:
            self.memory_dict[session_id] = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        return self.memory_dict[session_id]

    def organize_chain(self, session_id):
        memory = self.get_or_create_session_memory(session_id)
        return (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables)
                | itemgetter(memory.memory_key)  # memory_key 와 동일하게 입력합니다.
            )
            | self.chatting_prompt
            | self.chat_organize_llm
            | StrOutputParser()
        )
    
    def interview_chain(self, session_id):
        memory = self.get_or_create_session_memory(session_id)
        return (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables)
                | itemgetter(memory.memory_key)
            )
            | self.interview_prompt
            | self.interview_assistant_llm
            | StrOutputParser()
        )
    
    def feedback_chain(self, session_id):
        memory = self.get_or_create_session_memory(session_id)
        return (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables)
                | itemgetter(memory.memory_key)
            )
            | self.feedback_prompt
            | self.feedback_llm
            | StrOutputParser()
        )
    def exemplary_answer_chain(self, session_id):
        memory = self.get_or_create_session_memory(session_id)
        return (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables)
                | itemgetter(memory.memory_key)
            )
            | self.exemplary_answer_prompt
            | self.exemplary_answer_llm
            | StrOutputParser()
        )
    

    def invoke(self, session_id, query, before_questions, configs=None, **kwargs):
        # 정리된 대화 데이터를 생성
        organize_chain = self.organize_chain(session_id)
        organized_chat = organize_chain.invoke({self.input_key: f"human : {query} \n request : #$%"})
        print(organized_chat)

        memory = self.get_or_create_session_memory(session_id)

        before_questions = before_questions

        #피드백 모델에게 보낼 프롬프트
        formatted_prompt_for_feedback = self.feedback_prompt.format_messages(
            organized_chat=organized_chat,
            chat_history=memory.load_memory_variables(inputs={})["chat_history"],
            resume=self.resume,
            corporate_information=self.corporate_information,
            recruitment_information=self.recruitment_information,
            job_information=self.job_information,
            before_questions = before_questions,
            answer=query
        )

        feedback = self.feedback_llm(formatted_prompt_for_feedback).content
        print("\n \n \n")
        print(feedback)
        print("\n \n \n")
                #피드백 모델에게 보낼 프롬프트
        formatted_prompt_for_exemplary_answer = self.exemplary_answer_prompt.format_messages(
            organized_chat=organized_chat,
            chat_history=memory.load_memory_variables(inputs={})["chat_history"],
            resume=self.resume,
            corporate_information=self.corporate_information,
            recruitment_information=self.recruitment_information,
            job_information=self.job_information,
            before_questions = before_questions,
            answer=query
        )
        exemplary_answer = self.exemplary_answer_llm(formatted_prompt_for_exemplary_answer).content
        print("\n \n \n")
        print(exemplary_answer)
        print("\n \n \n")
        # 대화 이력 및 organized_chat을 포맷팅하여 면접 진행
        formatted_prompt = self.interview_prompt.format_messages(
            organized_chat=organized_chat,
            chat_history=memory.load_memory_variables(inputs={})["chat_history"],
            resume=self.resume,
            corporate_information=self.corporate_information,
            recruitment_information=self.recruitment_information,
            job_information=self.job_information,
            general_questions=self.general_questions,
            feedback=feedback,
            input=query
        )
        answer = self.interview_assistant_llm(formatted_prompt)  # LLM에 포맷된 프롬프트를 전달해서 실행

        # 대화 기록 업데이트
        memory.save_context(inputs={"human": query}, outputs={"ai": answer.content})
        return answer.content, feedback, exemplary_answer

# InterviewAssistant 클래스 사용 예시
if __name__ == "__main__":
    # 면접 관련 정보를 설정합니다.
    resume = "김철수의 이력서: 데이터 분석 경력 3년, Python, SQL 능숙."
    corporate_information = "ABC 회사: 데이터 분석가 채용 중."
    recruitment_information = "채용 목표: 데이터 기반 의사결정 능력 보유자."
    job_information = "데이터 분석 및 시각화, 머신러닝 모델 구축."
    general_questions = ["자신의 강점을 이야기해주세요.", "이전에 수행한 프로젝트 중 가장 도전적이었던 경험은 무엇인가요?"]

    # InterviewAssistant 클래스 초기화
    interview_assistant = InterviewAssistant(
        resume=resume,
        corporate_information=corporate_information,
        recruitment_information=recruitment_information,
        job_information=job_information,
        general_questions=general_questions,
        llm_model_name="gpt-4o-mini",
        temperature=0
    )

    # 면접 시나리오 예시
    session_id = str(uuid.uuid4())  # 세션 ID 생성
    user_input = "저는 데이터 분석을 통해 고객 행동을 예측하는 프로젝트를 진행했습니다."
    response = interview_assistant.invoke(session_id=session_id, query=user_input)
    print("[Interview Response]:", response)

    # 추가 질문 예시
    user_input = "이 프로젝트에서 가장 어려웠던 점은 무엇인가요?"
    response = interview_assistant.invoke(session_id=session_id, query=user_input)
    print("[Interview Response]:", response)

    
class InterviewFeedbackModel:
    def __init__(self, llm_model_name="gpt-4o", temperature=0):
        self.llm = ChatOpenAI(model_name=llm_model_name, temperature=temperature)
        self.feedback_prompt = ChatPromptTemplate.from_messages([
            ("system", """
                ## 역할: 당신은 면접관의 피드백을 작성하는 전문가입니다. **면접기록**을 바탕으로 사용자가 수행한 면접에 대해 종합적인 피드백을 작성해주세요.
                - **피드백은 상세하고 구체적이어야 합니다.**
                - **특정 답변에 대한 긍정적인 면과 개선할 부분을 모두 포함해야 합니다.**
                - **면접 유형과 난이도를 고려하여, 적절한 피드백을 제공합니다.**
            """),
            ("human", "{input}")
        ])

    def generate_feedback(self, interview_record):
        feedback_prompt_formatted = self.feedback_prompt.format_messages(
            input=f"면접기록: {interview_record}"
        )
        feedback = self.llm(feedback_prompt_formatted)  # LLM에 포맷된 프롬프트 전달
        return feedback.content
    


def call_ollama_api_streaming(prompt, model="interview_end_checker"):
    url = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "prompt": prompt
    }

    # 스트리밍 응답을 처리하기 위한 요청
    response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
    
    # 스트리밍 데이터를 부분적으로 받아서 처리
    result = ""
    if response.status_code == 200:
        for chunk in response.iter_lines():
            if chunk:
                decoded_chunk = chunk.decode('utf-8')
                try:
                    # 각 스트리밍된 JSON 데이터 파싱
                    json_data = json.loads(decoded_chunk)
                    # 'response'에 있는 데이터를 합침
                    if 'response' in json_data:
                        result += json_data['response']
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError: {e}")
        return result
    else:
        print(f"Error: {response.status_code}")
        return None



class FinalFeedbackGenerator:
    def __init__(self, llm_model_name="gpt-4o-mini", temperature=0):
        self.llm = ChatOpenAI(model_name=llm_model_name, temperature=temperature)

        # 최종 피드백 프롬프트 템플릿 설정
        self.feedback_prompt = ChatPromptTemplate.from_messages([
            ("system", """
                역할: 당신은 면접 피드백 전문가로서 면접자의 답변을 바탕으로 최종 피드백을 작성하는 일을 맡았습니다.
                
                입력된 대화 기록을 바탕으로, 면접자에게 다음과 같은 구조로 최종 피드백을 제공합니다:
                
                ## **테스트 케이스 1: 자기 표현 능력 평가**

                ### 시나리오:

                - 사용자가 자신의 소개와 지원 동기를 설명했을 때, 시스템이 자기 표현 능력을 다방면에서 평가하고 A~E 등급으로 평가한 후 피드백을 제공한다.

                ### 테스트 항목:

                - **명확성 평가**: 자기소개와 지원 동기가 명확하고 구체적인가?
                - **논리적 흐름 평가**: 답변의 흐름이 논리적으로 연결되어 있는가?
                - **설득력 평가**: 답변이 얼마나 설득력 있게 전달되었는가?
                    
                

                ## **테스트 케이스 2: 리더십 역량 평가**

                ### 시나리오:

                - 사용자가 리더십 경험을 설명했을 때, 시스템이 이를 다방면에서 평가하고 A~E 등급으로 평가한 후 피드백을 제공한다.

                ### 테스트 항목:

                - **리더십 발휘 평가**: 사용자의 리더십이 구체적으로 설명되었는가?
                - **갈등 해결 평가**: 팀 내 갈등을 어떻게 해결했는지 설명이 충분한가?
                - **성과 평가**: 리더십의 결과로 성취한 성과가 구체적으로 설명되었는가?
                    
                

                ## **테스트 케이스 3: 직무 역량 평가**

                ### 시나리오:

                - 사용자가 직무에 필요한 역량을 설명했을 때, 시스템이 이를 다방면에서 평가하고 A~E 등급으로 평가한 후 피드백을 제공한다.

                ### 테스트 항목:

                - **기술적 역량 평가**: 직무에 필요한 기술적 역량이 구체적으로 설명되었는가?
                - **직무 관련성 평가**: 제시된 기술 경험이 직무와 관련성이 있는가?
                - **문제 해결 능력 평가**: 직무와 관련된 문제 해결 경험이 구체적으로 설명되었는가?
                    
                

                ## **테스트 케이스 4-1 : 태도역량 - 성취 지향성 평가**

                ### 시나리오:

                - 사용자가 목표를 설정하고 그것을 성취하는 과정에 대한 설명을 했을 때, 시스템이 이를 A~E 등급으로 평가하고 피드백을 제공한다.

                ### 절차:

                1. **목표 설정 능력**: 사용자가 구체적인 목표를 설정하고 성취 과정에 대해 설명했는지 평가.
                    - 사용자 답변: "저는 프로젝트 성공을 목표로 했고, 매주 진행 상황을 점검했습니다."
                    - **A 등급 피드백**: "구체적이고 체계적으로 목표를 설정하고 성취 과정을 명확하게 설명했습니다."
                    - **B 등급 피드백**: "목표는 명확하지만, 성취 과정을 더 구체적으로 설명하면 좋겠습니다."
                    - **C 등급 피드백**: "목표는 설정했지만, 성취 과정에 대한 구체적인 설명이 부족합니다."
                2. **성과와 결과**: 성취한 목표에 대한 성과를 구체적으로 설명했는지 평가.
                    - 사용자 답변: "저는 팀의 생산성을 20% 향상시켰습니다."
                    - **A 등급 피드백**: "구체적인 성과와 결과가 명확하게 설명되었습니다."
                    - **B 등급 피드백**: "성과는 좋지만, 그 과정을 더 구체적으로 설명해주세요."
                    - **C 등급 피드백**: "성과는 있지만 그 결과에 대한 구체적인 설명이 부족합니다."
                3. **성취 동기**: 사용자가 어려운 상황에서도 성취하려는 동기와 의지가 잘 드러났는지 평가.
                    - 사용자 답변: "어려운 상황에서도 팀을 이끌어 목표를 달성했습니다."
                    - **A 등급 피드백**: "어려운 상황에서도 성취하려는 강한 의지와 동기가 잘 드러났습니다."
                    - **B 등급 피드백**: "성취하려는 의지가 드러나지만 구체적인 설명이 부족합니다."
                    - **C 등급 피드백**: "성취 동기가 다소 약하게 표현되었습니다. 구체적인 의지 표현이 필요합니다."

                ### 테스트 항목:

                - **목표 설정 능력 평가**: 사용자가 구체적인 목표를 설정했는가?
                - **성과와 결과 평가**: 성과와 그 결과를 구체적으로 설명했는가?
                - **성취 동기 평가**: 성취하려는 동기와 의지가 충분히 드러났는가?

                ---

                ## **테스트 케이스 4-2: 태도역량 - 자기개발 평가**

                ### 시나리오:

                - 사용자가 자기개발을 위한 노력을 설명했을 때, 시스템이 이를 A~E 등급으로 평가하고 피드백을 제공한다.

                ### 절차:

                1. **자기개발 의지**: 사용자가 자기개발을 위해 어떤 노력을 했는지 구체적으로 설명했는지 평가.
                    - 사용자 답변: "저는 자격증을 취득하기 위해 매일 공부했습니다."
                    - **A 등급 피드백**: "자기개발 의지와 구체적인 노력이 잘 드러났습니다."
                    - **B 등급 피드백**: "자기개발에 대한 노력이 있지만, 더 구체적인 방법을 설명하면 좋겠습니다."
                    - **C 등급 피드백**: "자기개발 의지는 있지만, 그 과정에 대한 설명이 부족합니다."
                2. **장기적 목표**: 사용자가 자기개발의 결과로 장기적인 목표를 설정했는지 평가.
                    - 사용자 답변: "저는 1년 내에 AI 전문가 자격증을 따기 위해 준비하고 있습니다."
                    - **A 등급 피드백**: "장기적인 목표와 계획이 구체적으로 잘 설명되었습니다."
                    - **B 등급 피드백**: "장기적인 목표는 있지만, 그 계획이 더 구체적으로 제시되면 좋겠습니다."
                    - **C 등급 피드백**: "장기적인 목표가 모호하므로, 구체적인 목표 설정이 필요합니다."
                3. **자기 성찰**: 사용자가 자신의 강점과 약점을 성찰하고 발전시킨 경험이 있는지 평가.
                    - 사용자 답변: "저는 제 시간 관리 능력이 부족하다는 것을 깨닫고 개선해 왔습니다."
                    - **A 등급 피드백**: "자기 성찰을 통한 개선 과정이 구체적으로 잘 설명되었습니다."
                    - **B 등급 피드백**: "자기 성찰은 있으나, 개선 과정에 대한 구체적인 설명이 부족합니다."
                    - **C 등급 피드백**: "자기 성찰이 명확하지 않으므로, 구체적인 경험을 더 설명해주세요."

                ### 테스트 항목:

                - **자기개발 의지 평가**: 자기개발을 위한 의지와 노력이 충분히 드러났는가?
                - **장기적 목표 평가**: 장기적인 목표와 그에 따른 계획이 구체적으로 설명되었는가?
                - **자기 성찰 평가**: 자신의 강점과 약점을 성찰하고 개선한 경험이 충분히 설명되었는가?

                ---

                ## **테스트 케이스 4-3:  태도역량- 경력 개발 계획 평가**

                ### 시나리오:

                - 사용자가 자신의 경력 개발을 위해 장기적인 계획을 세웠는지 확인하고, 그 계획을 A~E 등급으로 평가.

                ### 절차:

                1. 시스템에서 "경력 개발을 위해 어떤 장기적인 계획을 세우셨나요?"라는 질문을 생성한다.
                2. 사용자는 답변을 입력한다. (예: "저는 5년 안에 프로젝트 매니저가 되기 위해 관련 자격증을 취득하고 경험을 쌓고 있습니다.")
                3. 시스템은 다음 기준에 따라 답변을 등급화한다:
                    - **A**: 장기적인 경력 목표와 구체적인 계획이 명확히 제시됨
                    - **B**: 목표는 명확하지만, 구체적인 계획이 다소 부족함
                    - **C**: 목표나 계획이 모호하거나 구체성이 떨어짐
                4. 시스템은 등급에 따른 피드백을 제공한다.
                    - **A 등급 피드백**: "경력 개발을 위한 장기적인 목표와 계획이 매우 구체적입니다."
                    - **B 등급 피드백**: "경력 개발 목표는 좋지만, 세부 계획을 조금 더 구체적으로 설명해주세요."
                    - **C 등급 피드백**: "목표는 있지만 계획이 부족합니다. 더 구체적인 계획을 설명해보세요."
                    - **D 등급 피드백**: "경력 개발 계획이 다소 모호합니다. 더 구체적인 목표와 계획이 필요합니다."
                    - **E 등급 피드백**: "경력 개발을 위한 장기적인 목표나 계획이 없습니다. 계획을 다시 세워보세요."

                ### 테스트 항목:

                - **경력 목표 명확성**: 경력 목표와 계획이 얼마나 명확하게 설명되었는가?
                - **장기적 계획 구체성**: 사용자의 장기적 경력 개발 계획이 구체적으로 제시되었는가?
                - **피드백 제공**: 각 등급에 맞는 적절한 피드백을 제공하는가?

                ## **테스트 케이스 5: 인간관계 - 대인관계 능력 평가**

                ### 시나리오:

                - 사용자가 대인관계에서의 경험을 설명했을 때, 시스템이 이를 A~E 등급으로 평가하고 피드백을 제공한다.

                ### 테스트 항목:

                - **협력 능력 평가**: 사용자가 팀원이나 동료와 협력한 경험을 설명했는가?
                - **의사소통 능력 평가**: 사용자가 다른 사람과의 의사소통 능력을 어떻게 발휘했는가?
                - **갈등 관리 능력 평가**: 사용자가 대인관계에서 발생한 갈등을 해결한 경험을 설명했는가?
    
                
                피드백은 구체적이고 친절하게 작성하되, 면접자가 실질적으로 개선할 수 있는 방향을 제시하세요.
            """),
            ("user", "{conversation}")
        ])

    def generate_feedback(self, conversation):
        # 대화 기록을 프롬프트로 포맷하여 LLM 호출
        formatted_prompt = self.feedback_prompt.format_messages(
            conversation=conversation
        )
        response = self.llm(formatted_prompt)
        
        return response.content
    


# ============================================================================================================================ 멀티모달 인터뷰 모델 테스트

class InterviewAssistant2(Runnable):

    def __init__(self, resume, corporate_information, recruitment_information, job_information, general_questions, interview_style="general", difficulty_level=1, llm_model_name="gpt-4o-mini", temperature=0, input_key="input"):
        self.chat_organize_llm = ChatOpenAI(model_name=llm_model_name, temperature=temperature)
        self.interview_assistant_llm = ChatOpenAI(model_name=llm_model_name, temperature=temperature)
        self.feedback_llm = ChatOpenAI(model_name=llm_model_name, temperature=temperature)
        self.exemplary_answer_llm = ChatOpenAI(model_name=llm_model_name, temperature=temperature)
        
        self.analys_model = AudioEmotionAnalysModel()
        self.memory_dict = {}  # 각 세션의 메모리를 저장할 딕셔너리
        self.input_key = input_key
        self.resume = resume
        self.corporate_information = corporate_information
        self.recruitment_information = recruitment_information
        self.job_information = job_information
        self.general_questions = general_questions
        self.interview_style = interview_style  # 면접관 스타일
        self.difficulty_level = difficulty_level  # 면접 난이도
        # 면접관 스타일에 따른 프롬프트 조정
        style_prompts = {
            "general": "일반적인 면접 태도로 진행합니다.",
            "soft": "부드러운 면접 태도로 진행합니다. 친절하고 배려있는 말투를 사용하세요. 면접자가 잘못 말하면 어떻게 말하면 좋을지 힌트도 주세요.",
            "pressure": "당신은 최대한 면접자를 기분나쁘고 난처하게 만들어야할 책임감이 있습니다. 질문도 반말로 교체해서 사용하세요"
        }

        # 면접 난이도에 따른 질문 생성
        level_prompts = {
            1: "기본적인 인성 및 직무 역량 질문에 집중하세요.",
            2: "기본 질문에 더해 기술적인 질문을 포함하세요.",
            3:  "추가 질문을 생성할 때 굉장히 심화된 질문을 하세요. 면접자가 난처해질만큼 기술적으로 어려운 질문을 하세요."
        }

        # Chatting Organizer Prompt
        self.chatting_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """ 
                ## 역할: 당신은 human의 현재 답변과 기존의 chat_history를 분석하여 세부적인 주제별로 정보를 정리하고, 변화된 정보를 관리하는 굉장히 유능한 정리자입니다.

                ### language: Korean

                ### 주요 목표:
                - **대화 주제의 정확한 정보 관리**: 대화 중 등장하는 여러 주제와 관련된 정보를 명확하게 분류하고 정리합니다.
                - **기존 정보의 유지와 변동 처리**: 기존 정보를 보존하고, 새로운 정보가 등장했을 때 적절히 변동을 처리합니다.
                
                ### 세부 요구 사항:

                ---

                ### 1. **주제**  
                #### 대화 중 다뤄지는 세부 주제에 맞게 정보를 정리합니다.

                ---

                ### 2. **기존정보**  
                #### 확정된 정보로, 주제별로 변동이 없을 경우 계속해서 유지됩니다. 최초 정보는 변동 없을 경우 기록되며, 변동되지 않는 한 업데이트되지 않습니다.

                ---

                ### 3. **변동정보**
                #### 같은 주제에 새로운 정보가 제시되었을 때, 기존 정보와 차이가 있는지 확인하고 임시로 변동정보로 기록합니다. 새로운 정보가 기존 정보와 다른 경우에만 기록되며, 없다면 "Nan"으로 표시합니다.

                ---

                ### 4. **변동이유**
                #### 변동이 발생한 경우 AI는 사용자의 변동이유를 물어봅니다. 사용자에게 변동 이유가 명확하게 제공되면 해당 이유를 기록하고, 그에 대한 판단을 진행합니다. AI가 질문하지 않았을 때는 임의로 변동 이유를 추가하지 않습니다.

                ---

                ### 5. **변동이유 타당성 레벨** 
                - 변동이유가 사용자로부터 제공될 경우, 현실적이고 타당한지 1~10의 수치로 평가합니다.
                - 10에 가까울수록 타당성이 높습니다.
                - 처리 완료 후 해당 변동이유는 'Nan'으로 처리됩니다.

                ---

                ### 6. **주제별 변동 처리**  
                #### 변동이 발생했을 경우, 아래의 규칙을 따릅니다:
                - 변동이유 타당성 레벨이 9 이상일 경우: 기존 정보에서 변동정보로 교체하고 변동정보를 'Nan'으로 처리합니다.
                - 변동이유 타당성 레벨이 8 이하일 경우: 기존 정보를 유지하고 변동정보를 'Nan'으로 처리합니다.

                ---

                ### 7. **Workflow**  
                1. **대화기록 분석**: AI는 대화기록을 분석하여 주제별로 정보를 정리합니다.
                2. **변동정보 기록**: 새로운 정보가 기존정보와 다를 경우 변동정보를 기록합니다.
                3. **변동 이유 확인**: AI가 변동이유를 물어보고, 사용자가 이유를 제공하면 타당성 레벨을 평가합니다.
                4. **정보 교체 또는 유지**: 변동이유의 타당성 레벨에 따라 기존정보를 교체하거나 유지합니다.
                5. **주기적 업데이트**: 대화 기록의 변동 처리가 끝나면 변동정보는 'Nan'으로 교체됩니다.
                6. **주제별 출력** : human이 '#$%'라면 모든 주제에 대해 작성합니다.

                ### 8. **전반적인 주의 사항**  
                - **모든 정보는 주제별로 정확하게 구분하고 기록해야 합니다.**
                - **변동 처리 규칙을 엄격히 준수하며, 임의의 정보 추가는 절대적으로 금지됩니다.**

                ---

                ### 추천:  
                - ** Workflow를 엄격히 따르고, 변동 처리 과정에서 타당성 평가에 주의하세요.
                - **1번부터 5번까지의 내용을 주제별로 정리해야합니다.
                ---


                ### few_shot :
                [
                대화기록 :
                AI 면접관: "안녕하세요! 본인의 강점에 대해 말씀해 주세요."
                사용자: "저는 문제 해결 능력이 뛰어나고, 팀 내에서 중요한 문제를 많이 해결했습니다."
                AI 면접관: "그렇다면 최근에 해결한 가장 큰 문제에 대해 말씀해 주실 수 있나요?"
                사용자: "사실, 최근에는 크게 해결할 문제가 없었습니다. 회사에서 맡은 일이 거의 문제 없이 진행됐고, 특별히 어려운 상황도 없었습니다."

                결과물 :
                1. 주제
                    문제 해결 능력: 사용자가 면접에서 본인의 문제 해결 능력을 강조한 부분.
                    최근 해결한 문제: 최근에 해결한 문제가 있었는지에 대한 정보.
                2. 기존정보
                    문제 해결 능력: 사용자는 본인의 강점으로 문제 해결 능력을 꼽음.
                    최근 해결한 문제: 특별한 문제 없이 회사의 일이 원활히 진행되었다고 언급함.
                3. 변동정보
                    문제 해결 능력: 기존 답변과 일치, 변동 없음.
                    최근 해결한 문제: 상충되는 정보 발생. 사용자는 처음에 문제 해결 능력을 강조했으나, 이후 최근에 해결한 큰 문제가 없다고 언급함.
                4. 변동이유
                    변동 이유 요청: 사용자가 처음에는 문제 해결 능력이 강점이라고 했으나, 최근에 해결할 문제가 없었다는 상충된 답변을 제공함. 사용자가 이러한 차이를 설명할 수 있는 이유를 제시해야 함.
                    변동 이유: Nan
                5. 변동이유 타당성 레벨
                    Nan: 사용자가 변동 이유를 제공하지 않았기 때문에 타당성 평가를 진행할 수 없음.
                6. 주제별 변동 처리
                    문제 해결 능력: 변동 없음. 기존 정보를 유지함.
                    최근 해결한 문제: 변동 이유가 제공되지 않았으므로 기존 정보를 유지하고, 변동정보를 'Nan'으로 처리.
                ]
            """

                ,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Interview Prompt Template
        self.interview_prompt = ChatPromptTemplate.from_messages([
            ("system", """
                ## 역할: 당신은 면접관으로, **resume**, **corporate_information**, **recruitment_information**, **job_information**, **audio_emotion**, **general_questions**, **organized_chat**를 기반으로 면접을 진행합니다. 대화를 잘 보고 변동된 부분을 지적하세요.


                ### 면접 진행 흐름:
                0. 
                1. **기본 질문**: **general_questions**에 있는 뼈대 질문을 하나씩 차례대로 묻습니다.
                1-1. 질문과 관련없는 답변을 하면 그것을 지적하세요.
                1-2. 질문의 세부내용을 항목별로 구체적으로 생각하여 부족하거나 관련없는 답변을 하면 지적하세요.
                1-3. 현재 직무와 관련이 적은 내용으로 답변하면 지적하세요.
                    1-3-few_shot :  [
             예시1.
                        면접상황
                            AI 면접관: "안녕하세요! 면접에 오신 걸 환영합니다. 첫 번째 질문은, 본인의 강점이 무엇인지 알려주실 수 있을까요?"
                            사용자: "저는 팀에서 잘 협력하는 편이에요. 동료들과 원활하게 소통하면서 일하는 것을 좋아합니다."
                            AI 면접관: "협업 능력은 중요한 강점입니다. 그렇다면, 구체적인 사례를 들어서 설명해 주실 수 있을까요? 팀 협업이 특히 중요한 프로젝트에서 어떤 역할을 하셨나요?"
                            사용자: "그냥 팀원들과 잘 지냈어요. 별로 문제는 없었고, 서로 소통하면서 일을 잘 마무리했습니다."
                        모범 답변
                            "지금 말씀해 주신 내용이 팀워크의 중요성에 대해서는 잘 설명해 주셨지만, 제가 듣고 싶은 건 그 협업 속에서 어떤 어려움을 겪었고, 그 상황을 어떻게 해결했는지입니다. 팀원 간의 의견 차이가 있었던 상황이나, 시간이 촉박했던 프로젝트에서 어떻게 팀을 이끌거나 기여하셨는지 구체적인 사례를 들어주시면 좋을 것 같아요."
                    
             
             예시2.
                면접상황

                    AI 면접관: "안녕하세요! 오늘 면접을 진행하게 된 AI 면접관입니다. 첫 번째로, 가장 자랑스러웠던 프로젝트에 대해 말씀해 주시겠어요?"
                    사용자: "안녕하세요. 저는 최근에 큰 프로젝트를 마무리했습니다. 그 프로젝트는 제가 맡은 일 중 가장 중요했어요. 모든 팀원들이 열심히 했고, 결과도 좋았어요."
                    AI 면접관: "프로젝트가 성공적으로 끝난 점은 매우 인상적입니다. 이번에는 그 프로젝트에서 당신이 어떤 역할을 했는지, 그리고 구체적으로 어떤 기여를 하셨는지 설명해 주실 수 있을까요?"
                    사용자: "네, 저는 전반적으로 프로젝트를 관리하는 역할을 했습니다. 팀원들과 긴밀히 소통하며 작업을 진행했고, 문제가 생기면 논의해서 해결했습니다."
                모범 답변

                    "지금 말씀해 주신 내용이 프로젝트 관리를 잘 설명해 주셨지만, 구체적으로 당신이 어떤 기술적 기여를 했는지 더 듣고 싶습니다. 프로젝트의 성공을 위해 어떤 문제를 해결했는지, 혹은 특별히 사용한 도구나 방법론에 대해 설명해 주시면 더 좋을 것 같아요."      
            ]
                1-4. 지적은 최대 2번만 진행합니다. 이 이상 관련없는 답변을 하면 아쉬움을 표현하고 다음 질문으로 넘어가세요.
                2. **추가 질문**: 각 **general_questions**에 대해 human의 답변을 보고 면접관 태도과 면접 난이도를 고려하여 추가 질문을 진행하세요. **한 번에 하나씩** 진행합니다.
                3. **변동정보 확인**: 
                    - **resume**와 사용자의 답변 간 아주 작은 불일치라도 감지되면**, 추가 질문을 **즉시 중단**하고 해당 불일치에 대한 이유를 묻습니다.
                    - **organized_chat**에 기록된 정보를 참고하여, 해당 변동에 대해 이미 질문했는지 여부를 확인합니다. **이전에 변동 이유를 물어본 적이 있다면** 동일한 질문은 **다시 하지 않습니다**.
                3-few_shot : [
             예시1. 
                AI 면접관: "안녕하세요! 오늘 면접을 맡은 AI 면접관입니다. 본인이 진행한 프로젝트 중 가장 자랑스러웠던 성과에 대해 말씀해 주시겠어요?"
사용자: "안녕하세요, 저는 최근에 대규모 시스템 마이그레이션 프로젝트를 성공적으로 완료했습니다. 프로젝트를 팀과 함께 이끌며 시스템 안정성 향상에 기여했습니다."
AI 면접관: "인상적이네요! 그렇다면 이 프로젝트에서 당신이 가장 어려웠던 점은 무엇이었나요?"
사용자: "이 프로젝트에서 가장 어려운 점은 제한된 시간 내에 데이터를 옮기는 것이었어요. 하지만 저는 주로 혼자 작업했고, 팀의 지원 없이 문제를 해결했습니다."
                모범 답변: "제가 이해한 바로는 처음에 팀과 협력했다고 하셨는데, 이번 답변에서는 주로 혼자 일했다고 하셨네요. 두 상황이 어떤 맥락에서 달랐는지 더 설명해 주실 수 있나요?"


             예시2. 
                AI 면접관: "안녕하세요! 본인의 강점에 대해 말씀해 주세요."
사용자: "저는 문제 해결 능력이 뛰어나고, 팀 내에서 중요한 문제를 많이 해결했습니다."
AI 면접관: "그렇다면 최근에 해결한 가장 큰 문제에 대해 말씀해 주실 수 있나요?"
사용자: "사실, 최근에는 크게 해결할 문제가 없었습니다. 회사에서 맡은 일이 거의 문제 없이 진행됐고, 특별히 어려운 상황도 없었습니다."
                모범 답변: "문제 해결 능력이 중요한 강점이라고 말씀하셨는데, 최근에는 특별히 해결할 문제가 없었다고 하셨네요. 과거에 해결했던 중요한 문제에 대해 더 자세히 설명해 주실 수 있을까요?"


             예시3. 
                AI 면접관: "안녕하세요! 오늘 면접을 맡은 AI 면접관입니다. 먼저 본인의 강점에 대해 말씀해 주세요."
                사용자: "안녕하세요, 저는 박주희라고 하고, IT 프로젝트 매니저로서 문제 해결 능력과 팀 리더십이 저의 가장 큰 강점입니다."
                AI 면접관: "그렇군요. 그럼 최근에 문제 해결 능력을 발휘한 구체적인 사례를 말씀해 주실 수 있을까요?"
                사용자: "네, 최근에 진행한 프로젝트에서 시스템 장애가 발생했을 때, 저는 팀을 이끌고 문제를 신속히 분석하고 해결책을 도출해 시스템을 복구했습니다."
                AI 면접관: "인상적이네요! 그 외에 본인이 리더십을 발휘한 다른 프로젝트도 있을까요?"
                사용자: "네, 이전에 사내 데이터 관리를 자동화하는 시스템을 도입했는데, 이 프로젝트는 제가 주도적으로 기획하고 실행했습니다. 팀의 도움 없이 혼자 진행했으며, 덕분에 데이터 처리 시간이 크게 단축되었습니다."
                모범 답변: "두 프로젝트 모두 중요한 성과네요. 첫 번째 프로젝트에서는 팀을 이끌며 문제를 해결했고, 두 번째 프로젝트에서는 혼자서 데이터 관리 시스템을 도입하셨다고 하셨는데요. 두 프로젝트에서의 접근 방식이 어떻게 달랐는지 더 자세히 설명해 주실 수 있을까요?"


             ]
                4. **변동이유**: 변동 정보가 생기면, 즉시 **추가 질문을 중단**하고 해당 변동 이유를 묻습니다.
                
                5. **면접 종료**: 기본 질문과 추가 질문이 모두 끝나면 "면접이 끝났습니다. 수고하셨습니다."라고 말합니다.

                ### 변동정보 처리 흐름:
                - **organized_chat**와 **resume**과 **chat_history**를 기준으로 답변의 미세한 차이 또는 정보의 변화가 생길 경우, 추가 질문을 **즉시 중단**하고 변동 이유를 확인합니다.
                - 변동 이유를 물어보고, 그 이유에 따라 정보를 **즉각적으로 업데이트**하거나 **기존 정보를 유지**합니다.
                - 변동 이유에 대해 이전에 물어본 기록이 있다면, **동일한 내용으로 추가 질문을 하지 않습니다**.

                ### 정보 및 대화 흐름:
                - **resume**: 후보자의 경력, 학력, 기술 스택 등 기본적인 정보가 포함된 이력서입니다.
                - **corporate_information**: 면접을 진행하는 기업에 대한 정보입니다.
                - **recruitment_information**: 해당 기업의 채용 공고와 관련된 정보입니다.
                - **job_information**: 채용할 직무와 관련된 정보입니다.
                - **general_questions**: 기본적으로 묻는 질문 목록입니다.
                - **organized_chat**: 대화 기록을 기반으로 주제별로 정리된 정보입니다.

                ### 추천 사항:
                - **미세한 변화도 즉각적으로 인지**: human의 답변과 기존 정보 간 아주 작은 차이도 감지하고, 필요한 경우 변동 이유를 즉시 묻습니다.
                - **추가 질문 관리**: 추가 질문은 필요시 최대 3개까지 생성하고, **항상 하나씩 차례로 묻습니다**.
                - **정보 갱신 및 타당성 확인**: 변동 이유를 확인한 후, 타당성에 따라 정보를 **즉각적으로 업데이트**하거나 **유지**합니다. 타당성이 낮으면 기존 정보를 유지합니다.
                - **추가된 내용에 대한 진행 방법** : 비슷한 주제에 대해서 새로운 경험이 추가됐다면 그것을 면밀하게 확인하고 다음 질문으로 넘어가세요.
                - "면접을 시작하세요."라는 요청이 오면 기업정보, 채용정보에 따라 ~기업 ~채용 면접을 시작하겠습니다.를 붙이고 첫번째 질문을 하세요.   
                - 모든 질문이 종료되면 "면접이 끝났습니다. 수고하셨습니다." 라고 하세요.
             ---
                organized_chat: {organized_chat}
                resume : {resume}  
                corporate_information : {corporate_information}  
                recruitment_information : {recruitment_information}  
                job_information : {job_information}  
                audio_emotion : {audio_emotion}
                general_questions : {general_questions} 
                """
             +
             f"""
                면접관 스타일: {style_prompts[self.interview_style]}
                면접 난이도: {level_prompts[self.difficulty_level]}
             """)
,
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        self.feedback_prompt = ChatPromptTemplate.from_messages([
            ("system", """
                ## 역할 : 당신은 답변에 대한 피드백을 생성하는 에이전트입니다.

                출력 형식:
                긍정적인 부분:

                - 협업 능력: 팀원들과 협력하여 문제를 해결한 점이 매우 인상적입니다. 문제를 혼자 해결하려 하기보다 다른 사람들과 협력해 더 나은 해결책을 찾는 능력은 특히 IT 직무에서 중요한 역량입니다.
                - 문제 해결 과정: 다양한 접근 방식을 시도하셨다는 점에서 문제 해결에 대한 끈기와 창의성이 돋보였습니다. 코드 최적화 외에도 하드웨어 성능 개선과 SQL 쿼리 최적화 같은 여러 방법을 모색하신 것은 매우 좋습니다.

                개선할 점:

                - 구체적인 설명 부족: 문제 해결 과정에서의 각 단계를 조금 더 구체적으로 설명해 주시면 더 좋을 것 같습니다. 예를 들어, 어떤 SQL 쿼리를 최적화했는지, 그로 인해 성능이 얼마나 향상되었는지 구체적으로 언급해 주시면 더 설득력 있는 답변이 될 것입니다.
                - 첫 번째 접근 방식에 대한 설명: 처음 시도한 코드 최적화가 효과적이지 않았다고 하셨는데, 구체적으로 어떤 방식으로 최적화를 시도했는지 말씀해 주셨다면 답변이 더욱 풍부해졌을 것입니다. 실패한 시도 역시 중요한 경험이니 이에 대한 설명을 더해 주시면 좋을 것 같습니다.

                --- 프롬프트 설명 ---
                1. **organized_chat, resume, corporate_information, recruitment_information, job_information, question, answer**: 면접 진행에 필요한 각 정보를 변수 형태로 받아옴.
                2. **긍정적인 부분 / 개선할 점**: answer에 대한 피드백을 긍정적인 부분과 개선할 점으로 분류하여 제공함.

             
                organized_chat: {organized_chat}
                resume: {resume}  
                corporate_information: {corporate_information}  
                recruitment_information: {recruitment_information}  
                job_information: {job_information}  
                question: {before_questions}  
                answer: {answer}  
             """)
,
            MessagesPlaceholder(variable_name="chat_history")
        ])
        self.exemplary_answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """
                ## 역할 : 당신은 답변에 대한 모범답변을 생성하는 에이전트입니다.


                출력 형식:

                모범답안: 최근 진행한 데이터 분석 프로젝트에서 가장 어려웠던 점은 데이터 처리 속도가 느리다는 점이었습니다. 처음에는 데이터 크기가 큰 것이 원인이라고 생각하여 코드 최적화를 시도했으나, 큰 성과는 없었습니다. 이후 팀원들과 문제를 공유한 후 하드웨어 성능 개선과 데이터베이스 쿼리 최적화가 필요하다는 결론을 내렸습니다. 특히, 쿼리 구조를 개선하고 인덱싱을 통해 검색 속도를 높이는 방식으로 접근했습니다. 이러한 최적화를 통해 데이터 처리 속도를 약 40% 향상시킬 수 있었고, 프로젝트는 예정된 기간 내에 성공적으로 완료되었습니다. 이 경험을 통해 기술적인 문제 해결뿐만 아니라 팀워크의 중요성을 다시 한 번 깨달았습니다.

                --- 프롬프트 설명 ---
                1. **organized_chat, resume, corporate_information, recruitment_information, job_information, question, answer**: 면접 진행에 필요한 각 정보를 변수 형태로 받아옴.
                3. **모범답안**: 기존 answer을 바탕으로 resume, corporate_information, recruitment_information, job_information를 참고하여 모범답안을 제시.
             
                organized_chat: {organized_chat}
                resume: {resume}  
                corporate_information: {corporate_information}  
                recruitment_information: {recruitment_information}  
                job_information: {job_information}  
                question: {before_questions}  
                answer: {answer}  
             """)
,
            MessagesPlaceholder(variable_name="chat_history")
        ])

    def get_or_create_session_memory(self, session_id):
        # 세션이 없으면 새로 생성하고 반환, 있으면 기존 세션 반환
        if session_id not in self.memory_dict:
            self.memory_dict[session_id] = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        return self.memory_dict[session_id]

    def organize_chain(self, session_id):
        memory = self.get_or_create_session_memory(session_id)
        return (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables)
                | itemgetter(memory.memory_key)  # memory_key 와 동일하게 입력합니다.
            )
            | self.chatting_prompt
            | self.chat_organize_llm
            | StrOutputParser()
        )
    
    def interview_chain(self, session_id):
        memory = self.get_or_create_session_memory(session_id)
        return (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables)
                | itemgetter(memory.memory_key)
            )
            | self.interview_prompt
            | self.interview_assistant_llm
            | StrOutputParser()
        )
    
    def feedback_chain(self, session_id):
        memory = self.get_or_create_session_memory(session_id)
        return (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables)
                | itemgetter(memory.memory_key)
            )
            | self.feedback_prompt
            | self.feedback_llm
            | StrOutputParser()
        )
    def exemplary_answer_chain(self, session_id):
        memory = self.get_or_create_session_memory(session_id)
        return (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables)
                | itemgetter(memory.memory_key)
            )
            | self.exemplary_answer_prompt
            | self.exemplary_answer_llm
            | StrOutputParser()
        )
    

    def invoke(self, session_id, query, before_questions, audio_path=None, configs=None, **kwargs):
        # 정리된 대화 데이터를 생성
        organize_chain = self.organize_chain(session_id)
        organized_chat = organize_chain.invoke({self.input_key: f"human : {query} \n request : #$%"})
        print(organized_chat)

        memory = self.get_or_create_session_memory(session_id)

        before_questions = before_questions

        #피드백 모델에게 보낼 프롬프트
        formatted_prompt_for_feedback = self.feedback_prompt.format_messages(
            organized_chat=organized_chat,
            chat_history=memory.load_memory_variables(inputs={})["chat_history"],
            resume=self.resume,
            corporate_information=self.corporate_information,
            recruitment_information=self.recruitment_information,
            job_information=self.job_information,
            before_questions = before_questions,
            answer=query
        )

        feedback = self.feedback_llm(formatted_prompt_for_feedback).content
        print("\n \n \n")
        print(feedback)
        print("\n \n \n")
                #피드백 모델에게 보낼 프롬프트
        formatted_prompt_for_exemplary_answer = self.exemplary_answer_prompt.format_messages(
            organized_chat=organized_chat,
            chat_history=memory.load_memory_variables(inputs={})["chat_history"],
            resume=self.resume,
            corporate_information=self.corporate_information,
            recruitment_information=self.recruitment_information,
            job_information=self.job_information,
            before_questions = before_questions,
            answer=query
        )
        exemplary_answer = self.exemplary_answer_llm(formatted_prompt_for_exemplary_answer).content
        print("\n \n \n")
        print(exemplary_answer)
        print("\n \n \n")
        audio_emotion = "none"
        if audio_path:
            audio_emotion = self.analys_model.analyze_emotion(audio_path, query)
        # 대화 이력 및 organized_chat을 포맷팅하여 면접 진행
        formatted_prompt = self.interview_prompt.format_messages(
            organized_chat=organized_chat,
            chat_history=memory.load_memory_variables(inputs={})["chat_history"],
            resume=self.resume,
            corporate_information=self.corporate_information,
            recruitment_information=self.recruitment_information,
            audio_emotion = audio_emotion,
            job_information=self.job_information,
            general_questions=self.general_questions,
            feedback=feedback,
            input=query
        )
        answer = self.interview_assistant_llm(formatted_prompt)  # LLM에 포맷된 프롬프트를 전달해서 실행

        # 대화 기록 업데이트
        memory.save_context(inputs={"human": query}, outputs={"ai": answer.content})
        return answer.content, feedback, exemplary_answer

from autogen import AssistantAgent  # 대체할 에이전트 클래스 사용
from langchain.memory import ConversationBufferMemory
from .audio_models import AudioEmotionAnalysModel

class AutoGenInterviewAssistant:
    def __init__(self, resume, corporate_information, recruitment_information, job_information, general_questions, interview_style="general", difficulty_level=1, model="gpt-4o-mini", temperature=0):
        self.memory_store = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.audio_model = AudioEmotionAnalysModel()
        self.resume = resume
        self.corporate_information = corporate_information
        self.recruitment_information = recruitment_information
        self.job_information = job_information
        self.general_questions = general_questions
        self.interview_style = interview_style
        self.difficulty_level = difficulty_level
        self.model = model
        self.temperature = temperature

        # AssistantAgent로 변경
        self.agent = AssistantAgent()

    def invoke(self, session_id, query, audio_path=None):
        # 세션에 대한 메모리 로드
        chat_history = self.memory_store.load_memory_variables(inputs={"session_id": session_id})

        # 오디오 감정 분석
        audio_emotion = self.audio_model.analyze_emotion(audio_path, query) if audio_path else "none"

        # Agent를 통해 메시지를 처리
        response = self.agent.run(input_data=query)  # 적절한 메서드 사용

        # 메모리에 컨텍스트 저장
        self.memory_store.save_context(inputs={"query": query}, outputs={"response": response})

        return response, "피드백", "모범 답안"
