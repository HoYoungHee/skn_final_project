from operator import itemgetter
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, Runnable
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()
class GeneralQuestionsGenerator(Runnable):
    def __init__(self, resume, corporate_information, recruitment_information, job_information, difficulty_level, llm_model_name="gpt-4o-mini", temperature=0):
        self.llm = ChatOpenAI(model_name=llm_model_name, temperature=temperature)
        self.resume_content = resume
        self.corporate_content = corporate_information
        self.recruitment_content = recruitment_information
        self.job_content = job_information
        self.difficulty_level = difficulty_level  # 면접 난이도
        
        level_prompts = {
            1: "기본적인 인성 및 직무 역량 질문에 집중하세요.",
            2: "기본 질문에 더해 기술적인 질문을 포함하세요.",
            3: "진짜 면접자가 기분나쁠만큼 이건 모르겠지? 싶은 것들만 질문하세요."
        }
        # General Questions Prompt Template
        self.general_questions_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
                면접 난이도: {level_prompts[self.difficulty_level]}"""+
             """
                당신은 면접관입니다. 이력서, 기업 정보, 채용 정보, 직무 정보를 바탕으로 지원자에게 물어볼 면접 질문을 생성하세요.
                질문은 지원자의 경험과 역량을 파악하는 데 도움이 되어야 하며, 구체적이고 직무와 관련이 있어야 합니다.
                이력서 내용: {resume_content}
                기업 정보: {corporate_content}
                채용 정보: {recruitment_content}
                직무 정보: {job_content}
            """),
            ("human", "일반 면접 질문을 생성해주세요.")
        ])

    def invoke(self):
        formatted_prompt = self.general_questions_prompt.format_messages(
            resume_content=self.resume_content,
            corporate_content=self.corporate_content,
            recruitment_content=self.recruitment_content,
            job_content=self.job_content
        )
        response = self.llm(formatted_prompt)
        print(response.content)
        return response.content

# InterviewAssistant 클래스 사용 예시
if __name__ == "__main__":
    # 면접 질문 생성 모델 초기화
    resume_content = "김철수의 이력서: 데이터 분석 경력 3년, Python, SQL 능숙."
    corporate_content = "ABC 회사: 데이터 분석가 채용 중."
    recruitment_content = "채용 목표: 데이터 기반 의사결정 능력 보유자."
    job_content = "데이터 분석 및 시각화, 머신러닝 모델 구축."

    general_questions_generator = GeneralQuestionsGenerator(
        resume_content=resume_content,
        corporate_content=corporate_content,
        recruitment_content=recruitment_content,
        job_content=job_content,
        llm_model_name="gpt-4o-mini",
        temperature=0
    )

    # 일반 면접 질문 생성
    general_questions = general_questions_generator.generate_general_questions()
    print("[Generated General Questions]:", general_questions)