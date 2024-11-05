from typing_extensions import Unpack
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
import uuid
from interview_model.interview_assistant_model import InterviewAssistant, call_ollama_api_streaming, FinalFeedbackGenerator, InterviewAssistant2, AutoGenInterviewAssistant
from interview_model.general_questions_generator_model import GeneralQuestionsGenerator  # 질문 생성 모델
from models.db_models import *  # DB 모델
from routers.db_router import *  # DB에서 데이터를 가져오는 함수

router = APIRouter()
feedback_model = FinalFeedbackGenerator(llm_model_name="gpt-4o-mini")


class InterviewRequest(BaseModel):
    user_id: str  # 사용자 ID
    resume_id: str
    corporate_id: str
    job_id: str
    interview_style: str  # 면접관 스타일 (일반, 부드러운, 압박)
    difficulty_level: int  # 면접 난이도 (1~3)

class AnswerRequest(BaseModel):
    user_id: str  # 사용자 ID
    user_answer: str

class UserInUse():
    def __init__(self,user_id,interview_assistant,question_id_in_use,interview_id) :
        self.user_id = user_id  # 사용자 ID
        self.interview_assistant = interview_assistant
        self.question_id_in_use = question_id_in_use
        self.interview_id = interview_id


interview_sessions = {}  # 사용자 ID 별로 면접 모델을 관리하는 dict
@router.post("/create_interview/")
async def create_interview(request: InterviewRequest):
    global interview_sessions
    try:
        # MySQL에서 이력서, 기업정보, 채용정보, 직무정보 받아오기
        resume = await get_data("이력서", request.resume_id)
        corporate_information = await get_data("기업정보", request.corporate_id)
        job_information,recruitment_information = await get_data("직무정보", request.job_id)

        # GeneralQuestionsGenerator 객체를 생성하고, 질문을 생성 (면접 난이도 추가)
        questions_generator = GeneralQuestionsGenerator(
            resume=resume,
            corporate_information=corporate_information,
            recruitment_information=recruitment_information,
            job_information=job_information,
            difficulty_level=request.difficulty_level  # 면접 난이도 전달
        )
        general_questions = questions_generator.invoke()

        # InterviewAssistant 객체를 생성하여 면접 모델 초기화 (면접관 스타일 및 난이도 추가)
        interview_assistant = InterviewAssistant(
            resume=resume,
            corporate_information=corporate_information,
            recruitment_information=recruitment_information,
            job_information=job_information,
            general_questions=general_questions,
            interview_style=request.interview_style,  # 면접관 스타일
            difficulty_level=request.difficulty_level  # 면접 난이도
        )


        # 면접 시작 메시지를 보내고 결과를 반환
        response, _, _ = interview_assistant.invoke(request.user_id, "면접을 시작하세요","")
        print(response)
        interview_id = save_interview_to_db(request)
        question_id = save_question_to_db(response, interview_id)
        user = UserInUse(request.user_id, interview_assistant, question_id, interview_id)
        # 사용자 ID를 세션 ID로 사용하여 면접 모델을 관리
        interview_sessions[request.user_id] = user
        return {"session_id": request.user_id, "message": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating interview: {str(e)}")

import asyncio

@router.post("/answer/")
async def answer_interview(request: AnswerRequest):
    global interview_sessions
    try:
        # 사용자 ID를 사용하여 해당 면접 모델을 불러옴
        interview_assistant = interview_sessions.get(request.user_id).interview_assistant
        if not interview_assistant:
            raise HTTPException(status_code=404, detail="Interview session not found")

        # 사용자의 답변을 모델에 입력하고 결과를 반환
        question_id_in_use = interview_sessions.get(request.user_id).question_id_in_use
        interview_id = interview_sessions.get(request.user_id).interview_id
        response, feedback, exemplary_answer = interview_assistant.invoke(request.user_id, request.user_answer, question_id_in_use)
        
        # 데이터베이스에 경로 정보 저장
        update_feedback_to_db(request.user_answer, feedback, exemplary_answer, question_id)
        question_id = save_question_to_db(response, interview_id)
        interview_sessions[request.user_id].question_id_in_use = question_id

        end_chack = call_ollama_api_streaming(response)
        print(end_chack)

        # 먼저 응답을 반환
        if end_chack == '종료' or request.user_answer == '차라리 날 죽여라!':
            # 비동기 작업을 실행하는 코드를 추가
            asyncio.create_task(handle_end_session(request.user_id, request.user_answer, question_id))
            if request.user_answer == '차라리 날 죽여라!':
                return {"message": ['죽어라!!', 'end']}
            return {"message": [response, 'end']}

        return {"message": [response, 'run']}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing answer: {str(e)}")


async def handle_end_session(user_id, user_answer, question_id):
    """비동기적으로 세션 종료 작업을 처리하는 함수"""
    # 데이터베이스 업데이트 및 세션 제거
    update_feedback_to_db(user_answer, 'end', 'end', question_id)
    update_final_feedback_to_db(user_id)
    interview_sessions.pop(user_id)



from whisper import load_model  # whisper 라이브러리를 사용해 모델을 불러옵니다
from pydantic import BaseModel
import tempfile

# Whisper Large 모델 로드
model = load_model("large")

@router.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # 받은 파일을 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Whisper 모델로 음성 파일을 변환
        result = model.transcribe(temp_file_path)

        # 임시 파일 삭제
        os.remove(temp_file_path)

        return {"transcription": result["text"]}
    except Exception as e:
        return {"error": str(e)}


interview_sessions2 = {}  # 사용자 ID 별로 면접 모델을 관리하는 dict
@router.post("/create_interview2/")
async def create_interview2(request: InterviewRequest):
    try:
        # MySQL에서 이력서, 기업정보, 채용정보, 직무정보 받아오기
        resume = await get_data("이력서", request.resume_id)
        corporate_information = await get_data("기업정보", request.corporate_id)
        job_information,recruitment_information = await get_data("직무정보", request.job_id)

        # GeneralQuestionsGenerator 객체를 생성하고, 질문을 생성 (면접 난이도 추가)
        questions_generator = GeneralQuestionsGenerator(
            resume=resume,
            corporate_information=corporate_information,
            recruitment_information=recruitment_information,
            job_information=job_information,
            difficulty_level=request.difficulty_level  # 면접 난이도 전달
        )
        general_questions = questions_generator.invoke()

        # InterviewAssistant 객체를 생성하여 면접 모델 초기화 (면접관 스타일 및 난이도 추가)
        interview_assistant = InterviewAssistant2(
            resume=resume,
            corporate_information=corporate_information,
            recruitment_information=recruitment_information,
            job_information=job_information,
            general_questions=general_questions,
            interview_style=request.interview_style,  # 면접관 스타일
            difficulty_level=request.difficulty_level  # 면접 난이도
        )


        # 면접 시작 메시지를 보내고 결과를 반환
        response, _, _ = interview_assistant.invoke(request.user_id, "면접을 시작하세요","")
        print(response)
        interview_id = save_interview_to_db(request)
        question_id = save_question_to_db(response, interview_id)
        user = UserInUse(request.user_id, interview_assistant, question_id, interview_id)
        # 사용자 ID를 세션 ID로 사용하여 면접 모델을 관리
        interview_sessions2[request.user_id] = user
        return {"session_id": request.user_id, "message": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating interview: {str(e)}")


@router.post("/answer2/")
async def answer_interview2(user_id: str, file: UploadFile = File(...)):
    try:
        # 사용자 ID를 사용하여 해당 면접 모델을 불러옴
        interview_assistant = interview_sessions2.get(user_id).interview_assistant
        if not interview_assistant:
            raise HTTPException(status_code=404, detail="Interview session not found")

        # 사용자의 답변을 모델에 입력하고 결과를 반환
        question_id_in_use = interview_sessions2.get(user_id).question_id_in_use
        interview_id = interview_sessions2.get(user_id).interview_id

        try:
            # 받은 파일을 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(await file.read())
                temp_file_path = temp_file.name

            # Whisper 모델로 음성 파일을 변환
            user_answer =  model.transcribe(temp_file_path)["text"]

            

        except Exception as e:
            return {"error": str(e)}
        print(f"위스퍼 답변인식 : {user_answer}")
        response,feedback,exemplary_answer = interview_assistant.invoke(user_id, user_answer, question_id_in_use, temp_file_path)
        # 임시 파일 삭제 (인터뷰 assistant가 wav를 받아야해서 이때 지움)
        os.remove(temp_file_path)
        # 데이터베이스에 경로 정보 저장
        update_feedback_to_db(user_answer, feedback,exemplary_answer,  question_id_in_use)
        question_id = save_question_to_db(response, interview_id)
        interview_sessions2[user_id].question_id_in_use = question_id
        end_chack = call_ollama_api_streaming(response)
        print(end_chack)
        if end_chack == '종료' :
            update_feedback_to_db(user_answer, 'end', 'end',  question_id)
            update_final_feedback_to_db(user_id)
            return {"message": [response,'end']}

        return {"message": [response,'run']}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing answer: {str(e)}")
    

import traceback
interview_sessions3 = {}  # 사용자 ID 별로 면접 모델을 관리하는 dict
@router.post("/create_interview3/")
async def create_interview2(request: InterviewRequest):
    try:
        # MySQL에서 이력서, 기업정보, 채용정보, 직무정보 받아오기
        resume = await get_data("이력서", request.resume_id)
        corporate_information = await get_data("기업정보", request.corporate_id)
        job_information,recruitment_information = await get_data("직무정보", request.job_id)

        # GeneralQuestionsGenerator 객체를 생성하고, 질문을 생성 (면접 난이도 추가)
        questions_generator = GeneralQuestionsGenerator(
            resume=resume,
            corporate_information=corporate_information,
            recruitment_information=recruitment_information,
            job_information=job_information,
            difficulty_level=request.difficulty_level  # 면접 난이도 전달
        )
        general_questions = questions_generator.invoke()

        # InterviewAssistant 객체를 생성하여 면접 모델 초기화 (면접관 스타일 및 난이도 추가)
        # 간단한 테스트를 위해 직접 인스턴스화하는 코드를 추가
        try:
            interview_assistant = AutoGenInterviewAssistant(
                resume="sample resume",
                corporate_information="sample corp info",
                recruitment_information="sample recruitment info",
                job_information="sample job info",
                general_questions=["질문 1", "질문 2"],
                interview_style="general",
                difficulty_level=1
            )
            print("Model instantiated successfully!")
        except Exception as e:
            print( traceback.format_exc())
            print(f"Error: {str(e)}")


        # 면접 시작 메시지를 보내고 결과를 반환
        response, _, _ = interview_assistant.invoke(request.user_id, "면접을 시작하세요","")
        print(response)
        interview_id = save_interview_to_db(request)
        question_id = save_question_to_db(response, interview_id)
        user = UserInUse(request.user_id, interview_assistant, question_id, interview_id)
        # 사용자 ID를 세션 ID로 사용하여 면접 모델을 관리
        interview_sessions3[request.user_id] = user
        return {"session_id": request.user_id, "message": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating interview: {str(e)}")