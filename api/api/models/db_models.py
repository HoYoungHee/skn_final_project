# models/db_models.py
import mysql.connector
from mysql.connector import Error
import uuid
import os
from interview_model.organize_models import ResumeOrganizeModel, CorporateInformationOrganizeModel, JobInformationOrganizeModel, RecruitmentInformationOrganizeModel
from interview_model.interview_assistant_model import FinalFeedbackGenerator



DB_CONFIG = {
    'host': '192.168.0.3',
    'user': 'myuser',
    'password': '1115',
    'database': 'interview_db',
    'port' : '3306'
}

def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None

def close_db_connection(connection):
    if connection.is_connected():
        connection.close()

# def save_data_to_db(file_path, user_id=None, corporate_id = None, organizer_class=None, table_name=None, file_column=None, processed_column=None):
#     connection = get_db_connection()
#     if connection is None:
#         print("Failed to connect to the database.")
#         return
    
#     cursor = connection.cursor()
#     try:
#         organizer = organizer_class(file_path)
#         file_text = organizer.extract_text_from_pdf()
#         processed_text = organizer.run()
#         data_id = str(uuid.uuid4())
        
#         # SQL 쿼리 실행 - 파일 경로 및 전처리 데이터 저장
#         if user_id:
#             query = f"""
#             INSERT INTO {table_name} ({table_name}_아이디, 사용자_아이디, {file_column}, {processed_column}, 업로드_일시, 수정_일시)

#             VALUES (%s, %s, %s, %s, NOW(), NOW())
#             """
#             cursor.execute(query, (data_id, user_id, file_text, processed_text))
#         elif corporate_id :
#             query = f"""
#             INSERT INTO {table_name} ({table_name}_아이디, 기업정보_아이디, {file_column}, {processed_column}, 업로드_일시, 수정_일시)
#             VALUES (%s, %s, %s, %s, NOW(), NOW())
#             """
#             cursor.execute(query, (data_id, corporate_id, file_text, processed_text))
#         else:
#             query = f"""
#             INSERT INTO {table_name} ({table_name}_아이디, {file_column}, {processed_column}, 업로드_일시, 수정_일시)
#             VALUES (%s, %s, %s, NOW(), NOW())
#             """
#             print(query)
#             cursor.execute(query, (data_id, file_text, processed_text))
        
#         connection.commit()
#         print(f"Data saved successfully in {table_name} with ID:", data_id)
#     except Error as e:
#         print(f"Error while saving data to database: {e}")
#     finally:
#         cursor.close()
#         close_db_connection(connection)




def save_data_and_organize_to_db(file_path, organizer_class, query, *query_params):
    connection = get_db_connection()
    if connection is None:
        print("Failed to connect to the database.")
        return

    cursor = connection.cursor()
    try:
        # Organizer 클래스를 이용해 파일 처리
        organizer = organizer_class(file_path)
        file_text = organizer.extract_text_from_pdf()
        processed_text = organizer.run()
        data_id = str(uuid.uuid4())

        # 데이터 값들을 튜플로 준비
        data_values = (data_id, *query_params, file_path, processed_text)

        # 쿼리 실행
        cursor.execute(query, data_values)

        connection.commit()
        print(f"Data saved successfully in {query} with ID:", data_id)
    except Error as e:
        print(f"Error while saving data to database: {e}")
    finally:
        cursor.close()
        close_db_connection(connection)

# 각 데이터 저장을 위한 함수들
def save_resume_to_db(pdf_path, user_id):
    query = """
    INSERT INTO 이력서 (이력서_아이디, 사용자_아이디, 이력서_이름, 이력서_파일, 이력서_전처리, 업로드_일시, 수정_일시)
    VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
    """
    connection = get_db_connection()
    if connection is None:
        print("Failed to connect to the database.")
        return


    cursor = connection.cursor()

    # 파라미터 바인딩을 통해 쿼리를 실행
    cursor.execute("SELECT COUNT(*) FROM 이력서 WHERE 사용자_아이디 = %s", (user_id,))
    count = cursor.fetchone()[0]
    resume_name = f"{count + 1}번째 이력서"
    save_data_and_organize_to_db(pdf_path, ResumeOrganizeModel, query, user_id, resume_name)

def save_corporate_to_db(pdf_path, corporate_name):
    query = """
    INSERT INTO 기업정보 (기업정보_아이디, 기업_이름, 기업_파일, 기업_전처리, 업로드_일시, 수정_일시)
    VALUES (%s, %s, %s, %s, NOW(), NOW())
    """
    save_data_and_organize_to_db(pdf_path, CorporateInformationOrganizeModel, query, corporate_name)

def save_job_to_db(pdf_path, corporate_id,job_name):
    query = """
    INSERT INTO 직무정보 (직무정보_아이디, 기업정보_아이디, 직무_이름, 직무_파일, 직무_전처리, 업로드_일시, 수정_일시)
    VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
    """
    save_data_and_organize_to_db(pdf_path, JobInformationOrganizeModel, query,  corporate_id, job_name)

def save_recruitment_to_db(pdf_path, job_id, recruitment_name):
    query = """
    INSERT INTO 채용정보 (채용정보_아이디, 직무정보_아이디, 채용_제목, 채용_파일, 채용_전처리, 업로드_일시, 수정_일시)
    VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
    """
    save_data_and_organize_to_db(pdf_path, RecruitmentInformationOrganizeModel, query, job_id, recruitment_name)








def save_data_to_db(query, *query_params):
    connection = get_db_connection()
    if connection is None:
        print("Failed to connect to the database.")
        return

    connection = get_db_connection()
    if connection is None:
        print("Failed to connect to the database.")
        return

    cursor = connection.cursor()
    try:
        id = str(uuid.uuid4())

        # 데이터 값들을 튜플로 준비
        data_values = (id, *query_params)

        # 쿼리 실행
        cursor.execute(query, data_values)

        connection.commit()
        print(f"Data saved successfully in {query} with ID:", id)
        return id
    except Error as e:
        print(f"Error while saving data to database: {e} \n\n {query}")
    finally:
        cursor.close()
        close_db_connection(connection)



def update_data_to_db(query,exemplary_answer, user_answer, feedback, question_id_in_use):


    connection = get_db_connection()
    if connection is None:
        print("Failed to connect to the database.")
        return

    cursor = connection.cursor()
    try:
        data_values = (exemplary_answer, user_answer, feedback, question_id_in_use)
        # 쿼리 실행
        cursor.execute(query, data_values)

        connection.commit()
        print(f"Data saved successfully in {query} ")
    except Error as e:
        print(f"Error while saving data to database: {e} \n\n {query}")
    finally:
        cursor.close()
        close_db_connection(connection)


def save_interview_to_db(request):
    connection = get_db_connection()
    cursor = connection.cursor()
    query = f"SELECT 채용정보_아이디 FROM 채용정보 WHERE 직무정보_아이디 = %s order by 업로드_일시"
    cursor.execute(query, (request.job_id,))
    recruitment_id = cursor.fetchone()[0]

    
    query = """
    INSERT INTO 면접기록 (면접기록_아이디, 사용자_아이디, 이력서_아이디, 직무정보_아이디, 채용정보_아이디, 면접_유형, 난이도, 면접_일시)
    VALUES (%s, %s, %s, %s, %s, %s, %s,  NOW())
    """

    id = save_data_to_db(query, request.user_id, request.resume_id, request.job_id, recruitment_id, request.interview_style, request.difficulty_level)
    return id
    



def save_question_to_db(response, interview_id):
    query = """
    INSERT INTO 질문 (질문_아이디, 면접기록_아이디, 질문_내용, 생성일)
    VALUES (%s, %s, %s, NOW())
    """
    id = save_data_to_db(query, interview_id, response)
    return id


def update_feedback_to_db(user_answer, feedback, exemplary_answer,  question_id_in_use):
    query = """
    UPDATE 질문 SET 모범답변 = %s, 사용자_답변 = %s, 피드백_내용 = %s WHERE 질문_아이디 = %s
    """
    update_data_to_db(query, exemplary_answer, user_answer, feedback, question_id_in_use)


def fetch_interview_id_with_no_feedback(user_id):
    connection = None
    cursor = None
    try:
        # DB 연결 및 커서 생성
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        # 피드백이 아직 없는 면접기록을 가져오는 쿼리
        query = """
        SELECT 면접기록_아이디 
        FROM 면접기록 
        WHERE 사용자_아이디 = %s AND 피드백 IS NULL
        order by 면접_일시 desc
        LIMIT 1
        """
        cursor.execute(query, (user_id,))
        result = cursor.fetchone()

        if not result:
            raise Exception(f"No interview records with missing feedback found for user_id {user_id}")
        
        return result['면접기록_아이디']

    except Exception as e:
        print(f"Error fetching interview record: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def fetch_questions_for_interview(interview_id):
    connection = None
    cursor = None
    try:
        # DB 연결 및 커서 생성
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        # 면접기록_아이디를 기반으로 질문과 답변을 가져오는 쿼리
        query = """
        SELECT 질문_내용, 사용자_답변, 피드백_내용 
        FROM 질문 
        WHERE 면접기록_아이디 = %s 
        ORDER BY 생성일 DESC
        """
        cursor.execute(query, (interview_id,))
        questions = cursor.fetchall()

        if not questions:
            raise Exception(f"No questions found for interview_id {interview_id}")

        # 대화 형식으로 정리
        conversation = ""
        for question in questions:
            conversation += f"면접관: {question['질문_내용']}\n"
            conversation += f"면접자: {question['사용자_답변']}\n"
            if question['피드백_내용']:
                conversation += f"피드백: {question['피드백_내용']}\n"
            conversation += "\n"
        
        return conversation

    except Exception as e:
        print(f"Error fetching questions: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()



feedback_generator = FinalFeedbackGenerator(
                llm_model_name="gpt-4o-mini"
            )


from langchain_openai import ChatOpenAI

def update_final_feedback_to_db(user_id) : 
    interview_id = fetch_interview_id_with_no_feedback(user_id)
    conversation = fetch_questions_for_interview(interview_id)
    final_feedback = feedback_generator.generate_feedback(conversation=conversation)
    try:
        # DB 연결 및 커서 생성
        connection = get_db_connection()
        cursor = connection.cursor()

        # 면접기록 테이블에 최종 피드백을 업데이트하는 쿼리
        update_query = """
        UPDATE 면접기록 
        SET 피드백 = %s
        WHERE 면접기록_아이디 = %s
        """
        cursor.execute(update_query, (final_feedback, interview_id))

        connection.commit()  # 변경 사항을 DB에 반영
        
        update_query = """
        UPDATE 면접기록 
        SET 총점_요약 = %s
        WHERE 면접기록_아이디 = %s
        """
        organized_feedback_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        organized_feedback = organized_feedback_llm(final_feedback+"이 내용을 요약해주세요.").content
        print(organized_feedback)
        cursor.execute(update_query, (organized_feedback, interview_id))

        connection.commit()  # 변경 사항을 DB에 반영

        print(f"Successfully updated feedback for interview_id {interview_id}")

    except Exception as e:
        print(f"Error updating feedback: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()