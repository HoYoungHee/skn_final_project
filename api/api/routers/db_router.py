# routers/db_router.py
from fastapi import APIRouter, HTTPException, UploadFile, File
from models.db_models import *
import shutil
import os


router = APIRouter()

@router.get("/get_data/{table}/{id}")
async def get_data(table: str, id: int):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        if table == "이력서" : 
            query = f"SELECT {table}_전처리 FROM {table} WHERE {table}_아이디 = %s"
        elif table == "직무정보":
            query = f"SELECT {table[:-2]}_전처리 FROM {table} WHERE {table}_아이디 = %s"
            cursor.execute(query, (id,))
            job = cursor.fetchone()
            query = f"SELECT 채용_전처리 FROM 채용정보 WHERE 직무정보_아이디 = %s order by 업로드_일시"
            cursor.execute(query, (id,))
            recruitment =  cursor.fetchone()
            if not job:
                raise HTTPException(status_code=404, detail=f"{table} entry not found")
            elif not recruitment:
                raise HTTPException(status_code=404, detail=f"채용정보 entry not found")
            return job, recruitment
        else:
            query = f"SELECT {table[:-2]}_전처리 FROM {table} WHERE {table}_아이디 = %s"
        cursor.execute(query, (id,))
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail=f"{table} entry not found")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        if connection:
            connection.close()

@router.post("/upload_corporate/")
async def upload_corporate(corporate_name : str, file: UploadFile = File(...)):
    try:
        # 파일 시스템에 파일 저장
        corporate_id = str(uuid.uuid4())
        save_path = f"../DB/corporate/{corporate_id}_{file.filename}"
        
        with open(save_path, "wb") as buffer:
            buffer.write(await file.read())

        # 데이터베이스에 경로 정보 저장
        print(f"Saving corporate data with path: {save_path}")

        save_corporate_to_db(save_path, corporate_name)
        print("Corporate data saved successfully")
        return {"message": "Corporate information uploaded and saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload_job/")
async def upload_job(job_name : str, corporate_id: str,  file: UploadFile = File(...)):
    try:
        # 파일 시스템에 파일 저장
        job_id = str(uuid.uuid4())
        save_path = f"../DB/job/{job_id}_{file.filename}"
        
        print(save_path)
        with open(save_path, "wb") as buffer:
            buffer.write(await file.read())
        # 데이터베이스에 경로 정보 저장
        save_job_to_db(save_path, corporate_id, job_name)
        
        return {"message": "Job information uploaded and saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload_recruitment/")
async def upload_recruitment(recruitment_name : str, job_id: str, file: UploadFile = File(...)):
    try:
        # 파일 시스템에 파일 저장
        recruitment_id = str(uuid.uuid4())
        save_path = f"../DB/recruitment/{recruitment_id}_{file.filename}"
        
        with open(save_path, "wb") as buffer:
            buffer.write(await file.read())

        # 데이터베이스에 경로 정보 저장
        save_recruitment_to_db(save_path, job_id, recruitment_name)
        
        return {"message": "Recruitment information uploaded and saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/upload_resume/")
async def upload_resume( user_id: str, file: UploadFile = File(...)):
    try:
        # 파일 시스템에 파일 저장
        resume_id = str(uuid.uuid4())
        save_path = f"../DB/resume/{resume_id}_{file.filename}"

        with open(save_path, "wb") as buffer:
            buffer.write(await file.read())

        # 데이터베이스에 경로 정보 저장
        save_resume_to_db(save_path, user_id)

        return {"message": "Resume uploaded and saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
