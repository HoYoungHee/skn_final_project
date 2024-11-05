# organize_models.py
import os
import PyPDF2
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

class BaseOrganizeModel:
    def __init__(self, pdf_path):
        # API KEY 정보로드
        load_dotenv()
        self.llm = ChatOpenAI(temperature=0)
        self.pdf_path = pdf_path

    # PDF 파일 읽기 및 텍스트 추출 함수
    def extract_text_from_pdf(self):
        text = ""
        with open(self.pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    # 프롬프트 생성 함수 (자식 클래스에서 구현)
    def create_prompt(self, text):
        raise NotImplementedError("create_prompt 메서드는 자식 클래스에서 구현해야 합니다.")

    # 텍스트를 일정 길이로 나누는 함수
    def split_text(self, text, max_tokens=3000):
        paragraphs = text.split("\n")
        chunks = []
        current_chunk = ""
        current_length = 0

        for paragraph in paragraphs:
            paragraph_length = len(paragraph.split())
            if current_length + paragraph_length > max_tokens:
                chunks.append(current_chunk)
                current_chunk = paragraph + "\n"
                current_length = paragraph_length
            else:
                current_chunk += paragraph + "\n"
                current_length += paragraph_length

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    # 내용을 항목별로 정리하는 함수
    def organize_content(self, text):
        chunks = self.split_text(text)
        organized_content = ""
        for chunk in chunks:
            prompt = self.create_prompt(chunk)
            response = self.llm.predict(prompt)
            organized_content += response + "\n"
        return organized_content

    # 실행 함수
    def run(self):
        raise NotImplementedError("create_prompt 메서드는 자식 클래스에서 구현해야 합니다.")


class ResumeOrganizeModel(BaseOrganizeModel):
    # 이력서 내용을 항목별로 정리하는 프롬프트 생성 함수
    def create_prompt(self, text):
        return (
            "다음은 이력서의 내용입니다:\n" + text +
            "\n이 내용을 항목별로 정리해주세요 (예: 경력, 학력, 기술, 연락처 등)."
        )
    
    
    # 실행 함수
    def run(self):
        text = self.extract_text_from_pdf()
        organized_content = self.organize_content(text)
        final_prompt = (
            "다음은 이력서의 내용입니다:\n" + organized_content +
            "\n이 내용을 정보의 손실 없이 최대한 자세하게 항목별로 정리해주세요 (예: 경력, 학력, 기술, 연락처 등)."
        )
        final_response = self.llm.predict(final_prompt)
        return final_response


class RecruitmentInformationOrganizeModel(BaseOrganizeModel):
    # 채용 정보를 항목별로 정리하는 프롬프트 생성 함수
    def create_prompt(self, text):
        return (
            "다음은 채용 정보의 내용입니다:\n" + text +
            "\n이 내용을 항목별로 정리해주세요 (예: 회사명, 직무, 요구 기술, 근무 조건 등)."
        )
     # 실행 함수
    def run(self):
        text = self.extract_text_from_pdf()
        organized_content = self.organize_content(text)
        final_prompt = (
            
            "다음은 채용 정보의 내용입니다:\n" + organized_content +
            "\n이 내용을 정보의 손실 없이 최대한 자세하게 항목별로 정리해주세요 (예: 회사명, 직무, 요구 기술, 근무 조건 등)."
        )
        final_response = self.llm.predict(final_prompt)
        return final_response


class JobInformationOrganizeModel(BaseOrganizeModel):
    # 직무 정보를 항목별로 정리하는 프롬프트 생성 함수
    def create_prompt(self, text):
        return (
            "다음은 직무 정보의 내용입니다:\n" + text +
            "\n이 내용을 항목별로 정리해주세요 (예: 직무 설명, 필요 역량, 관련 기술 등)."
        )
     # 실행 함수
    def run(self):
        text = self.extract_text_from_pdf()
        organized_content = self.organize_content(text)
        final_prompt = (
          "다음은 직무 정보의 내용입니다:\n" + organized_content +
            "\n이 내용을 정보의 손실 없이 최대한 자세하게 항목별로 정리해주세요 (예: 직무 설명, 필요 역량, 관련 기술 등)."
        )
        final_response = self.llm.predict(final_prompt)
        return final_response


class CorporateInformationOrganizeModel(BaseOrganizeModel):
    # 기업 정보를 항목별로 정리하는 프롬프트 생성 함수
    def create_prompt(self, text):
        return (
            "다음은 기업 정보의 내용입니다:\n" + text +
            "\n이 내용을 항목별로 정리해주세요 (예: 회사 소개, 업종, 위치, 규모 등)."
        )
     # 실행 함수
    def run(self):
        text = self.extract_text_from_pdf()
        organized_content = self.organize_content(text)
        final_prompt = (
            "다음은 기업 정보의 내용입니다:\n" + organized_content +
            "\n이 내용을 정보의 손실 없이 최대한 자세하게 항목별로 정리해주세요 (예: 회사 소개, 업종, 위치, 규모 등)."
        )
        final_response = self.llm.predict(final_prompt)
        return final_response


