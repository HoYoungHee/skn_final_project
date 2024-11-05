from models.db_models import *

# models/create_tables.py
def create_tables_if_not_exists():
    connection = get_db_connection()
    if connection is None:
        print("Failed to connect to the database.")
        return
    
    cursor = connection.cursor()
    try:
        # 테이블 생성
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS 기업정보 (
          기업정보_아이디 CHAR(36) NOT NULL,
          기업_이름 CHAR(36) NOT NULL,
          기업_파일 MEDIUMTEXT NOT NULL,
          기업_전처리 TEXT NULL,
          업로드_일시 TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
          수정_일시 TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (기업정보_아이디)
        );
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS 면접기록 (
          면접기록_아이디 CHAR(36) NOT NULL,
          사용자_아이디 CHAR(36) NOT NULL,
          이력서_아이디 CHAR(36) NOT NULL,
          직무정보_아이디 CHAR(36) NOT NULL,
          채용정보_아이디 CHAR(36) NOT NULL,
          면접_유형 VARCHAR(50) NULL,
          난이도 INT NULL,
          피드백 TEXT NULL,
          면접_일시 TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (면접기록_아이디)
        );
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS 사용자 (
          사용자_아이디 CHAR(36) NOT NULL,
          사용자_이름 CHAR(36) NOT NULL,
          이메일 VARCHAR(255) NOT NULL,
          비밀번호 CHAR(60) NOT NULL,
          계정_활성화여부 BOOLEAN NULL DEFAULT TRUE,
          생성일 TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
          수정일 TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (사용자_아이디)
        );
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS 이력서 (
          이력서_아이디 CHAR(36) NOT NULL,
          사용자_아이디 CHAR(36) NOT NULL,
          이력서_이름 CHAR(36) NOT NULL,
          이력서_파일 MEDIUMTEXT NOT NULL,
          이력서_전처리 TEXT NULL,
          업로드_일시 TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
          수정_일시 TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (이력서_아이디)
        );
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS 직무정보 (
          직무정보_아이디 CHAR(36) NOT NULL,
          기업정보_아이디 CHAR(36) NOT NULL,
          직무_이름 CHAR(36) NOT NULL,
          직무_파일 MEDIUMTEXT NOT NULL,
          직무_전처리 TEXT NULL,
          업로드_일시 TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
          수정_일시 TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (직무정보_아이디)
        );
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS 질문 (
          질문_아이디 CHAR(36) NOT NULL,
          면접기록_아이디 CHAR(36) NOT NULL,
          질문_내용 TEXT NOT NULL,
          모범답변 TEXT  NULL,
          생성일 TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
          사용자_답변 TEXT  NULL,
          피드백_내용 TEXT NOT NULL,
          PRIMARY KEY (질문_아이디)
        );
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS 채용정보 (
          채용정보_아이디 CHAR(36) NOT NULL,
          직무정보_아이디 CHAR(36) NOT NULL,
          채용_제목 CHAR(36) NOT NULL,
          채용_파일 MEDIUMTEXT NOT NULL,
          채용_전처리 TEXT NULL,
          업로드_일시 TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
          수정_일시 TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (채용정보_아이디)
        );
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS 피드백 (
          피드백_아이디 CHAR(36) NOT NULL,
          질문_아이디 CHAR(36) NOT NULL,
          피드백_내용 TEXT NULL,
          PRIMARY KEY (피드백_아이디)
        );
        ''')

        # Foreign Key 설정
        cursor.execute('''
        ALTER TABLE 이력서
        ADD CONSTRAINT FK_사용자_TO_이력서_0
          FOREIGN KEY (사용자_아이디)
          REFERENCES 사용자 (사용자_아이디);
        ''')

        cursor.execute('''
        ALTER TABLE 채용정보
        ADD CONSTRAINT FK_직무정보_TO_채용정보_0
          FOREIGN KEY (직무정보_아이디)
          REFERENCES 직무정보 (직무정보_아이디);
        ''')

        cursor.execute('''
        ALTER TABLE 직무정보
        ADD CONSTRAINT FK_기업정보_TO_직무정보_0
          FOREIGN KEY (기업정보_아이디)
          REFERENCES 기업정보 (기업정보_아이디);
        ''')

        cursor.execute('''
        ALTER TABLE 면접기록
        ADD CONSTRAINT FK_사용자_TO_면접기록_0
          FOREIGN KEY (사용자_아이디)
          REFERENCES 사용자 (사용자_아이디);
        ''')

        cursor.execute('''
        ALTER TABLE 면접기록
        ADD CONSTRAINT FK_이력서_TO_면접기록_0
          FOREIGN KEY (이력서_아이디)
          REFERENCES 이력서 (이력서_아이디);
        ''')

        cursor.execute('''
        ALTER TABLE 피드백
        ADD CONSTRAINT FK_면접기록_TO_피드백_0
          FOREIGN KEY (면접기록_아이디)
          REFERENCES 면접기록 (면접기록_아이디);
        ''')

        cursor.execute('''
        ALTER TABLE 질문
        ADD CONSTRAINT FK_면접기록_TO_질문_0
          FOREIGN KEY (면접기록_아이디)
          REFERENCES 면접기록 (면접기록_아이디);
        ''')

        cursor.execute('''
        ALTER TABLE 면접기록
        ADD CONSTRAINT FK_직무정보_TO_면접기록_0
          FOREIGN KEY (직무정보_아이디)
          REFERENCES 직무정보 (직무정보_아이디);
        ''')

        cursor.execute('''
        ALTER TABLE 면접기록
        ADD CONSTRAINT FK_채용정보_TO_면접기록_0
          FOREIGN KEY (채용정보_아이디)
          REFERENCES 채용정보 (채용정보_아이디);
        ''')
        
        connection.commit()
    except Error as e:
        print(f"Error while creating tables: {e}")
    finally:
        cursor.close()
        close_db_connection(connection)