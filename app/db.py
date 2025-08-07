import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:mypassword@10.30.0.3:5432/chatdb")

engine = create_engine(DATABASE_URL, connect_args={"options": "-c search_path=public"})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    uid = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    password = Column(String, nullable=False)
    email = Column(String, nullable=True)
    affiliation = Column(String, nullable=True)
    register_date = Column(DateTime, default=datetime.utcnow)
    login_logs = relationship("LoginLog", back_populates="user")
    notes = relationship("Note", back_populates="user")
    prompt_configs = relationship("PromptConfig", back_populates="user")

class LoginLog(Base):
    __tablename__ = "login_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.user_id"), index=True)
    name = Column(String)
    login_date = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="login_logs")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False)
    first_question = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    chat_logs = relationship("ChatLog", back_populates="session")

class ChatLog(Base):
    __tablename__ = "chat_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    question = Column(Text)
    answer = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("ChatSession", back_populates="chat_logs")

class Note(Base):
    __tablename__ = "notes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False, index=True)
    session_id = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    role = Column(String, nullable=False)  # "ai" or "user"
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="notes")

class PromptConfig(Base):
    __tablename__ = "prompt_configs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.uid"), nullable=False, index=True)
    prompt_type = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    user = relationship("User", back_populates="prompt_configs")


def init_db():
    Base.metadata.create_all(bind=engine)

# def init_db():
#     pass

def create_chat_session(db, user_id: str, first_question: str, session_id: str):
    session_entry = ChatSession(id=session_id, user_id=user_id, first_question=first_question)
    db.add(session_entry)
    db.commit()
    db.refresh(session_entry)
    return session_entry

def log_chat(db, user_id: str, session_id: str, question: str, answer: str):
    chat_log = ChatLog(user_id=user_id, session_id=session_id, question=question, answer=answer)
    db.add(chat_log)
    db.commit()

def get_prompt_config(db, user_id: int, prompt_type: str = "role"):
    """
    role 프롬프트는 사용자별, 그 외는 user_id=1(기본)만 리턴
    """
    if prompt_type == "role":
        row = db.query(PromptConfig).filter_by(user_id=user_id, prompt_type="role").first()
        if row:
            return row
    # 없는 경우 or prompt_type != 'role': default(user_id=1) 리턴
    return db.query(PromptConfig).filter_by(user_id=1, prompt_type=prompt_type).first()

def upsert_prompt_config(db, user_id: int, content: str, prompt_type: str = "role"):
    """
    role만 사용자별 업데이트, 그 외는 무시(수정 불가)
    """
    if prompt_type != "role":
        raise ValueError("Only 'role' prompt can be customized.")
    prompt = db.query(PromptConfig).filter_by(user_id=user_id, prompt_type="role").first()
    if prompt:
        prompt.content = content
    else:
        prompt = PromptConfig(user_id=user_id, prompt_type="role", content=content)
        db.add(prompt)
    db.commit()
    return prompt

def delete_prompt_config(db, user_id: int, prompt_type: str = "role"):
    if prompt_type != "role":
        raise ValueError("Only 'role' prompt can be deleted.")
    db.query(PromptConfig).filter_by(user_id=user_id, prompt_type="role").delete()
    db.commit()