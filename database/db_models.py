from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Instance(Base):
    __tablename__ = "instance"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    routings = relationship("Routing", back_populates="instance", cascade="all, delete-orphan")


class Routing(Base):
    __tablename__ = "routing"

    id = Column(String(255), primary_key=True)
    instance_id = Column(Integer, ForeignKey("instance.id"), nullable=False)
    instance = relationship("Instance", back_populates="routings")
    operations = relationship("Operation", back_populates="routing", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="routing", cascade="all, delete-orphan")


class Operation(Base):
    __tablename__ = "operation"
    id = Column(Integer, primary_key=True, autoincrement=True)
    routing_id = Column(String(255), ForeignKey("routing.id"), nullable=False)
    machine = Column(String(255), nullable=False)
    duration = Column(Integer, nullable=False)
    routing = relationship("Routing", back_populates="operations")


class Job(Base):
    __tablename__ = "job"

    id = Column(String(255), primary_key=True)  # z.B. "J25-0001"
    routing_id = Column(String(255), ForeignKey("routing.id"), nullable=False)
    arrival = Column(Integer, nullable=False)
    ready_time = Column(Integer, nullable=False)
    deadline = Column(Integer, nullable=False)

    routing = relationship("Routing", back_populates="jobs")