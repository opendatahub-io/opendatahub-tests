import logging
import os

from sqlalchemy import Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from utilities.must_gather_collector import get_base_dir

LOGGER = logging.getLogger(__name__)

TEST_DB = "opendatahub-tests.db"


class Base(DeclarativeBase):
    pass


class OpenDataHubTestTable(Base):
    __tablename__ = "OpenDataHubTestTable"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, nullable=False)
    test_name: Mapped[str] = mapped_column(String(500))
    start_time: Mapped[int] = mapped_column(Integer, nullable=False)


class Database:
    def __init__(self, database_file_name: str = TEST_DB, verbose: bool = True) -> None:
        self.database_file_path = os.path.join(get_base_dir(), database_file_name)
        self.connection_string = f"sqlite:///{self.database_file_path}"
        self.verbose = verbose
        self.engine = create_engine(url=self.connection_string, echo=self.verbose)
        Base.metadata.create_all(bind=self.engine)

    def insert_test_start_time(self, test_name: str, start_time: int) -> None:
        with Session(bind=self.engine) as db_session:
            new_table_entry = OpenDataHubTestTable(test_name=test_name, start_time=start_time)
            db_session.add(new_table_entry)
            db_session.commit()

    def get_test_start_time(self, test_name: str) -> int:
        with Session(bind=self.engine) as db_session:
            result_row = (
                db_session
                .query(OpenDataHubTestTable)
                .with_entities(OpenDataHubTestTable.start_time)
                .filter_by(test_name=test_name)
                .first()
            )
            if result_row:
                start_time_value = result_row[0]
            else:
                start_time_value = 0
                LOGGER.warning(f"No test found with name: {test_name}")
            return start_time_value
