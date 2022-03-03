import os
from pathlib import Path

TESTS_FOLDER_PATH = Path(os.path.abspath(__file__)).parent.__str__()
TEST_DATA_PATH = TESTS_FOLDER_PATH+'/__test_data__/data.pkl'
