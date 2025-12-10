# main.py
from src.indexer import Indexer
from src.searcher import Searcher

DATA_DIR = r"E:\3-2 알고리즘\167.과학기술표준분류 대응 특허 데이터\01-1.정식개방데이터\Training\01.원천데이터\unzipped"
INDEX_DIR = "index"
DOC_TABLE_FILE = "doc_table.json"
TERM_DICT_FILE = "term_dict.json"
POSTINGS_FILE = "postings.bin"

if __name__ == "__main__":
    while True:
        task = input("작업을 선택하세요 (index/search/exit): ").strip().lower()
        if task in ("exit", "e"):
            print("프로그램 종료")
            break
        elif task in ("index", "i"):
            indexer = Indexer(DATA_DIR, INDEX_DIR, DOC_TABLE_FILE, TERM_DICT_FILE, POSTINGS_FILE)
            indexer.build_index()
            print(f"색인이 완료되었습니다. 색인 결과는 '{INDEX_DIR}'에 저장되었습니다.")
        elif task in ("search", "s"):
            searcher = Searcher(INDEX_DIR, DOC_TABLE_FILE, TERM_DICT_FILE, POSTINGS_FILE)
            searcher.set_data_dir(DATA_DIR)
            while True:
                input_query = input("검색어를 입력하세요: ").strip()
                if not input_query:
                    break
                searcher.process_query(input_query)
            searcher.close()
        else:
            print("잘못된 입력입니다. index/search/exit 중 선택하세요.")