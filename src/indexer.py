# src/indexer.py
import os
import json
import struct
from collections import defaultdict
from src.tokenizer import extract_terms

RECORD_FMT = "ii"   # doc_id, tf
RECORD_SIZE = struct.calcsize(RECORD_FMT)  # 8 bytes


class Indexer:
    def __init__(self, data_dir, index_dir, doc_table_file, term_dict_file, postings_file):
        self.data_dir = os.path.abspath(data_dir)
        self.index_dir = os.path.abspath(index_dir)
        os.makedirs(self.index_dir, exist_ok=True)

        self.doc_table_file = os.path.join(self.index_dir, doc_table_file)
        self.term_dict_file = os.path.join(self.index_dir, term_dict_file)
        self.postings_file = os.path.join(self.index_dir, postings_file)

        # term_postings[term][field] = { doc_id: tf }
        self.term_postings = defaultdict(lambda: {"T": {}, "A": {}, "C": {}})
        self.doc_table = []
        self.N = 0

    def _iter_data_files(self):
        """DATA_DIR 아래 모든 .json 파일을 재귀적으로 찾아 반환"""
        for root, _, files in os.walk(self.data_dir):
            for fname in sorted(files):
                if fname.lower().endswith(".json"):
                    fullpath = os.path.join(root, fname)
                    relpath = os.path.relpath(fullpath, self.data_dir).replace("\\", "/")
                    yield fullpath, relpath

    def _safe_get_field(self, data, keys):
        """data에서 keys 순서로 값 찾기"""
        for k in keys:
            if k in data and data[k]:
                return data[k]
        
        ds = data.get("dataset")
        if isinstance(ds, dict):
            for k in keys:
                if k in ds and ds[k]:
                    return ds[k]
        return ""

    def build_index(self):
        """필드별 분리 인덱싱"""
        sum_len = {"T": 0, "A": 0, "C": 0}
        doc_id = 0

        print("[INDEXER] 파일 스캔 중...")
        all_files = list(self._iter_data_files())
        total_files = len(all_files)
        print(f"[INDEXER] 총 {total_files}개 JSON 파일 발견")

        for idx, (fullpath, relpath) in enumerate(all_files):
            if (idx + 1) % 1000 == 0:
                print(f"[INDEXER] {idx + 1}/{total_files} ({(idx + 1) * 100 / total_files:.1f}%) 처리 중...")

            try:
                with open(fullpath, encoding="utf-8") as f:
                    j = json.load(f)
            except Exception as e:
                print(f"[WARN] 파일 읽기 실패 {fullpath}: {e}")
                continue

            # 필드 추출
            title_raw = self._safe_get_field(j, ["invention_title", "title", "T"])
            abstract_raw = self._safe_get_field(j, ["abstract", "A", "summary"])
            claims_raw = self._safe_get_field(j, ["claims", "C", "claim"])

            if isinstance(claims_raw, list):
                claims_text = " ".join(str(x) for x in claims_raw if x)
            else:
                claims_text = str(claims_raw) if claims_raw else ""

            title_text = str(title_raw) if title_raw else ""
            abstract_text = str(abstract_raw) if abstract_raw else ""

            # 토큰화
            T_terms = extract_terms(title_text)
            A_terms = extract_terms(abstract_text)
            C_terms = extract_terms(claims_text)

            len_T = len(T_terms)
            len_A = len(A_terms)
            len_C = len(C_terms)
            
            sum_len["T"] += len_T
            sum_len["A"] += len_A
            sum_len["C"] += len_C

            # doc_table 저장
            self.doc_table.append({
                "doc_id": doc_id,
                "relpath": relpath,
                "len_T": len_T,
                "len_A": len_A,
                "len_C": len_C,
                "T_text": title_text
            })

            # 필드별 포스팅 집계
            for t in T_terms:
                self.term_postings[t]["T"][doc_id] = self.term_postings[t]["T"].get(doc_id, 0) + 1
            for t in A_terms:
                self.term_postings[t]["A"][doc_id] = self.term_postings[t]["A"].get(doc_id, 0) + 1
            for t in C_terms:
                self.term_postings[t]["C"][doc_id] = self.term_postings[t]["C"].get(doc_id, 0) + 1

            doc_id += 1

        self.N = doc_id
        avg_len_T = (sum_len["T"] / self.N) if self.N else 1.0
        avg_len_A = (sum_len["A"] / self.N) if self.N else 1.0
        avg_len_C = (sum_len["C"] / self.N) if self.N else 1.0

        print(f"[INDEXER] 포스팅 작성 중...")
        
        # postings.bin 작성: 필드별 분리
        term_dict = {}
        with open(self.postings_file, "wb") as fp:
            for term in sorted(self.term_postings.keys()):
                field_postings = self.term_postings[term]
                
                # 전역 df (모든 필드 통합)
                all_docs = set()
                for field in ["T", "A", "C"]:
                    all_docs.update(field_postings[field].keys())
                df = len(all_docs)
                
                term_entry = {"df": df}
                
                # 각 필드별 포스팅
                for field in ["T", "A", "C"]:
                    posting_map = field_postings[field]
                    if not posting_map:
                        term_entry[field] = {"start": 0, "length": 0}
                        continue
                    
                    start = fp.tell()
                    length = len(posting_map)
                    
                    for did in sorted(posting_map.keys()):
                        tf = posting_map[did]
                        fp.write(struct.pack(RECORD_FMT, int(did), int(tf)))
                    
                    term_entry[field] = {"start": start, "length": length}
                
                term_dict[term] = term_entry

            # 메타 정보
            term_dict["_meta"] = {
                "N": self.N,
                "avg_len_T": avg_len_T,
                "avg_len_A": avg_len_A,
                "avg_len_C": avg_len_C
            }

        # JSON 저장
        with open(self.doc_table_file, "w", encoding="utf-8") as f:
            json.dump(self.doc_table, f, ensure_ascii=False, indent=2)
        
        with open(self.term_dict_file, "w", encoding="utf-8") as f:
            json.dump(term_dict, f, ensure_ascii=False, indent=2)

        print(f"[INDEXER] 완료: N={self.N}, 고유 term={len(self.term_postings)}")