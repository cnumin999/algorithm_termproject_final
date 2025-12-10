# src/searcher.py
import os
import json
import math
import struct
import re
from src.tokenizer import extract_terms

RECORD_FMT = "ii"
RECORD_SIZE = struct.calcsize(RECORD_FMT)

# BM25F 파라미터
W = {"T": 2.5, "A": 1.5, "C": 1.1}
B = {"T": 0.3, "A": 0.75, "C": 0.8}
K1 = 2.0
WINDOW = 80

def normalize_space(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or "").strip())

def ci_find_all(text_low: str, sub_low: str):
    """대소문자 무시 검색"""
    res = []
    start = 0
    while True:
        idx = text_low.find(sub_low, start)
        if idx == -1:
            break
        res.append((idx, idx + len(sub_low)))
        start = idx + 1
    return res

def center_window_bounds(length: int, match_start: int, match_end: int, window: int = WINDOW):
    """매치를 중심으로 윈도우 생성"""
    center = (match_start + match_end) // 2
    half = window // 2
    s = max(0, center - half)
    e = min(length, s + window)
    if e - s < window:
        s = max(0, e - window)
    return s, e

def highlight_terms(snippet: str, terms: list):
    """검색어 하이라이트"""
    if not snippet:
        return snippet
    low = snippet.lower()
    terms_low = sorted({t.lower() for t in terms}, key=lambda x: -len(x))
    
    spans = []
    for t in terms_low:
        pos = 0
        while True:
            idx = low.find(t, pos)
            if idx == -1:
                break
            spans.append((idx, idx + len(t)))
            pos = idx + 1
    
    if not spans:
        return snippet
    
    # 겹치는 span 병합
    spans.sort()
    merged = []
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    
    # 하이라이트 적용
    result = []
    last = 0
    for s, e in merged:
        result.append(snippet[last:s])
        result.append("<<")
        result.append(snippet[s:e])
        result.append(">>")
        last = e
    result.append(snippet[last:])
    return "".join(result)


class Searcher:
    def __init__(self, index_dir, doc_table_file, term_dict_file, postings_file):
        self.index_dir = os.path.abspath(index_dir)
        self.doc_table_path = os.path.join(self.index_dir, doc_table_file)
        self.term_dict_path = os.path.join(self.index_dir, term_dict_file)
        self.postings_path = os.path.join(self.index_dir, postings_file)

        doc_list = json.load(open(self.doc_table_path, encoding="utf-8"))
        self.doc_table = {entry["doc_id"]: entry for entry in doc_list}
        self.term_dict = json.load(open(self.term_dict_path, encoding="utf-8"))
        self.fp = open(self.postings_path, "rb")

        meta = self.term_dict.get("_meta", {})
        self.N = int(meta.get("N", len(self.doc_table)))
        self.avg_len = {
            "T": float(meta.get("avg_len_T", 1.0)),
            "A": float(meta.get("avg_len_A", 1.0)),
            "C": float(meta.get("avg_len_C", 1.0))
        }
        
        # DATA_DIR 경로 추정 (relpath 기준)
        self.data_dir = None

    def set_data_dir(self, data_dir):
        """DATA_DIR 설정 (main.py에서 호출)"""
        self.data_dir = os.path.abspath(data_dir)
    
    def _load_field_text(self, info: dict, field: str):
        """필요 시 JSON 파일에서 필드 텍스트 로드"""
        if field == "T":
            return normalize_space(info.get("T_text", ""))
        
        if not self.data_dir:
            return ""
        
        relpath = info.get("relpath", "")
        if not relpath:
            return ""
        
        fullpath = os.path.join(self.data_dir, relpath)
        
        try:
            with open(fullpath, encoding="utf-8") as f:
                j = json.load(f)
            
            # 필드 추출
            if field == "A":
                keys = ["abstract", "A", "summary"]
            elif field == "C":
                keys = ["claims", "C", "claim"]
            else:
                return ""
            
            # dataset 고려
            for k in keys:
                if k in j and j[k]:
                    raw = j[k]
                    if isinstance(raw, list):
                        return normalize_space(" ".join(str(x) for x in raw if x))
                    return normalize_space(str(raw) if raw else "")
            
            ds = j.get("dataset")
            if isinstance(ds, dict):
                for k in keys:
                    if k in ds and ds[k]:
                        raw = ds[k]
                        if isinstance(raw, list):
                            return normalize_space(" ".join(str(x) for x in raw if x))
                        return normalize_space(str(raw) if raw else "")
            
            return ""
        except Exception as e:
            print(f"[WARN] 파일 읽기 실패 {fullpath}: {e}")
            return ""

    def close(self):
        if self.fp:
            self.fp.close()

    def parse_query(self, q: str):
        """쿼리 파싱: [PREFIX][FIELD] terms"""
        q = q.strip()
        prefix = set()
        fields = []
        
        # PREFIX, FIELD 추출
        pattern = re.compile(r'\[([^\]]+)\]')
        matches = pattern.findall(q)
        
        for m in matches:
            m_upper = m.strip().upper()
            if m_upper == "VERBOSE" or m_upper == "V":
                prefix.add("VERBOSE")
            elif m_upper == "AND" or m_upper == "A":
                prefix.add("AND")
            elif m_upper == "PHRASE" or m_upper == "P":
                prefix.add("PHRASE")
            elif m_upper.startswith("FIELD="):
                val = m_upper.split("=", 1)[1].strip()
                if val in ("T", "A", "C"):
                    fields.append(val)
        
        # 쿼리 텀 추출
        rest = pattern.sub("", q).strip()
        parts = rest.split() if rest else []
        
        token_terms = []
        for p in parts:
            toks = extract_terms(p)
            token_terms.extend(toks if toks else [p.lower()])
        
        if "AND" in prefix and "PHRASE" in prefix:
            raise ValueError("AND와 PHRASE는 동시 사용 불가")
        
        return {
            "prefix": prefix,
            "fields": list(dict.fromkeys(fields)) if fields else None,
            "token_terms": token_terms,
            "raw_terms": parts,
            "raw_query": rest
        }

    def _read_field_postings(self, term: str, field: str):
        """필드별 포스팅 읽기"""
        if term not in self.term_dict:
            return {}
        
        entry = self.term_dict[term]
        field_info = entry.get(field, {})
        start = int(field_info.get("start", 0))
        length = int(field_info.get("length", 0))
        
        if length <= 0:
            return {}
        
        self.fp.seek(start)
        res = {}
        for _ in range(length):
            data = self.fp.read(RECORD_SIZE)
            if len(data) != RECORD_SIZE:
                break
            did, tf = struct.unpack(RECORD_FMT, data)
            res[int(did)] = int(tf)
        return res

    def get_postings(self, term: str, field_set=None):
        """통합 포스팅 반환"""
        posts = {"T": {}, "A": {}, "C": {}}
        
        for field in ["T", "A", "C"]:
            if field_set is None or field in field_set:
                posts[field] = self._read_field_postings(term, field)
        
        all_docs = set()
        for field in ["T", "A", "C"]:
            all_docs.update(posts[field].keys())
        
        result = {}
        for did in all_docs:
            result[did] = {
                "T": posts["T"].get(did, 0),
                "A": posts["A"].get(did, 0),
                "C": posts["C"].get(did, 0)
            }
        return result

    def candidate_docs(self, token_terms: list, mode="OR", fields=None, phrase_raw=None):
        """후보 문서 선택"""
        if mode == "PHRASE":
            # PHRASE: Title에서 exact match
            phrase_norm = normalize_space(phrase_raw).lower()
            if not phrase_norm:
                return set()
            
            cands = set()
            for did, info in self.doc_table.items():
                title = self._load_field_text(info, "T").lower()
                if phrase_norm in title:
                    cands.add(did)
            return cands

        # OR/AND
        posting_sets = []
        for t in token_terms:
            p = self.get_postings(t, field_set=fields)
            posting_sets.append(set(p.keys()))
        
        if not posting_sets:
            return set()
        
        if mode == "OR":
            res = set()
            for s in posting_sets:
                res |= s
            return res
        else:  # AND
            res = posting_sets[0].copy()
            for s in posting_sets[1:]:
                res &= s
            return res

    def idf(self, term: str):
        """BM25 IDF"""
        df = int(self.term_dict.get(term, {}).get("df", 0))
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score_bm25f(self, candidates: set, token_terms: list, fields=None):
        """BM25F 점수 계산"""
        scores = {}
        term_posts = {t: self.get_postings(t, field_set=fields) for t in token_terms}
        
        for did in candidates:
            info = self.doc_table.get(did, {})
            len_d = {
                "T": int(info.get("len_T", 0)),
                "A": int(info.get("len_A", 0)),
                "C": int(info.get("len_C", 0))
            }
            
            score = 0.0
            for t in token_terms:
                posts = term_posts.get(t, {})
                ent = posts.get(did, {"T": 0, "A": 0, "C": 0})
                
                # 필드별 가중 TF 계산
                tf_weighted = 0.0
                for f in ["T", "A", "C"]:
                    if fields and f not in fields:
                        continue
                    
                    tf_f = int(ent.get(f, 0))
                    avg = self.avg_len.get(f, 1.0) or 1.0
                    denom = (1.0 - B[f] + B[f] * (len_d[f] / avg))
                    tf_norm = (tf_f / denom) if denom != 0 else 0
                    tf_weighted += W[f] * tf_norm
                
                if tf_weighted > 0:
                    numerator = (K1 + 1.0) * tf_weighted
                    denominator = K1 + tf_weighted
                    score += self.idf(t) * (numerator / denominator)
            
            if score > 0:
                scores[did] = score
        
        return scores

    def _snippet_or(self, info: dict, raw_terms: list, fields_priority: list):
        """OR: 가장 많은 term 포함한 한 부분 (모든 필드 고려)"""
        terms_low = [t.lower() for t in raw_terms]
        
        best = None
        
        for field_idx, field in enumerate(fields_priority):
            # 필드별 텍스트 가져오기
            text_orig = self._load_field_text(info, field)
            if not text_orig:
                continue
            
            text_low = text_orig.lower()
            
            positions = []
            for t in set(terms_low):
                for s, e in ci_find_all(text_low, t):
                    positions.append((s, e))
            
            if not positions:
                continue
            
            # 각 위치에서 윈도우 생성하고 distinct term 개수 계산
            for s, e in positions:
                ws, we = center_window_bounds(len(text_low), s, e, WINDOW)
                window = text_low[ws:we]
                distinct = sum(1 for t in set(terms_low) if t in window)
                
                # 우선순위: 1) distinct 개수, 2) 필드 우선순위, 3) 앞쪽 위치
                if best is None or \
                   distinct > best[0] or \
                   (distinct == best[0] and field_idx < best[1]) or \
                   (distinct == best[0] and field_idx == best[1] and ws < best[3]):
                    best = (distinct, field_idx, field, ws, we, text_orig)
        
        if not best:
            return None, None
        
        _, _, field, ws, we, text_orig = best
        snippet = text_orig[ws:we]
        highlighted = highlight_terms(snippet, raw_terms)
        return field, highlighted

    def _snippets_and(self, info: dict, raw_terms: list, fields_priority: list):
        """AND: 모든 term 커버할 때까지 여러 부분 (모든 필드 고려)"""
        terms_low = [t.lower() for t in raw_terms]
        needed = set(terms_low)
        snippets = []
        
        # 필드 우선순위대로 순회
        for field in fields_priority:
            if not needed:
                break
            
            # 필드별 텍스트 가져오기
            text_orig = self._load_field_text(info, field)
            if not text_orig:
                continue
            
            text_low = text_orig.lower()
            
            # 현재 필드에서 필요한 term 찾기
            occurrences = []
            for t in list(needed):
                for s, e in ci_find_all(text_low, t):
                    occurrences.append((s, e, t))
            
            if not occurrences:
                continue
            
            occurrences.sort()
            used_windows = []
            
            for s, e, _ in occurrences:
                if not needed:
                    break
                
                ws, we = center_window_bounds(len(text_low), s, e, WINDOW)
                
                # 중복 윈도우 체크
                is_dup = False
                for pw_s, pw_e in used_windows:
                    overlap = max(0, min(we, pw_e) - max(ws, pw_s))
                    if overlap > WINDOW * 0.5:
                        is_dup = True
                        break
                
                if is_dup:
                    continue
                
                window_low = text_low[ws:we]
                window_orig = text_orig[ws:we]
                
                # 이 윈도우에서 커버되는 term 찾기
                found = [t for t in list(needed) if t in window_low]
                
                if found:
                    highlighted = highlight_terms(window_orig, raw_terms)
                    snippets.append((field, highlighted))
                    used_windows.append((ws, we))
                    
                    for t in found:
                        needed.discard(t)
        
        return snippets

    def _snippet_phrase(self, info: dict, phrase_raw: str):
        """PHRASE: 정확히 일치하는 한 곳 (Title만)"""
        title_orig = self._load_field_text(info, "T")
        title_low = title_orig.lower()
        phrase_norm = normalize_space(phrase_raw).lower()
        
        idx = title_low.find(phrase_norm)
        if idx == -1:
            return None, None
        
        ws, we = center_window_bounds(len(title_orig), idx, idx + len(phrase_norm), WINDOW)
        snippet = title_orig[ws:we]
        
        # 정확한 구문만 하이라이트
        pattern = re.escape(phrase_norm)
        highlighted = re.sub(pattern, lambda m: f"<<{m.group(0)}>>", snippet, flags=re.IGNORECASE, count=1)
        
        return "T", highlighted

    def process_query(self, user_query: str):
        """쿼리 처리"""
        try:
            parsed = self.parse_query(user_query)
        except Exception as e:
            print(f"쿼리 파싱 오류: {e}")
            return

        prefix = parsed["prefix"]
        fields = parsed["fields"]
        token_terms = parsed["token_terms"]
        raw_terms = parsed["raw_terms"]
        raw_query = parsed["raw_query"]

        if "PHRASE" in prefix:
            mode = "PHRASE"
        elif "AND" in prefix:
            mode = "AND"
        else:
            mode = "OR"

        field_set = set(fields) if fields else None

        # 후보 선택
        if mode == "PHRASE":
            cands = self.candidate_docs([], mode="PHRASE", phrase_raw=raw_query)
            scores = self.score_bm25f(cands, token_terms, fields={"T"})
        else:
            if not token_terms:
                print("검색어가 없습니다.")
                return
            cands = self.candidate_docs(token_terms, mode=mode, fields=field_set)
            scores = self.score_bm25f(cands, token_terms, fields=field_set)

        if not scores:
            print(f"\nRESULT:")
            print(f"검색어입력: {user_query}")
            print(f"총0개문서검색")
            print("검색결과가없습니다.")
            return

        sorted_docs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        total = len(sorted_docs)

        # 기본 출력
        print(f"\nRESULT:")
        print(f"검색어입력: {user_query}")
        print(f"총{total}개문서검색")
        top_n = min(5, total)
        print(f"상위{top_n}개문서:")
        for did, sc in sorted_docs[:5]:
            relpath = self.doc_table.get(did, {}).get("relpath", f"doc{did}")
            filename = relpath.split('/')[-1]
            print(f"{filename}    {sc:.2f}")

        # VERBOSE
        if "VERBOSE" in prefix:
            print("-" * 50)
            
            # 필드 우선순위 설정
            if field_set:
                fields_priority = [f for f in ["T", "A", "C"] if f in field_set]
            else:
                fields_priority = ["T", "A", "C"]
            
            for did, sc in sorted_docs[:5]:
                info = self.doc_table.get(did, {})
                relpath = info.get("relpath", f"doc{did}")
                filename = relpath.split('/')[-1]
                print(f"\nDocID: {did}, 파일명: {filename}, 점수: {sc:.4f}")

                if mode == "PHRASE":
                    # PHRASE는 항상 Title만
                    f, snippet = self._snippet_phrase(info, raw_query)
                    if snippet:
                        print(f"[TITLE] {snippet}")
                    else:
                        print(" (TITLE에서 일치하는 구문 없음)")
                
                elif mode == "OR":
                    # OR: 가장 많은 term 포함한 한 부분
                    f, snippet = self._snippet_or(info, raw_terms, fields_priority)
                    if snippet:
                        label = {"T": "TITLE", "A": "ABSTRACT", "C": "CLAIMS"}.get(f, f)
                        print(f"[{label}] {snippet}")
                    else:
                        # fallback
                        for field in fields_priority:
                            text = self._load_field_text(info, field)
                            if text:
                                text = text[:WINDOW]
                                label = {"T": "TITLE", "A": "ABSTRACT", "C": "CLAIMS"}.get(field, field)
                                print(f"[{label}] {text}")
                                break
                
                elif mode == "AND":
                    # AND: 모든 term 나올 때까지 여러 부분
                    snippets = self._snippets_and(info, raw_terms, fields_priority)
                    if snippets:
                        for f, sn in snippets:
                            label = {"T": "TITLE", "A": "ABSTRACT", "C": "CLAIMS"}.get(f, f)
                            print(f"[{label}] {sn}")
                    else:
                        # fallback
                        for field in fields_priority:
                            text = self._load_field_text(info, field)
                            if text:
                                text = text[:WINDOW]
                                label = {"T": "TITLE", "A": "ABSTRACT", "C": "CLAIMS"}.get(field, field)
                                print(f"[{label}] {text}")
                                break
            
            print("-" * 50)