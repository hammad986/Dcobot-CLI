import sys
sys.path.insert(0, 'src')
from cli import KnowledgeBase
from pathlib import Path
from collections import deque

kb = KnowledgeBase()
chat = deque()

print("=== 1. INITIAL STATE ===")
print("is_empty:", kb.is_empty, "  docs:", kb.get_indexed_filenames(), "  chunks:", kb.get_total_chunks())

print("\n=== 2. ADD DOCUMENT ===")
r = kb.add_files([Path("Muhammed_Hammad_S_Resume.pdf")])
print("Indexed:", r["indexed"])
print("Skipped:", r["skipped"])
print("is_empty:", kb.is_empty, "  docs:", kb.get_indexed_filenames(), "  chunks:", kb.get_total_chunks())

print("\n=== 3. ADD SAME FILE AGAIN (must skip) ===")
r2 = kb.add_files([Path("Muhammed_Hammad_S_Resume.pdf")])
print("Skipped:", r2["skipped"])
assert r2["indexed"] == [], "FAIL: should not re-index duplicate"
assert len(r2["skipped"]) == 1, "FAIL: should report one skip"

print("\n=== 4. CLEAR ALL ===")
kb.clear_all()
chat.clear()
print("is_empty:", kb.is_empty, "  docs:", kb.get_indexed_filenames(), "  chunks:", kb.get_total_chunks())
assert kb.is_empty, "FAIL: should be empty after clear"
assert kb.get_indexed_filenames() == [], "FAIL: filenames should be empty"
assert kb.get_total_chunks() == 0, "FAIL: chunks should be 0"

print("\n=== 5. RE-ADD AFTER CLEAR (must NOT skip) ===")
r3 = kb.add_files([Path("Muhammed_Hammad_S_Resume.pdf")])
print("Indexed:", r3["indexed"])
print("Skipped:", r3["skipped"])
assert r3["indexed"], "FAIL: should index after clear"
assert r3["skipped"] == [], "FAIL: should not skip after clear"
assert not kb.is_empty, "FAIL: should not be empty after re-add"
print("is_empty:", kb.is_empty, "  docs:", kb.get_indexed_filenames(), "  chunks:", kb.get_total_chunks())

print("\n=== ALL TESTS PASSED ===")
