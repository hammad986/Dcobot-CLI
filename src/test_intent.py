import sys
sys.path.insert(0, 'src')
from intent_router import classify

cases = [
    ('/help',                         'command'),
    ('/add resume.pdf',               'command'),
    ('/clear',                        'command'),
    ('/files',                        'command'),
    ('/stats',                        'command'),
    ('/debug',                        'command'),
    ('hi',                            'general_conversation'),
    ('hello',                         'general_conversation'),
    ('Hey!',                          'general_conversation'),
    ('good morning',                  'general_conversation'),
    ('how are you',                   'general_conversation'),
    ('who are you',                   'general_conversation'),
    ('what can you do',               'general_conversation'),
    ('can you speak Hinglish',        'general_conversation'),
    ('tell me a joke',                'general_conversation'),
    ('thank you',                     'general_conversation'),
    ('thanks',                        'general_conversation'),
    ('bye',                           'farewell'),
    ('goodbye',                       'farewell'),
    ('exit',                          'farewell'),
    ('quit',                          'farewell'),
    ('Summarize the document',        'document_question'),
    ('What are his skills?',          'document_question'),
    ('List certifications',           'document_question'),
    ('What is his education?',        'document_question'),
    ('Who is mentioned in the file?', 'document_question'),
    ('Summarize in Hinglish',         'mixed'),
    ('Explain in Urdu',               'mixed'),
]

passed = 0
failed = 0
for text, expected in cases:
    r = classify(text)
    ok = r.category == expected
    status = "OK  " if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  {status}  [{r.category:22s}] {repr(text)}")

print(f"\n  {passed}/{len(cases)} passed, {failed} failed")
assert failed == 0, f"{failed} intent routing test(s) failed!"
print("  ALL TESTS PASSED")
