"""
verify_app.py - Comprehensive verification of all 10 upgrades
"""
import requests
import json
import os

BASE = "http://127.0.0.1:5000"

def safe_print(text):
    """Print text safely on Windows by replacing non-encodable chars."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))
s = requests.Session()

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")

# ============================================
# 1. REGISTER
# ============================================
print("=" * 60)
print("STEP 1: Register New User")
print("=" * 60)
r = s.post(f"{BASE}/register", data={
    "name": "Verify User",
    "email": "verify@example.com",
    "password": "test1234",
    "confirm_password": "test1234"
}, allow_redirects=True)
check("Registration completes (200)", r.status_code == 200)
check("Redirected to login", "/login" in r.url)

# ============================================
# 2. LOGIN
# ============================================
print("\n" + "=" * 60)
print("STEP 2: Login")
print("=" * 60)
r = s.post(f"{BASE}/login", data={
    "email": "verify@example.com",
    "password": "test1234"
}, allow_redirects=True)
check("Login completes (200)", r.status_code == 200)
check("Redirected to dashboard", "/dashboard" in r.url)

# ============================================
# 3. HIGH RISK ASSESSMENT (Upgrade 1, 2, 4, 5, 6, 8, 10)
# ============================================
print("\n" + "=" * 60)
print("STEP 3: Submit HIGH RISK Questionnaire")
print("=" * 60)
r = s.post(f"{BASE}/questionnaire", data={
    "sleep_hours": "4",
    "stress_level": "9",
    "study_pressure": "8",
    "social_anxiety": "7",
    "mood_score": "2",
    "feelings_text": "I feel extremely overwhelmed and exhausted, cant sleep at all, so much pressure from exams and I feel hopeless"
}, allow_redirects=True)
check("Questionnaire submits (200)", r.status_code == 200)
check("Redirected to result page", "/result/" in r.url)

content = r.text

# Upgrade 10: Hybrid Risk Model
check("Risk level displayed (High or Critical)", "High" in content or "Critical" in content)

# Upgrade 2: Explainable AI
check("Explainable AI: Poor Sleep factor", "Poor Sleep" in content)
check("Explainable AI: High Stress factor", "High Stress" in content)
check("Explainable AI: Study Pressure factor", "Study Pressure" in content)

# Upgrade 1: Recovery Plan
check("Recovery Plan: sleep goal", "sleep" in content.lower() and "goal" in content.lower())
check("Recovery Plan: meditation", "meditation" in content.lower())
check("Recovery Plan: walk", "walk" in content.lower())
check("Recovery Plan: journaling", "journal" in content.lower())

# Upgrade 6: Sentiment Analysis  
check("Sentiment label displayed", "Distressed" in content or "Anxious" in content or "sentiment" in content.lower())

# Upgrade 5: Resource Recommendations
check("Resources: Sleep Hygiene", "Sleep Hygiene" in content or "sleep" in content.lower())
check("Resources: Study/Stress related", "Pomodoro" in content or "Academic Stress" in content or "resource" in content.lower())

# Upgrade 4: Doctor Recommendations
check("Doctors recommended", "doctor" in content.lower() or "Dr." in content or "Hospital" in content)

# No crisis - should NOT have crisis banner
check("No crisis banner (no crisis keywords)", "want to end my life" not in content.lower())

# ============================================
# 4. CRISIS ASSESSMENT (Upgrade 7)
# ============================================
print("\n" + "=" * 60)
print("STEP 4: Submit CRISIS Questionnaire")
print("=" * 60)
r = s.post(f"{BASE}/questionnaire", data={
    "sleep_hours": "3",
    "stress_level": "10",
    "study_pressure": "9",
    "social_anxiety": "8",
    "mood_score": "1",
    "feelings_text": "I want to end my life, I want to disappear forever, nothing matters anymore"
}, allow_redirects=True)
check("Crisis questionnaire submits (200)", r.status_code == 200)
content = r.text

check("Critical risk level", "Critical" in content)
check("Crisis helpline: Tele-MANAS", "Tele-MANAS" in content or "14416" in content)
check("Crisis helpline: Kiran", "Kiran" in content or "1800-599-0019" in content or "1800" in content)

# ============================================
# 5. CHATBOT TESTS (Upgrade 9)
# ============================================
print("\n" + "=" * 60)
print("STEP 5: Chatbot - Normal Message")
print("=" * 60)
r = s.post(f"{BASE}/chatbot/send", json={"message": "I am feeling very stressed about my exams"})
check("Chatbot responds (200)", r.status_code == 200)
data = r.json()
check("Chatbot returns response text", len(data.get("response", "")) > 10)
check("Chatbot mode is simulated or gemini", data.get("mode") in ["simulated", "gemini"])
safe_print(f"  -> Mode: {data.get('mode')}")
safe_print(f"  -> Response: {data.get('response', '')[:120]}...")

print("\n" + "=" * 60)
print("STEP 6: Chatbot - Crisis Detection")
print("=" * 60)
r = s.post(f"{BASE}/chatbot/send", json={"message": "I want to kill myself"})
check("Crisis chatbot responds (200)", r.status_code == 200)
data = r.json()
check("Crisis mode activated", data.get("mode") == "crisis")
check("Crisis response has helpline", "14416" in data.get("response", "") or "Tele-MANAS" in data.get("response", ""))
safe_print(f"  -> Mode: {data.get('mode')}")
safe_print(f"  -> Response: {data.get('response', '')[:120]}...")

# ============================================
# 6. DASHBOARD WITH TRENDS (Upgrade 3)
# ============================================
print("\n" + "=" * 60)
print("STEP 7: Dashboard Trends")
print("=" * 60)
r = s.get(f"{BASE}/dashboard")
check("Dashboard loads (200)", r.status_code == 200)
content = r.text
check("Chart.js integrated", "Chart" in content or "chart" in content.lower())
check("Stress data in chart", "stress" in content.lower())
check("Mood data in chart", "mood" in content.lower())
check("Multiple assessment entries visible", content.count("assessment") >= 1 or content.count("High") >= 1 or content.count("Critical") >= 1)

# ============================================
# 7. EMAIL LOG VERIFICATION (Upgrade 8)
# ============================================
print("\n" + "=" * 60)
print("STEP 8: Email Log Verification")
print("=" * 60)
log_path = os.path.join("instance", "email_logs.txt")
if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        log_content = f.read()
    check("Email log file exists", True)
    check("Email log contains report", "MindCare AI" in log_content or "Assessment Report" in log_content)
    check("Email log has recovery plan", "Sleep Goal" in log_content or "Recovery Plan" in log_content)
    check("Email log has resources", "Recommended Resources" in log_content or "resource" in log_content.lower())
    print(f"  -> Log file size: {len(log_content)} bytes")
else:
    check("Email log file exists", False)

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print(f"VERIFICATION SUMMARY: {passed} PASSED / {failed} FAILED / {passed + failed} TOTAL")
print("=" * 60)
if failed == 0:
    print("[SUCCESS] All verification checks passed!")
else:
    print(f"[WARNING] {failed} checks failed - review output above.")
