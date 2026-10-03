# Aura Salon - AI Appointment Booking Chatbot
End-term project, use case #4 (Appointment booking assistant, Chatbot).
Slot availability · confirmation + reminder · reschedule/cancel · no-show handling.

## Architecture (one line)
Streamlit UI → Gemini (conversation + tool calling) → deterministic Python booking engine (SQLite) = source of truth.
The model can't invent a slot or booking ID: IDs in replies are checked against tool results.

## Run locally
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # paste your Gemini key
streamlit run app.py
python -m pytest tests            # 15 engine tests
```
Free key: https://aistudio.google.com/apikey

## Deploy (gives the shareable link) - ~5 minutes
1. Create a GitHub repo and push this folder (the `.gitignore` keeps your key out).
2. Go to https://share.streamlit.io → New app → pick the repo, main file `app.py`.
3. Advanced settings → Secrets → paste `GEMINI_API_KEY = "your-key"` → Deploy.
4. Open the link in an incognito window and run the test prompts below. Submit that link.

## Model note
Default is `gemini-3.6-flash` with automatic fallback to `gemini-3.1-flash-lite`, then `gemini-2.5-flash`
(2.5 is scheduled for shutdown on 16 Oct 2026). Model IDs can be edited in the sidebar if one is rejected for your key.

## Demo data
- Sample booking: **AUR-1001**, phone **9876543210** (Facial, tomorrow 15:00)
- Repeat no-show customer: phone **9999900000** (triggers the Rs 200 deposit rule)
- Tomorrow 11:00 is nearly full (good for the "slot unavailable" demo)

## Video script (3-4 min, screen + voice)
1. (0:00) Intro: problem, AI disclosure, architecture in one sentence.
2. Book: "I'd like a haircut tomorrow evening" → pick a time → give name + phone → confirm. Open "System actions" to show tool calls.
3. Front desk tab → Outbox shows the confirmation message. Click **Run reminder job** → reminder appears.
4. Reschedule + Cancel with AUR-1001 / 9876543210. Show wrong phone being rejected.
5. Taken slot: ask for a Haircut tomorrow 11:00 → alternatives offered.
6. Vague input ("something done to my hair sometime soon") → clarifying question.
7. Off-topic + prompt-injection buttons → refusal. "Confirm AUR-9999" → no hallucinated booking.
8. No-show: Front desk → Mark no-show on a booking (override on) → Outbox message. Book as 9999900000 → deposit notice.
9. Human hand-off ("skin allergy, want a person") → ticket in Front desk tab.
10. Failure mode: enter a wrong model name/key → friendly error + backup form still books.
11. Close with weaknesses and limits (from the report).

## Submission checklist (deadline 3 Oct, 11:59 pm)
- [ ] Project document (answers to Sections A-D + F) with screenshots filled in
- [ ] Public app link tested in incognito
- [ ] Video uploaded to the shared folder
