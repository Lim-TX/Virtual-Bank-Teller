import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import streamlit.components.v1 as components
import time
import speech_recognition as sr
import whisper
import edge_tts
import asyncio
import re
import json
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import torch
import concurrent.futures
from teller_helpers import write_teller_state, cleanup_old_tts_files
from action_registry import ACTION_REGISTRY, get_action, is_workflow, is_modal
from workflows.send_money import render_send_money
from workflows.pay_card import render_pay_card
from workflows.pay_loan import render_pay_loan
from workflows.pay_bill import render_pay_bill
from workflows.modals import show_promotions, show_mailbox, show_account_summary, show_maintenance

# ── Perf logging ── set False to silence all timing output ──────────────────
PERF_LOG = True
def _perf(label: str, elapsed: float | None = None) -> None:
    if not PERF_LOG:
        return
    if elapsed is None:
        print(f"[PERF] {label}", flush=True)
    else:
        print(f"[PERF] {label}: {elapsed*1000:.0f} ms", flush=True)
# ────────────────────────────────────────────────────────────────────────────

# Ensure static directories exist
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
STATE_FILE = os.path.join(STATIC_DIR, "state.json")

# Initialize state.json with the formal teller state contract
write_teller_state(STATE_FILE, "idle", "")

# Initialize core models
@st.cache_resource
def load_models():
    # Detect if CUDA is available for the 5070; fallback to CPU otherwise
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stt = whisper.load_model("tiny.en", device=device)
    llm = OllamaLLM(model="hf.co/MaziyarPanahi/Qwen3-4B-GGUF:Q4_K_M", temperature=0.1)
    return stt, llm

stt_model, llm = load_models()

# ── LLM warmup — fires a minimal inference in the background so the GGUF
#    model is resident in VRAM before the first real user message arrives.
def _warmup_llm():
    try:
        llm.invoke("hi")
    except Exception:
        pass

concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(_warmup_llm)

# ── Mock account data — single source of truth for all teller responses ─────
MOCK_ACCOUNT = {
    "name": "John Doe",
    "account_type": "Current Account",
    "account_number": "1234-5678-9012",
    "balance": "RM 12,480.50",
    "last_login": "17 Apr 2026, 09:14:03 AM (GMT+8)",
    "credit_card": {
        "number": "**** **** **** 4321",
        "outstanding_balance": "RM 1,250.00",
        "minimum_payment": "RM 50.00",
        "due_date": "25 Apr 2026",
        "credit_limit": "RM 10,000.00",
        "available_credit": "RM 8,750.00",
    },
    "loan": {
        "type": "Home Financing",
        "outstanding_balance": "RM 185,300.00",
        "monthly_instalment": "RM 1,120.00",
        "next_due_date": "1 May 2026",
        "remaining_tenure": "18 years",
        "payment_options": ["Full instalment", "Partial payment (min RM 500)", "Extra payment to reduce principal"],
    },
    "send_money": {
        "daily_limit": "RM 50,000.00",
        "per_transaction_limit": "RM 10,000.00",
        "used_today": "RM 0.00",
        "remaining_today": "RM 50,000.00",
        "supported_banks": ["Maybank", "CIMB", "Public Bank", "RHB", "Hong Leong", "AmBank", "BSN", "Bank Rakyat", "OCBC", "Standard Chartered"],
        "fee": "Free for IBG transfers; Instant Transfer (DuitNow) is free",
        "transfer_types": ["DuitNow (by phone/IC/account number)", "Interbank Giro (IBG)", "Own account transfer"],
    },
    "pay_card": {
        "payment_options": [
            {"option": "Minimum payment", "amount": "RM 50.00"},
            {"option": "Full outstanding balance", "amount": "RM 1,250.00"},
            {"option": "Custom amount", "amount": "Any amount between RM 50.00 and RM 1,250.00"},
        ],
        "payment_source": "Current Account 1234-5678-9012",
        "processing_time": "Payments reflect within 1 business day",
        "autopay_status": "Not enrolled — you can set up AutoPay in card settings",
    },
    "pay_loan": {
        "payment_options": [
            {"option": "Monthly instalment", "amount": "RM 1,120.00"},
            {"option": "Partial payment", "amount": "Minimum RM 500.00"},
            {"option": "Full settlement", "amount": "RM 185,300.00 (subject to early settlement fee)"},
        ],
        "payment_source": "Current Account 1234-5678-9012",
        "early_settlement_fee": "1% of outstanding balance",
        "processing_time": "Payments reflect within 1 business day",
    },
    "pay_bill": {
        "registered_billers": [
            {"biller": "Tenaga Nasional Berhad (TNB)", "account_ref": "7701234567", "amount_due": "RM 112.40", "due": "20 Apr 2026"},
            {"biller": "Syabas (Water)", "account_ref": "AJ3345678", "amount_due": "RM 34.80", "due": "22 Apr 2026"},
            {"biller": "Unifi Broadband", "account_ref": "0312345678", "amount_due": "RM 149.00", "due": "28 Apr 2026"},
        ],
        "payment_source": "Current Account 1234-5678-9012",
        "processing_time": "Bill payments are processed instantly",
        "new_biller": "You can add a new biller by selecting 'Add Biller' and entering the biller code and reference number",
    },
    "recent_transactions": [
        {"date": "13 Apr 2026 07:55 PM", "description": "Card Alert — Petronas KLCC", "amount": "-RM 80.00"},
        {"date": "10 Apr 2026 09:08 PM", "description": "Card Transaction — Grab Food", "amount": "-RM 32.50"},
        {"date": "09 Apr 2026 02:26 PM", "description": "Card Payment Received", "amount": "+RM 500.00"},
        {"date": "02 Apr 2026 01:53 AM", "description": "Card Transaction — Shopee", "amount": "-RM 145.90"},
        {"date": "01 Apr 2026 07:44 PM", "description": "Card Alert — TnG eWallet Top-Up", "amount": "-RM 100.00"},
    ],
    "pending_bills": [
        {"biller": "Tenaga Nasional Berhad (TNB)", "amount": "RM 112.40", "due": "20 Apr 2026"},
        {"biller": "Syabas (Water)", "amount": "RM 34.80", "due": "22 Apr 2026"},
        {"biller": "Unifi Broadband", "amount": "RM 149.00", "due": "28 Apr 2026"},
    ],
    "savings_account": {
        "account_number": "9876-5432-1098",
        "balance": "RM 5,210.00",
        "interest_rate": "2.0% p.a.",
    },
    "promotions": [
        {
            "title": "Grand Rewards Programme",
            "description": "Earn 2x reward points on all retail purchases made with your VirtualBank credit card until 30 June 2026. Redeem points for cashback, air miles, or merchandise at grandrewards.virtualbank.com.",
            "valid_until": "30 Jun 2026",
        },
        {
            "title": "Cashback & Petrol Deals",
            "description": "Get 5% cashback on petrol purchases at Shell, Petronas, and BHPetrol when you pay with your VirtualBank credit card. Capped at RM 30 cashback per month. Auto-credited within 5 business days.",
            "valid_until": "31 May 2026",
        },
    ],
    "maintenance_notice": "VirtualBank will be temporarily unavailable on 5, 6, 7, 11, 12 May 2026 from 12:00 am to 6:00 am.",
}

def _build_account_context() -> str:
    a = MOCK_ACCOUNT
    txn_lines = "\n".join(
        f"  - {t['date']}: {t['description']} ({t['amount']})"
        for t in a["recent_transactions"]
    )
    bill_lines = "\n".join(
        f"  - {b['biller']} (ref {b['account_ref']}): {b['amount_due']} due {b['due']}"
        for b in a["pay_bill"]["registered_billers"]
    )
    promo_lines = "\n".join(
        f"  - {p['title']}: {p['description']} (valid until {p['valid_until']})"
        for p in a["promotions"]
    )
    pay_card_opts = "\n".join(
        f"  - {o['option']}: {o['amount']}"
        for o in a["pay_card"]["payment_options"]
    )
    pay_loan_opts = "\n".join(
        f"  - {o['option']}: {o['amount']}"
        for o in a["pay_loan"]["payment_options"]
    )
    sm = a["send_money"]
    return f"""You are Haru, a helpful and friendly virtual bank teller for VirtualBank.
You have access to the following real account data for the logged-in customer. Use it to answer questions accurately and naturally.
Keep answers concise and readable. For single facts, reply in 1-2 short sentences. For multiple items (e.g. promotions, account details, options), use short bullet points (- item). Do not use headers, bold, tables, or emojis. Speak conversationally.
If asked something outside your knowledge or unrelated to banking, politely decline.

PERSONALITY AND TONE:
- You are warm, approachable, and emotionally aware — not robotic or clinical.
- For greetings and confirmations: be genuinely upbeat and friendly (e.g. "Happy to help!" or "Great, all set!").
- For promotions, rewards, or savings: show genuine enthusiasm (e.g. "Great news — you can earn 2x points right now!").
- For concerns like overdue bills or insufficient balance: lead with empathy before facts (e.g. "I understand that's a bit urgent — here's what I can see...").
- For standard transactional queries: be calm, clear, and professional.
- Never use hollow filler phrases like "Certainly!" or "Of course!" alone — always follow immediately with useful information.

IMPORTANT — always refer to features by their exact dashboard button names:
  - Use "Send Money" (never "transfer" or "remittance")
  - Use "Pay Card" (never "credit card payment" or "card repayment")
  - Use "Pay Loan / Financing" (never "loan repayment" or "instalment payment")
  - Use "Pay Bill" (never "utility payment" or "bill settlement")
When guiding the customer to perform an action, embed a clickable action link using this exact format: [ACTION:send_money:Send Money], [ACTION:pay_card:Pay Card], [ACTION:pay_loan:Pay Loan / Financing], [ACTION:pay_bill:Pay Bill].
Example: "You can tap [ACTION:send_money:Send Money] to send funds to another account."
Only embed a link when you are directing the customer to use that specific feature. Do not embed more than one link per response.
IMPORTANT: When the customer provides enough details to complete a Send Money transfer (recipient phone/IC, amount), ALWAYS include [ACTION:send_money:Send Money] in your response so they can proceed directly. If recipient or amount is missing, ask for the missing detail in a single short question.

IMPORTANT: When the customer wants to Pay Card and has specified the payment option (minimum payment / full balance / a custom RM amount), ALWAYS include [ACTION:pay_card:Pay Card]. If they have not specified, ask: "How much would you like to pay — minimum (RM 50.00), full outstanding balance (RM 1,250.00), or a custom amount?"

IMPORTANT: When the customer wants to Pay Loan / Financing and has specified the payment type (monthly instalment / partial payment / full settlement), ALWAYS include [ACTION:pay_loan:Pay Loan / Financing]. If they have not specified, ask: "Which payment type — monthly instalment (RM 1,120.00), a partial payment (min RM 500.00), or full settlement?"

IMPORTANT: When the customer wants to Pay Bill and has specified which biller (TNB / Syabas / Unifi), ALWAYS include [ACTION:pay_bill:Pay Bill]. If they have not specified, ask: "Which bill — TNB (RM 112.40), Syabas (RM 34.80), or Unifi (RM 149.00)?"

CUSTOMER: {a['name']}
CURRENT ACCOUNT: {a['account_number']} | Balance: {a['balance']}
SAVINGS ACCOUNT: {a['savings_account']['account_number']} | Balance: {a['savings_account']['balance']} | Rate: {a['savings_account']['interest_rate']}

CREDIT CARD ({a['credit_card']['number']}):
  Outstanding: {a['credit_card']['outstanding_balance']} | Min payment: {a['credit_card']['minimum_payment']} | Due: {a['credit_card']['due_date']}
  Credit limit: {a['credit_card']['credit_limit']} | Available: {a['credit_card']['available_credit']}

LOAN ({a['loan']['type']}):
  Outstanding: {a['loan']['outstanding_balance']} | Monthly instalment: {a['loan']['monthly_instalment']}
  Next due: {a['loan']['next_due_date']} | Remaining tenure: {a['loan']['remaining_tenure']}

SEND MONEY:
  Daily limit: {sm['daily_limit']} | Per-transaction limit: {sm['per_transaction_limit']}
  Used today: {sm['used_today']} | Remaining today: {sm['remaining_today']}
  Fee: {sm['fee']}
  Transfer types: {', '.join(sm['transfer_types'])}
  Supported banks: {', '.join(sm['supported_banks'])}

PAY CREDIT CARD OPTIONS:
{pay_card_opts}
  Source account: {a['pay_card']['payment_source']} | Processing: {a['pay_card']['processing_time']}
  AutoPay: {a['pay_card']['autopay_status']}

PAY LOAN OPTIONS:
{pay_loan_opts}
  Source account: {a['pay_loan']['payment_source']} | Early settlement fee: {a['pay_loan']['early_settlement_fee']}

PAY BILLS (registered billers):
{bill_lines}
  Processing: {a['pay_bill']['processing_time']}

RECENT TRANSACTIONS:
{txn_lines}

CURRENT PROMOTIONS:
{promo_lines}

SYSTEM NOTICE: {a['maintenance_notice']}"""

# Build the prompt with injected account context
prompt_template = ChatPromptTemplate.from_messages([
    ("system", _build_account_context()),
    ("user", "{input}")
])

chain = prompt_template | llm

def clean_text(text):
    # Strip bullet-point prefixes so TTS doesn't read out "dash item"
    text = re.sub(r'^\s*-\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s.,!?\'"-\[\]:]', '', text)
    return text.strip()

# Maps ACTION tokens emitted by the LLM to query-param keys — driven by registry
def render_action_links(text: str) -> str:
    """Replace [ACTION:key:Label] tokens with styled HTML anchor tags.
    Uses data-nav attribute — click handled by the parent-context JS injector
    so React error #231 (caused by onclick string handlers) is avoided."""
    _workflow_keys = {k for k, v in ACTION_REGISTRY.items() if v["surface"] == "workflow"}

    def _replace(m: re.Match) -> str:
        key, label = m.group(1), m.group(2)
        if key not in _workflow_keys:
            return label
        return (
            f'<a class="chat-action-link" data-nav="{key}" style="cursor:pointer;">'
            f'{label}</a>'
        )
    return re.sub(r'\[ACTION:(\w+):([^\]]+)\]', _replace, text)

# Transfer-type aliases → match to the exact option labels in the selectbox
_TRANSFER_TYPE_ALIASES = {
    "account transfer":      "Interbank Giro (IBG)",
    "ibg":                   "Interbank Giro (IBG)",
    "interbank giro":        "Interbank Giro (IBG)",
    "duitnow":               "DuitNow (by phone/IC/account number)",
    "instant transfer":      "DuitNow (by phone/IC/account number)",
    "ibft":                  "DuitNow (by phone/IC/account number)",
    "own account":           "Own account transfer",
    "own account transfer":  "Own account transfer",
    "self transfer":         "Own account transfer",
}

def _parse_send_money_prefill(text: str) -> dict | None:
    """Extract Send Money form fields from a natural-language message.
    Returns a dict with any subset of: recipient, amount, note, type, source.
    Returns None if no useful params found."""
    t = text.lower()
    result = {}

    # Recipient: phone number (Malaysian 01x format, optionally hyphenated), IC, or account number
    phone = re.search(r'\b(01[0-9][-\s]?[0-9]{7,8})\b', text)
    if phone:
        result["recipient"] = re.sub(r'[-\s]', '', phone.group(1))
    else:
        ic = re.search(r'\b(\d{6}-\d{2}-\d{4})\b', text)
        if ic:
            result["recipient"] = ic.group(1)
        else:
            acct = re.search(r'\b(\d{10,16})\b', text)
            if acct:
                result["recipient"] = acct.group(1)

    # Amount: RM 50, 50 ringgit, 50rm
    amt = re.search(r'rm\s*(\d+(?:\.\d{1,2})?)', t)
    if not amt:
        amt = re.search(r'(\d+(?:\.\d{1,2})?)\s*(?:ringgit|rm)', t)
    if amt:
        result["amount"] = float(amt.group(1))

    # Note: "note (that) says X", "with a note X", "note: X", "reference X"
    note = re.search(
        r'(?:note(?:\s+that)?\s+(?:says?|reads?|is)?\s*["\']?|with\s+(?:a\s+)?note\s*["\']?|reference\s*["\']?)([A-Za-z0-9 ]+)',
        t
    )
    if note:
        result["note"] = note.group(1).strip().title()

    # Transfer type
    for alias, canonical in _TRANSFER_TYPE_ALIASES.items():
        if alias in t:
            result["type"] = canonical
            break

    # Source account
    if "current" in t:
        result["source"] = "Current Account  1234-5678-9012"
    elif "saving" in t:
        result["source"] = "Savings Account  9876-5432-1098"

    return result if result else None


def _parse_pay_card_prefill(text: str) -> dict | None:
    """Extract Pay Card form fields from a natural-language message.
    Returns a dict with any subset of: option, custom.
    Returns None if no useful params found."""
    t = text.lower()
    result = {}
    if any(w in t for w in ["minimum", "min payment", "minimum payment"]):
        result["option"] = "Minimum payment  —  RM 50.00"
    elif any(w in t for w in ["full balance", "full outstanding", "full amount", "everything", "all of it"]):
        result["option"] = "Full outstanding balance  —  RM 1,250.00"
    elif "full" in t and "pay" in t:
        result["option"] = "Full outstanding balance  —  RM 1,250.00"
    else:
        amt = re.search(r'rm\s*(\d+(?:\.\d{1,2})?)', t)
        if not amt:
            amt = re.search(r'(\d+(?:\.\d{1,2})?)\s*(?:ringgit|rm)', t)
        if amt:
            result["option"] = "Custom amount  —  Any amount between RM 50.00 and RM 1,250.00"
            result["custom"] = float(amt.group(1))
    return result if result else None


def _parse_pay_loan_prefill(text: str) -> dict | None:
    """Extract Pay Loan form fields from a natural-language message.
    Returns a dict with any subset of: option, partial.
    Returns None if no useful params found."""
    t = text.lower()
    result = {}
    if any(w in t for w in ["full settlement", "settle in full", "pay off", "payoff", "full pay"]):
        result["option"] = "Full settlement  —  RM 185,300.00 (subject to early settlement fee)"
    elif any(w in t for w in ["partial", "extra"]):
        result["option"] = "Partial payment  —  Minimum RM 500.00"
        amt = re.search(r'rm\s*(\d+(?:\.\d{1,2})?)', t)
        if not amt:
            amt = re.search(r'(\d+(?:\.\d{1,2})?)\s*(?:ringgit|rm)', t)
        if amt:
            result["partial"] = float(amt.group(1))
    elif any(w in t for w in ["monthly", "instalment", "installment", "regular", "normal"]):
        result["option"] = "Monthly instalment  —  RM 1,120.00"
    return result if result else None


# Maps recognisable biller keywords → exact biller name from MOCK_ACCOUNT
_BILLER_ALIASES = {
    "Tenaga Nasional Berhad (TNB)": ["tnb", "tenaga", "electricity", "electric", "tenaga nasional"],
    "Syabas (Water)":               ["syabas", "water bill", "water", "air"],
    "Unifi Broadband":              ["unifi", "broadband", "internet", "tm unifi", "tmunifi"],
}

def _parse_pay_bill_prefill(text: str) -> dict | None:
    """Extract Pay Bill form fields from a natural-language message.
    Returns a dict with any subset of: biller_name, pay_full, amount.
    Returns None if no useful params found."""
    t = text.lower()
    result = {}
    for biller_name, keywords in _BILLER_ALIASES.items():
        if any(kw in t for kw in keywords):
            result["biller_name"] = biller_name
            break
    # Custom amount overrides pay-full
    amt = re.search(r'rm\s*(\d+(?:\.\d{1,2})?)', t)
    if not amt:
        amt = re.search(r'(\d+(?:\.\d{1,2})?)\s*(?:ringgit|rm)', t)
    if amt:
        result["amount"] = float(amt.group(1))
        result["pay_full"] = False
    elif result:
        result["pay_full"] = True
    return result if result else None


# ── Emotion-aware TTS prosody ─────────────────────────────────────────────────
# Maps a detected emotional tone to edge-tts prosody rate/pitch adjustments.
# AriaNeural responds noticeably to these — cheerful/excited raise pitch & tempo,
# empathetic slows and softens, default is neutral customerservice delivery.
_EMOTION_PROSODY: dict[str, dict[str, str]] = {
    "cheerful":   {"rate": "+12%", "pitch": "+2Hz"},
    "empathetic": {"rate": "-10%", "pitch": "-1Hz"},
    "excited":    {"rate": "+20%", "pitch": "+4Hz"},
    "default":    {"rate": "+0%",  "pitch": "+0Hz"},
}

# Order matters — more specific patterns checked first
_EMOTION_KEYWORDS: dict[str, list[str]] = {
    "excited":    ["promotion", "cashback", "reward", "earn", "deal", "offer",
                   "benefit", "points", "save ", "discount", "prize"],
    "empathetic": ["sorry", "apologize", "unfortunately", "unable to",
                   "understand your", "i see that", "i'm sorry", "concern",
                   "that must", "i can see", "i understand"],
    "cheerful":   ["happy to help", "great", "sure!", "absolutely", "of course",
                   "glad", "welcome", "hi!", "hello", "good news",
                   "perfect", "no problem", "all set"],
}

def _detect_emotion(text: str) -> str:
    """Classify the emotional tone of a teller response for TTS prosody selection.
    Returns one of: 'cheerful', 'empathetic', 'excited', 'default'."""
    t = text.lower()
    for emotion, keywords in _EMOTION_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return emotion
    return "default"


def record_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.toast("Listening... Speak now.", icon="🎤")
        r.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            return None
    
    with open("temp_input.wav", "wb") as f:
        f.write(audio.get_wav_data())
    return "temp_input.wav"

async def generate_tts_and_update_state(text, max_retries=3):
    # Clean up old files before generating a new one (keep 3 most recent)
    cleanup_old_tts_files(STATIC_DIR, keep=3)

    timestamp = int(time.time() * 1000)
    filename = f"response_{timestamp}.mp3"
    filepath = os.path.join(STATIC_DIR, filename)

    # Select prosody based on the emotional tone of this sentence
    emotion = _detect_emotion(text)
    prosody = _EMOTION_PROSODY[emotion]

    for attempt in range(max_retries):
        try:
            communicate = edge_tts.Communicate(
                text, "en-US-AriaNeural",
                rate=prosody["rate"],
                pitch=prosody["pitch"],
            )
            await communicate.save(filepath)
            write_teller_state(STATE_FILE, "speaking", filename)
            return filepath
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to generate TTS after {max_retries} attempts: {e}")
                raise
            else:
                print(f"TTS API Error ({e}). Retrying... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(0.5 * (attempt + 1))

def run_tts_sync(text):
    asyncio.run(generate_tts_and_update_state(text))

# Read the local pixi-live2d-display script to bypass strict MIME checking
live2d_js_path = os.path.join(STATIC_DIR, "cubism4.min.js")
try:
    with open(live2d_js_path, "r", encoding="utf-8") as f:
        live2d_display_script = f.read()
except FileNotFoundError:
    live2d_display_script = ""

# --- HTML/JS Component for Live2D WebGL Avatar ---
LIVE2D_HTML = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { margin: 0; padding: 0; overflow: hidden; display: flex; justify-content: center; align-items: center; background: transparent; }
        #wrapper { 
            position: relative; width: 460px; height: 460px;
            background: transparent; overflow: hidden;
            display: flex; justify-content: center; align-items: flex-end;
        }
        canvas { display: block; }

    </style>
    <!-- Cache Buster: v9 -->
    <script src="https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/pixi.js@6.5.2/dist/browser/pixi.min.js"></script>
    <script>window.PIXI = PIXI;</script>
    <!-- LIVE2D_DISPLAY_SCRIPT_MOUNT -->
</head>
<body>
    <div id="wrapper">
        <canvas id="live2d-canvas" width="460" height="460"></canvas>
    </div>

    <script>
        window.PIXI = PIXI;

        // srcdoc iframes have opaque origin; use parent frame's origin instead
        const BASE_URL = parent.location.origin;

        // Curated motion index sets (0-based index into model3.json Motions[""] array)
        // Index 0 = haru_g_idle (10s looping) — used as the return-to-idle anchor
        // Only motions ≤ 2.53s used for thinking (brief, unobtrusive).
        // Talking motions are intentionally omitted so lip-sync runs uninterrupted.
        // Durations: m02=2.03s, m05=2.03s, m13=2.53s
        const THINKING_MOTIONS = [2, 5, 13];
        const IDLE_MOTION_IDX  = 0;   // haru_g_idle

        const AUDIO_QUEUE_LIMIT = 3;

        let app, model;
        let lastAudio = "";
        let lastTellerState = "";
        let audioCtx;
        let audioQueue = [];
        let isPlaying = false;
        let smoothedMouth = 0;
        let blinkTimer = null;
        let thinkingTimeout = null;  // cancelled when speaking starts

        // Auto-initialize model immediately on page load (no user gesture needed)
        window.addEventListener('DOMContentLoaded', async () => { await init(); });

        // Initialize AudioContext when user first interacts with the parent page
        // (allow-same-origin lets us add a listener to the parent document)
        try {
            window.parent.document.addEventListener('click', async function initAudioOnce() {
                if (!audioCtx) {
                    const AudioContext = window.AudioContext || window.webkitAudioContext;
                    audioCtx = new AudioContext();
                }
                if (audioCtx.state !== 'running') { await audioCtx.resume(); }
                // Keep listening so resume() is re-called if context gets suspended again
            });
        } catch(e) {
            // Cross-origin fallback: init on canvas click
            document.getElementById('live2d-canvas').addEventListener('click', async function() {
                if (!audioCtx) {
                    const AudioContext = window.AudioContext || window.webkitAudioContext;
                    audioCtx = new AudioContext();
                    await audioCtx.resume();
                } else if (audioCtx.state !== 'running') { await audioCtx.resume(); }
            });
        }

        async function init() {
            // Step 5: backgroundAlpha replaces deprecated transparent:true
            app = new PIXI.Application({
                view: document.getElementById('live2d-canvas'),
                backgroundAlpha: 0,
                autoStart: true
            });

            // Step 1: Use origin-relative URL — works on any host/port
            const modelUrl = BASE_URL + "/app/static/live2d/model/haru_greeter/haru_greeter_t05.model3.json";

            try {
                model = await PIXI.live2d.Live2DModel.from(modelUrl);
                app.stage.addChild(model);
                model.scale.set(0.28);
                model.x = 55;
                model.y = 140;

                startBlinkLoop();

                // Step 4: Poll at 300ms — balances responsiveness with request volume
                setInterval(pollState, 300);
            } catch (err) {
                console.error("Failed to load Live2D model:", err);
                document.getElementById('wrapper').style.cssText +=
                    ';display:flex;justify-content:center;align-items:center;color:#EF4444;font-size:14px;';
                document.getElementById('live2d-canvas').style.display = 'none';
                const errEl = document.createElement('div');
                errEl.textContent = 'Avatar unavailable';
                document.getElementById('wrapper').appendChild(errEl);
            }
        }

        // Step 3: Autonomous blink loop
        function startBlinkLoop() {
            function scheduleBlink() {
                // Random interval between 2s and 6s
                const delay = 2000 + Math.random() * 4000;
                blinkTimer = setTimeout(async () => {
                    await doBlink();
                    scheduleBlink();
                }, delay);
            }
            scheduleBlink();
        }

        async function doBlink() {
            if (!model || !model.internalModel || !model.internalModel.coreModel) return;
            const core = model.internalModel.coreModel;
            const CLOSE_FRAMES = 3;
            const OPEN_FRAMES = 3;
            for (let i = 0; i < CLOSE_FRAMES; i++) {
                const v = 1 - (i + 1) / CLOSE_FRAMES;
                core.setParameterValueById('ParamEyeLOpen', v);
                core.setParameterValueById('ParamEyeROpen', v);
                await new Promise(r => setTimeout(r, 16));
            }
            for (let i = 0; i < OPEN_FRAMES; i++) {
                const v = (i + 1) / OPEN_FRAMES;
                core.setParameterValueById('ParamEyeLOpen', v);
                core.setParameterValueById('ParamEyeROpen', v);
                await new Promise(r => setTimeout(r, 16));
            }
        }

        // Step 2: Poll state.json and react to teller_state transitions
        async function pollState() {
            try {
                const res = await fetch(BASE_URL + "/app/static/state.json?t=" + Date.now());
                const data = await res.json();

                // React to teller state change
                if (data.teller_state && data.teller_state !== lastTellerState) {
                    lastTellerState = data.teller_state;
                    if (data.teller_state === "thinking" && model && model.internalModel) {
                        const idx = THINKING_MOTIONS[Math.floor(Math.random() * THINKING_MOTIONS.length)];
                        model.motion("", idx, 2);
                        // Cancel any previous pending idle return, then schedule a new one
                        clearTimeout(thinkingTimeout);
                        thinkingTimeout = setTimeout(() => {
                            // Only snap to idle if we're not currently speaking
                            if (!isPlaying && model && model.internalModel)
                                model.motion("", IDLE_MOTION_IDX, 1);
                        }, 2600);
                    } else if (data.teller_state === "speaking") {
                        // Cancel any pending thinking-idle return so it doesn't fire mid-lipsync
                        clearTimeout(thinkingTimeout);
                        thinkingTimeout = null;
                    } else if (data.teller_state === "idle" && model && model.internalModel) {
                        clearTimeout(thinkingTimeout);
                        thinkingTimeout = null;
                        model.motion("", IDLE_MOTION_IDX, 1);
                    }
                }

                // React to new audio
                if (data.latest_audio && data.latest_audio !== lastAudio) {
                    lastAudio = data.latest_audio;
                    const audioUrl = BASE_URL + "/app/static/" + data.latest_audio;
                    // Step 4: Cap queue to avoid unbounded stale audio backlog
                    if (audioQueue.length < AUDIO_QUEUE_LIMIT) {
                        audioQueue.push(audioUrl);
                        processAudioQueue();
                    }
                }
            } catch (e) {
                // Silent: ignore transient polling failures
            }
        }

        async function processAudioQueue() {
            if (isPlaying || audioQueue.length === 0 || !audioCtx) return;
            isPlaying = true;
            const nextAudioUrl = audioQueue.shift();
            await playAudioAndSync(nextAudioUrl);
        }

        async function playAudioAndSync(audioUrl) {
            if (!audioCtx) return;
            try {
                const response = await fetch(audioUrl);
                const arrayBuffer = await response.arrayBuffer();
                const buffer = await audioCtx.decodeAudioData(arrayBuffer);

                const source = audioCtx.createBufferSource();
                source.buffer = buffer;
                const analyser = audioCtx.createAnalyser();
                analyser.fftSize = 256;
                source.connect(analyser);
                analyser.connect(audioCtx.destination);
                source.start(0);

                const dataArray = new Uint8Array(analyser.frequencyBinCount);
                smoothedMouth = 0;

                // Step 3: Lerp-smoothed lip-sync — eliminates jitter
                const lipSyncUpdate = () => {
                    analyser.getByteTimeDomainData(dataArray);
                    let sum = 0;
                    for (let i = 0; i < dataArray.length; i++) {
                        const v = (dataArray[i] - 128) / 128.0;
                        sum += v * v;
                    }
                    const rms = Math.sqrt(sum / dataArray.length);
                    const rawMouth = Math.min(rms * 5.0, 1.0);
                    smoothedMouth = smoothedMouth * 0.6 + rawMouth * 0.4;
                    if (model && model.internalModel && model.internalModel.coreModel) {
                        model.internalModel.coreModel.setParameterValueById('ParamMouthOpenY', smoothedMouth);
                    }
                };

                app.ticker.add(lipSyncUpdate);

                source.onended = () => {
                    app.ticker.remove(lipSyncUpdate);
                    smoothedMouth = 0;
                    if (model && model.internalModel && model.internalModel.coreModel) {
                        model.internalModel.coreModel.setParameterValueById('ParamMouthOpenY', 0);
                    }
                    isPlaying = false;
                    // Only return to idle once all queued audio is drained
                    if (audioQueue.length === 0) {
                        if (model && model.internalModel) model.motion("", IDLE_MOTION_IDX, 1);
                    }
                    processAudioQueue();
                };

            } catch (err) {
                console.error('Audio playback error:', err);
                isPlaying = false;
                processAudioQueue();
            }
        }
    </script>
</body>
</html>
"""

# --- Page config (must be first Streamlit call) ---
st.set_page_config(
    page_title="VirtualBank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Demo credentials — prototype only, not for production ────────────────────
_DEMO_USER_ID = "demo"
_DEMO_PASSWORD = "VirtualBank2026"


def _render_login_page() -> None:
    """Full-page login surface. Calls st.stop() so nothing else renders.

    Layout: position:fixed hero on left 58% (pure HTML/CSS) +
    Streamlit stMain repositioned to right 42% via CSS.
    No st.columns() — avoids the flex-wrap DOM fragmentation issue.
    """
    st.markdown("""
    <style>
        /* ── Strip Streamlit chrome ── */
        [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebar"],
        [data-testid="stDecoration"], #MainMenu, footer, .stDeployButton {
            display: none !important;
        }
        html, body { overflow: hidden !important; background: white !important; }
        .stApp, [data-testid="stAppViewContainer"] {
            background: white !important;
            height: 100vh !important; overflow: hidden !important;
        }

        /* ── Fixed left hero panel (58 vw) ── */
        .login-hero-fixed {
            position: fixed; top: 0; left: 0;
            width: 58vw; height: 100vh;
            background: linear-gradient(135deg, #fff5f5 0%, #fce4e8 40%, #e8d0f0 100%);
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            z-index: 10; overflow: hidden;
        }
        .login-hero-fixed::before {
            content: '';
            position: absolute; inset: 0;
            background:
                radial-gradient(circle at 20% 20%, rgba(200,16,46,0.12) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(100,0,180,0.08) 0%, transparent 50%),
                radial-gradient(circle at 60% 10%, rgba(200,16,46,0.08) 0%, transparent 40%);
            pointer-events: none;
        }
        .login-poly { position: absolute; opacity: 0.18; pointer-events: none; }
        .login-poly-1 {
            top: -60px; right: -40px; width: 260px; height: 260px;
            background: linear-gradient(135deg, #C8102E, #ff6b8a);
            clip-path: polygon(50% 0%, 100% 38%, 82% 100%, 18% 100%, 0% 38%);
        }
        .login-poly-2 {
            bottom: 40px; left: -60px; width: 200px; height: 200px;
            background: linear-gradient(135deg, #C8102E, #e84c6a);
            clip-path: polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%);
        }
        .login-poly-3 {
            top: 30%; right: 30px; width: 120px; height: 120px;
            background: linear-gradient(135deg, #C8102E, #ffb3c1);
            clip-path: polygon(50% 0%, 100% 100%, 0% 100%);
        }
        .login-hero-logo { display: flex; align-items: center; gap: 14px; margin-bottom: 60px; z-index: 1; }
        .login-hero-logo-icon {
            width: 56px; height: 56px; background: #C8102E; border-radius: 14px;
            display: flex; align-items: center; justify-content: center;
        }
        .login-hero-logo-icon svg { fill: white; width: 30px; height: 30px; }
        .login-hero-logo-text { font-size: 28px; font-weight: 800; color: #1A1A1A; }
        .login-hero-logo-text span { color: #C8102E; }
        .login-hero-headline {
            font-size: 68px; font-weight: 900; color: #C8102E;
            line-height: 1.0; letter-spacing: -2px; z-index: 1;
            text-align: center; margin-bottom: 10px;
        }
        .login-hero-sub {
            font-size: 22px; font-weight: 700; color: #444;
            letter-spacing: 2px; text-transform: uppercase; z-index: 1;
            text-align: center; margin-bottom: 16px;
        }
        .login-hero-tag { font-size: 16px; color: #888; z-index: 1; text-align: center; }
        .login-hero-tag b { color: #C8102E; }

        /* ── Right panel: reposition Streamlit's main section ── */
        section[data-testid="stMain"] {
            position: fixed !important;
            top: 0 !important; bottom: 0 !important;
            left: 58vw !important; right: 0 !important;
            width: 42vw !important;          /* override Streamlit width:100% */
            overflow-x: hidden !important;
            overflow-y: auto !important;
            padding: 0 !important;
            background: white !important;
            box-shadow: -4px 0 24px rgba(0,0,0,0.06) !important;
            z-index: 5 !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: stretch !important;
        }
        [data-testid="stMainBlockContainer"] {
            padding: 0 52px !important;
            max-width: 100% !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
            gap: 4px !important;
        }

        /* Panel title */
        .login-panel-title {
            font-size: 26px !important; font-weight: 700 !important;
            color: #1A1A1A !important; margin-bottom: 28px !important;
            text-align: center !important; display: block !important;
            line-height: 1.3 !important; padding: 0 !important;
        }
        /* Inputs */
        [data-testid="stTextInput"] { margin-bottom: 4px !important; }
        [data-testid="stTextInput"] label p {
            font-size: 14px !important; font-weight: 600 !important; color: #555 !important;
        }
        [data-testid="stTextInput"] input {
            border: 1.5px solid #E0E0E0 !important; border-radius: 8px !important;
            font-size: 15px !important; color: #1A1A1A !important;
            background: #FAFAFA !important; padding: 10px 14px !important;
        }
        [data-testid="stTextInput"] input:focus {
            border-color: #C8102E !important; background: white !important;
            box-shadow: 0 0 0 3px rgba(200,16,46,0.10) !important;
        }
        /* Log In button */
        button[data-testid="stBaseButton-primary"] {
            background: #C8102E !important; border: none !important;
            border-radius: 8px !important; font-size: 16px !important;
            font-weight: 700 !important; letter-spacing: 0.5px !important;
            height: 48px !important; margin-top: 8px !important;
        }
        button[data-testid="stBaseButton-primary"]:hover { background: #a00c25 !important; }
        /* Demo hint */
        .login-demo-hint {
            margin-top: 20px; padding: 12px 18px;
            background: #F8F9FF; border: 1px solid #E0E4F5;
            border-radius: 10px; font-size: 13px; color: #666;
            text-align: center; line-height: 1.6;
        }
        .login-demo-hint code { background: #EEF0FF; color: #3B5BDB; padding: 1px 6px; border-radius: 4px; }
        /* Quick links */
        .login-quick { display: flex; gap: 28px; margin-top: 28px; justify-content: center; }
        .login-quick-item { display: flex; flex-direction: column; align-items: center; gap: 8px; cursor: pointer; }
        .login-quick-icon {
            width: 56px; height: 56px; border-radius: 50%;
            border: 1.5px solid #E0E0E0;
            display: flex; align-items: center; justify-content: center;
            transition: border-color 0.15s, background 0.15s;
        }
        .login-quick-icon:hover { border-color: #C8102E; background: #FFF0F2; }
        .login-quick-icon svg {
            width: 24px; height: 24px; stroke: #555; fill: none;
            stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round;
        }
        .login-quick-label { font-size: 12px; color: #888; text-align: center; }
        [data-testid="stAlert"] { border-radius: 8px !important; font-size: 14px !important; }
    </style>

    <!-- Fixed left hero panel (pure HTML, z-index 10) -->
    <div class="login-hero-fixed">
        <div class="login-poly login-poly-1"></div>
        <div class="login-poly login-poly-2"></div>
        <div class="login-poly login-poly-3"></div>
        <div class="login-hero-logo">
            <div class="login-hero-logo-icon">
                <svg viewBox="0 0 24 24"><path d="M3 9l9-7 9 7v2H3V9zm2 3h14v8H5V12zm3 1v6h2v-6H8zm4 0v6h2v-6h-2zm4 0v6h2v-6h-2z"/></svg>
            </div>
            <span class="login-hero-logo-text"><span>Virtual</span>Bank</span>
        </div>
        <div class="login-hero-headline">VB<br>2026</div>
        <div class="login-hero-sub">Your Digital Bank</div>
        <div class="login-hero-tag"><b>Bank</b> for the <b>Future</b></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Form widgets (Streamlit stMain repositioned to right 42% via CSS) ──
    st.markdown('<div class="login-panel-title">Welcome to VirtualBank</div>', unsafe_allow_html=True)

    user_id  = st.text_input("User ID",  key="login_user_id",  placeholder="Enter your User ID")
    password = st.text_input("Password", key="login_password", placeholder="Enter your password", type="password")

    login_clicked = st.button("Log In", key="login_btn", type="primary", use_container_width=True)

    if login_clicked:
        if user_id == _DEMO_USER_ID and password == _DEMO_PASSWORD:
            st.session_state["is_logged_in"] = True
            st.rerun()
        else:
            st.error("Incorrect User ID or password. Please try again.")

    st.markdown("""
    <div class="login-demo-hint">
        <b>Demo credentials</b><br>
        User ID: <code>demo</code> &nbsp;&nbsp; Password: <code>VirtualBank2026</code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="login-quick">
        <div class="login-quick-item">
            <div class="login-quick-icon"><svg viewBox="0 0 24 24"><path d="M16 21v-2a4 4 0 00-4-4H6a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><line x1="19" y1="8" x2="19" y2="14"/><line x1="22" y1="11" x2="16" y2="11"/></svg></div>
            <span class="login-quick-label">Registration</span>
        </div>
        <div class="login-quick-item">
            <div class="login-quick-icon"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></div>
            <span class="login-quick-label">Forgot Password</span>
        </div>
        <div class="login-quick-item">
            <div class="login-quick-icon"><svg viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></div>
            <span class="login-quick-label">Report Fraud</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ── Login gate — show login page if not authenticated ─────────────────────
if not st.session_state.get("is_logged_in", False):
    _render_login_page()

# --- Query-param handlers (view routing + chat input + PTT) ---
# Legacy ?ql= quick-link now redirects to ?view= for workflow surfaces
_ql = st.query_params.get("ql")
if _ql and is_workflow(_ql):
    st.session_state["active_view"] = _ql
    st.query_params.clear()
    st.rerun()

# ?view= sets the active workflow surface
_view_param = st.query_params.get("view")
if _view_param and is_workflow(_view_param):
    st.session_state["active_view"] = _view_param
    st.query_params.clear()
    st.rerun()
elif _view_param == "dashboard":
    st.session_state["active_view"] = None
    # Clear all workflow prefill — user is returning to dashboard (fresh context)
    for _k in ("_sm_prefill", "_pc_prefill", "_pl_prefill", "_pb_prefill"):
        st.session_state.pop(_k, None)
    st.query_params.clear()
    st.rerun()

# ?modal= triggers an informational dialog
_modal_param = st.query_params.get("modal")
if _modal_param and is_modal(_modal_param):
    st.session_state["pending_modal"] = _modal_param
    st.query_params.clear()
    st.rerun()

_msg = st.query_params.get("msg")
if _msg:
    st.session_state["text_input"] = _msg
    st.session_state["audio_file"] = None
    st.query_params.clear()
    st.rerun()

_ptt = st.query_params.get("ptt")
if _ptt == "1":
    st.query_params.clear()
    _audio_file = record_audio()
    if _audio_file:
        st.session_state["audio_file"] = _audio_file
        st.session_state["text_input"] = None
    st.rerun()

# --- Session state initialisation ---
if "active_view" not in st.session_state:
    st.session_state["active_view"] = None
if "pending_modal" not in st.session_state:
    st.session_state["pending_modal"] = None

# Dispatch pending modals (must happen before layout renders)
_pending = st.session_state.get("pending_modal")
if _pending:
    st.session_state["pending_modal"] = None
    if _pending == "promotions":
        show_promotions(MOCK_ACCOUNT["promotions"])
    elif _pending == "mailbox":
        show_mailbox(MOCK_ACCOUNT["recent_transactions"])
    elif _pending == "account_summary":
        show_account_summary(MOCK_ACCOUNT)
    elif _pending == "maintenance":
        show_maintenance(MOCK_ACCOUNT["maintenance_notice"])

# --- Custom CSS Styling ---
st.markdown("""
<style>
    /* ── Base ───────────────────────────────────────────────── */
    .stApp { background-color: #F0F2F5 !important; font-family: 'Segoe UI', Arial, sans-serif !important; color: #1A1A2E !important; }

    /* ── Base ───────────────────────────────────────────────── */
    .stApp { background-color: #F0F2F5 !important; font-family: 'Segoe UI', Arial, sans-serif !important; color: #1A1A2E !important; }

    /* Hide Streamlit chrome */
    button[title="View fullscreen"] { display: none !important; }
    .stDeployButton { display: none !important; }
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    [data-testid="stHeader"] { display: none !important; height: 0 !important; min-height: 0 !important; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    /* Full-page lock — no page-level scroll */
    html, body { height: 100vh !important; overflow: hidden !important; }
    [data-testid="stApp"],
    [data-testid="stAppViewContainer"] { height: 100vh !important; overflow: hidden !important; }
    section[data-testid="stMain"] {
        height: 100vh !important; overflow: hidden !important;
        padding: 0 !important;
    }
    /* Block container: 100vh tall, 64px top + 68px left pad for fixed topbar+sidebar */
    [data-testid="stMainBlockContainer"] {
        height: 100vh !important; overflow: hidden !important;
        padding: 64px 0 0 68px !important; max-width: 100% !important;
        box-sizing: border-box !important;
        display: flex !important; flex-direction: column !important;
    }
    /* Main stVerticalBlock: flex column — stLayoutWrapper fills remaining height */
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
        flex: 1 !important; min-height: 0 !important; overflow: hidden !important;
        display: flex !important; flex-direction: column !important;
        gap: 0 !important; padding: 0 !important;
    }
    /* stLayoutWrapper wraps the two-column row */
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] {
        flex: 1 !important; min-height: 0 !important; overflow: hidden !important;
    }
    /* Two-column row: explicit height so columns can rely on 100% */
    [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {
        height: calc(100vh - 64px) !important; overflow: hidden !important;
        display: flex !important; align-items: stretch !important;
        margin: 0 !important; gap: 0 !important; flex-wrap: nowrap !important;
    }
    /* Dashboard column: scrollable — anchored to stLayoutWrapper to avoid matching nested columns */
    [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
        flex: 1 1 0 !important; min-width: 0 !important; height: 100% !important;
        overflow-y: auto !important; overflow-x: hidden !important;
        padding: 8px 16px 2rem !important;
    }
    /* Inner stVerticalBlock in dashboard column: grows to content height (enables scroll) */
    [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child > [data-testid="stVerticalBlock"] {
        min-height: max-content !important;
    }

    /* ── Fixed left sidebar ─────────────────────────────────── */
    .bank-sidebar {
        position: fixed; top: 0; left: 0; bottom: 0; width: 68px; z-index: 999;
        background: white; border-right: 1px solid #EBEBEB;
        display: flex; flex-direction: column; align-items: center;
        padding: 18px 0; gap: 8px; overflow: hidden;
    }
    .sb-logo {
        width: 46px; height: 46px; background: #C8102E; border-radius: 10px;
        display: flex; align-items: center; justify-content: center; margin-bottom: 18px;
    }
    .sb-logo svg { fill: white; width: 24px; height: 24px; }
    .sb-item {
        width: 46px; height: 46px; border-radius: 10px; cursor: pointer;
        display: flex; align-items: center; justify-content: center;
        color: #AAAAAA; transition: background 0.15s, color 0.15s;
    }
    .sb-item:hover { background: #FFF0F2; color: #C8102E; }
    .sb-item.active { background: #C8102E; color: white; }
    .sb-item svg { width: 22px; height: 22px; }
    .sb-spacer { flex: 1; }

    /* ── Custom topbar — FIXED at top ────────────────────────── */
    .bank-topbar {
        position: fixed; top: 0; left: 68px; right: 0; z-index: 998;
        background: white; border-bottom: 1px solid #EBEBEB;
        padding: 0 30px; height: 76px; display: flex; align-items: center; gap: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,.06);
    }
    .bank-brand { font-size: 24px; font-weight: 800; color: #1A1A1A; letter-spacing: -0.3px; }
    .bank-brand span { color: #C8102E; }
    .bank-topbar-meta { margin-left: auto; font-size: 16px; color: #999; }
    .bank-topbar-meta strong { color: #555; }
    .bank-topbar-avatar {
        width: 40px; height: 40px; border-radius: 50%;
        background: #C8102E; color: white;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 16px; font-weight: 700; margin-left: 16px; flex-shrink: 0;
    }

    /* ── Welcome hero ───────────────────────────────────────── */
    .bank-welcome {
        background: linear-gradient(130deg, #C8102E 0%, #8B0A20 100%);
        border-radius: 14px; padding: 20px 28px;
        color: white; display: flex; align-items: center; gap: 12px;
        margin-bottom: 26px; box-shadow: 0 4px 16px rgba(0,0,0,.10);
    }
    .bank-welcome-avatar {
        width: 62px; height: 62px; border-radius: 50%;
        background: rgba(255,255,255,0.18); border: 2px solid rgba(255,255,255,0.3);
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }
    .bank-welcome-avatar svg { width: 32px; height: 32px; fill: white; }
    .bank-welcome h2 { font-size: 25px; font-weight: 700; margin: 0 0 -10px 0; }
    .bank-welcome p  { font-size: 15px; opacity: .75; margin: 0; }

    /* ── Account summary card ───────────────────────────────── */
    .bank-summary {
        background: #FFFFFF; border-radius: 14px;
        border: 1px solid #F0F0F0; padding: 22px 24px; margin-bottom: 26px;
        box-shadow: 0 1px 4px rgba(0,0,0,.07);
        display: flex; align-items: center; justify-content: space-between;
    }
    .bank-summary-amount { font-size: 36px; font-weight: 800; letter-spacing: -0.5px; color: #1A1A1A; }
    .bank-summary-label  { font-size: 15px; color: #BBBBBB; margin-bottom: 7px; }
    .bank-summary-sub    { font-size: 15px; color: #AAAAAA; margin-top: 5px; }
    .bank-summary-link {
        width: 42px; height: 42px; border: 1.5px solid #E0E0E0; border-radius: 50%;
        display: flex; align-items: center; justify-content: center; cursor: pointer; flex-shrink: 0;
    }
    .bank-summary-link svg { width: 18px; stroke: #C8102E; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }

    /* ── Section title ──────────────────────────────────────── */
    .bank-section-title {
        font-size: 14px; font-weight: 700; color: #999;
        text-transform: uppercase; letter-spacing: 1.4px; margin-bottom: 13px;
    }

    /* ── Quick-action cards ─────────────────────────────────── */
    .bank-actions { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 26px; }
    .bank-action-card {
        background: #FFFFFF; border-radius: 14px; border: 1px solid #F0F0F0;
        padding: 20px 10px; text-align: center;
        cursor: pointer; box-shadow: 0 1px 4px rgba(0,0,0,.07);
        transition: box-shadow .15s, border-color .15s;
    }
    .bank-action-card:hover { box-shadow: 0 4px 14px rgba(200,16,46,0.10); border-color: #FFBBC4; }
    .bank-action-icon {
        width: 54px; height: 54px; border-radius: 12px;
        background: #FFF0F2; margin: 0 auto 12px;
        display: flex; align-items: center; justify-content: center;
    }
    .bank-action-label { font-size: 15px; font-weight: 600; color: #333; line-height: 1.35; }

    /* ── Topbar logout pill ─────────────────────────────────── */
    .bank-topbar-logout {
        background: transparent; border: 1.5px solid rgba(200,16,46,0.40);
        color: #C8102E; font-size: 13px; font-weight: 600;
        padding: 5px 16px; border-radius: 20px;
        cursor: pointer; white-space: nowrap;
        margin-left: 16px; flex-shrink: 0;
        transition: background 0.15s, border-color 0.15s; user-select: none;
    }
    .bank-topbar-logout:hover { background: #FFF0F2; border-color: #C8102E; }

    /* ── Quick-link column cards (st.button approach) ────────── */
    /* Card wrapper: the stColumn itself becomes the card.
       Selector: column whose DIRECT stVerticalBlock > DIRECT stElementContainer contains .ql-icon
       (the final descendant gap covers stMarkdown > stMarkdownContainer wrapping).
       The dashboard outer column is excluded because its direct stElementContainers don't hold .ql-icon. */
    [data-testid="stColumn"]:has(> [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] .ql-icon) {
        background: white !important; border-radius: 14px !important;
        border: 1px solid #F0F0F0 !important; box-shadow: 0 1px 4px rgba(0,0,0,.07) !important;
        overflow: hidden !important; transition: box-shadow .15s, border-color .15s !important;
        padding: 0 !important;
    }
    [data-testid="stColumn"]:has(> [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] .ql-icon):hover {
        box-shadow: 0 4px 14px rgba(200,16,46,0.10) !important; border-color: #FFBBC4 !important;
    }
    /* Kill ALL default stElementContainer chrome inside ql cards */
    [data-testid="stColumn"]:has(> [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] .ql-icon) [data-testid="stElementContainer"] {
        background: transparent !important; border: none !important;
        padding: 0 !important; margin: 0 !important; box-shadow: none !important;
    }
    /* Zero gap between icon and label */
    [data-testid="stColumn"]:has(> [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] .ql-icon) > [data-testid="stVerticalBlock"] {
        gap: 0 !important; padding: 0 !important;
    }
    /* Icon area */
    .ql-icon { text-align: center; padding: 20px 10px 8px; background: transparent; }
    .ql-icon .bank-action-icon { margin: 0 auto; }
    /* Button: invisible chrome, full-width, label text only */
    [data-testid="stColumn"]:has(> [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] .ql-icon) button[data-testid="stBaseButton-secondary"] {
        background: transparent !important; border: none !important; box-shadow: none !important;
        color: #333 !important; font-size: 15px !important; font-weight: 600 !important;
        padding: 4px 8px 20px !important; min-height: unset !important; line-height: 1.35 !important;
        width: 100% !important;
    }
    [data-testid="stColumn"]:has(> [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] .ql-icon) button[data-testid="stBaseButton-secondary"]:hover {
        background: transparent !important; color: #C8102E !important;
    }
    /* Reset nested-column row height (overrides the outer calc(100vh-64px) rule) */
    [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {
        height: auto !important; min-height: 0 !important;
        overflow: visible !important; gap: 12px !important;
    }
    [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 0 !important; min-width: 0 !important;
        height: auto !important; overflow: visible !important; padding: 0 !important;
    }

    /* ── System notice ──────────────────────────────────────── */
    .bank-notice {
        background: #FFFCF0; border: 1px solid #F5D86E;
        border-radius: 12px; padding: 16px 18px; margin-bottom: 26px;
        display: flex; align-items: flex-start; gap: 16px;
    }
    .bank-notice-icon {
        width: 64px; height: 64px; background: #F5A623; border-radius: 8px;
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
        margin-top: 4px;
    }
    .bank-notice-icon svg { fill: white; width: 48px; }
    .bank-notice h4 { font-size: 17px; font-weight: 700; color: #C8102E; margin: 0 0 -10px 0; }
    .bank-notice p  { font-size: 15px; color: #666; margin: 0; line-height: 1.6; }

    /* ── Mailbox / Promotions cards ─────────────────────────── */
    .info-card { background: white; border-radius: 14px; border: 1px solid #F0F0F0; padding: 18px 20px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
    .info-card-hdr { display: flex; align-items: center; margin-bottom: 14px; }
    .info-card-title { font-size: 17px; font-weight: 700; color: #1A1A1A; }
    .info-card-action { margin-left: auto; font-size: 14px; color: #C8102E; border: 1px solid #C8102E; border-radius: 20px; padding: 4px 14px; cursor: pointer; }
    .mailbox-tbl { width: 100%; border-collapse: collapse; }
    .mailbox-tbl th { font-size: 13px; font-weight: 700; color: #AAAAAA; text-transform: uppercase; letter-spacing: .8px; padding: 6px 4px; text-align: left; border-bottom: 1px solid #F0F0F0; }
    .mailbox-tbl td { font-size: 15px; color: #666; padding: 9px 4px; border-bottom: 1px solid #F8F8F8; vertical-align: middle; }
    .mailbox-tbl td:last-child { font-weight: 500; color: #333; }
    .red-dot { display: inline-block; width: 8px; height: 8px; background: #C8102E; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
    .promo-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 4px; }
    .promo-red  { height: 86px; background: linear-gradient(135deg, #C8102E, #E84C6A); border-radius: 10px 10px 0 0; display: flex; align-items: center; justify-content: center; color: white; font-size: 14px; font-weight: 600; text-align: center; padding: 10px; line-height: 1.4; }
    .promo-blue { height: 86px; background: linear-gradient(135deg, #1A56DB, #3B82F6); border-radius: 10px 10px 0 0; display: flex; align-items: center; justify-content: center; color: white; font-size: 14px; font-weight: 600; text-align: center; padding: 10px; line-height: 1.4; }
    .promo-label { font-size: 14px; color: #666; padding: 6px 7px 5px; background: #F8F8F8; border-radius: 0 0 10px 10px; }
    /* ── Mailbox + Promotions equal-height row ──────────────── */
    .info-cards-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .info-cards-row .info-card { height: 100%; box-sizing: border-box; }

    /* ── Chat bubbles ────────────────────────────────────────── */
    .chat-bubble-user {
        background: #C8102E; color: white;
        padding: 12px 16px; border-radius: 18px 18px 4px 18px;
        margin: 8px 0; margin-left: auto; max-width: 80%;
        font-size: 16px; line-height: 1.55;
    }
    .chat-bubble-teller {
        background: #FFFFFF; color: #333;
        padding: 12px 16px; border-radius: 4px 18px 18px 18px;
        margin: 8px 0; border: 1px solid #EBEBEB;
        max-width: 80%; font-size: 16px; line-height: 1.55;
        box-shadow: 0 1px 4px rgba(0,0,0,.05);
    }
    .chat-action-link {
        display: inline-block; margin-top: 6px;
        background: #C8102E; color: white !important;
        font-size: 13px; font-weight: 700;
        padding: 5px 14px; border-radius: 20px;
        cursor: pointer; text-decoration: none;
        box-shadow: 0 2px 6px rgba(200,16,46,0.30);
        transition: background 0.15s;
    }
    .chat-action-link:hover { background: #a00c25; }
    .role-label { font-size: 13px; font-weight: 700; color: #CCCCCC; margin-bottom: 5px; text-transform: uppercase; letter-spacing: .8px; }

    [data-testid="stForm"] { display: none !important; }
    [data-testid="stFormSubmitButton"] { display: none !important; }

    /* ── Teller panel header ─────────────────────────────────── */
    .teller-panel-header {
        background: linear-gradient(130deg, #C8102E 0%, #8B0A20 100%); color: white;
        padding: 35px 20px;
        display: flex; align-items: center; gap: 15px; flex-shrink: 0;
    }
    .teller-panel-header h3 { font-size: 18px; font-weight: 700; margin: 0 !important; padding: 0 !important; line-height: 1.2; }
    .teller-panel-header small { font-size: 14px; opacity: .72; display: block; margin: 0 !important; padding: 0 !important; line-height: 1.3; }
    .teller-panel-header > div { display: flex; flex-direction: column; gap: 0 !important; }
    .teller-online-dot { width: 10px; height: 10px; border-radius: 50%; background: #4ADE80; flex-shrink: 0; animation: blink 2s infinite; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.4} }
    .teller-online-badge { margin-left: auto; font-size: 14px; background: rgba(255,255,255,.18); color: white; padding: 4px 13px; border-radius: 20px; font-weight: 600; }

    /* ── Teller column — anchored to stLayoutWrapper to avoid nested column match ── */
    [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
        flex: 0 0 460px !important; width: 460px !important; max-width: 460px !important;
        height: 100% !important; overflow: hidden !important;
        display: flex !important; flex-direction: column !important;
        background: white !important; border-left: 1px solid #EBEBEB !important;
        padding: 0 !important;
    }
    /* stVerticalBlock inside teller column: flex column */
    [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] {
        flex: 1 !important; min-height: 0 !important;
        display: flex !important; flex-direction: column !important;
        overflow: hidden !important; padding: 0 !important; gap: 0 !important;
    }
    /* All direct children of teller stVerticalBlock: shrink-free (header, nameplate, Live2D, input bar) */
    [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"],
    [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] > [data-testid="stCustomComponentV1"] {
        flex-shrink: 0 !important;
    }
    /* Chat container stLayoutWrapper: fills remaining space and scrolls */
    [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] {
        flex: 1 !important; min-height: 0 !important;
        overflow-y: auto !important; overflow-x: hidden !important;
        background: #FAFAFA !important; padding: 16px 18px !important;
        border-top: 1px solid #E2E5EF !important;
    }

    /* ── Spinner ───────────────────────────────────────────── */
    [data-testid="stSpinner"] { color: #C8102E !important; }

    /* ── Teller stage & nameplate ─────────────────────────────── */
    [data-testid="stCustomComponentV1"] iframe { background: transparent !important; }
    .teller-nameplate { text-align: center; padding: 0 0 10px; position: relative; z-index: 5; }
    .teller-nameplate-tag {
        background: #C8102E; color: white; font-size: 14px; font-weight: 800;
        padding: 5px 28px; border-radius: 20px; letter-spacing: 2.5px; display: inline-block;
        box-shadow: 0 2px 8px rgba(200,16,46,0.35);
    }

    /* ── Quick-link SVG icons ───────────────────────────────── */
    .bank-action-icon svg { width: 26px; height: 26px; stroke: #C8102E; fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }

    /* ── Teller panel: stage gradient background for Live2D ────── */
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(2) {
        background: linear-gradient(to bottom, #fff5f6 0%, #fce4e8 100%) !important;
    }
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(2) iframe {
        background: transparent !important;
    }
    /* Teller stVerticalBlock: position relative so PTT button can use absolute positioning */
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] {
        position: relative !important;
    }

    /* ── Input bar always at bottom of teller panel ──────────── */
    /* Chat area stVerticalBlock (chat_container): fills space, scrollable */
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] {
        background: #FAFAFA !important;
    }
    /* Nameplate and header containers: white background */
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(1),
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(3) {
        background: white !important;
    }
    /* ── Style st.chat_input() ─────────────────────────────── */
    [data-testid="stChatInput"] {
        padding: 10px 14px 10px 14px !important; background: white !important;
        border-top: 1px solid #EBEBEB !important;
        box-sizing: border-box !important;
        min-height: 72px !important; height: 72px !important;
    }
    /* Override Streamlit's dark inner container — all layers white */
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] > div > div > div {
        background: white !important;
        border: none !important; box-shadow: none !important;
    }
    /* Inner wrapper: add right padding so text stays left of PTT button */
    [data-testid="stChatInput"] > div {
        padding-right: 58px !important;
    }
    /* Textarea itself */
    [data-testid="stChatInputTextArea"] {
        background: #F5F6FA !important; color: #333 !important;
        border: 1.5px solid #E0E0E0 !important; border-radius: 22px !important;
        font-size: 14px !important; padding: 10px 16px !important;
        resize: none !important; line-height: 1.4 !important;
        min-height: 38px !important; height: 38px !important; max-height: 38px !important;
        box-shadow: none !important; overflow: hidden !important;
    }
    [data-testid="stChatInputTextArea"]:focus {
        border-color: #C8102E !important; background: white !important;
        outline: none !important; box-shadow: none !important;
    }
    [data-testid="stChatInputTextArea"]::placeholder { color: #AAAAAA !important; }
    /* Hide the native send arrow */
    [data-testid="stChatInputSubmitButton"] { display: none !important; }

    /* ── PTT mic button — positioned at bottom-right of chat input ── */
    /* The stElementContainer wrapping the PTT st.button */
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has([data-testid="stBaseButton-secondary"]) {
        position: absolute !important;
        bottom: 8px !important; right: 10px !important;
        z-index: 200 !important;
        width: 52px !important; height: 52px !important;
        padding: 0 !important; margin: 0 !important;
        background: transparent !important;
    }
    /* PTT button itself: red circle with SVG mic icon */
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] button[data-testid="stBaseButton-secondary"] {
        width: 48px !important; height: 48px !important;
        border-radius: 50% !important; background: #C8102E !important;
        color: transparent !important; border: none !important;
        font-size: 0 !important; line-height: 1 !important;
        padding: 0 !important; min-height: unset !important;
        box-shadow: 0 2px 10px rgba(200,16,46,0.40) !important;
        display: flex !important; align-items: center !important; justify-content: center !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M12 15c1.66 0 3-1.34 3-3V6c0-1.66-1.34-3-3-3S9 4.34 9 6v6c0 1.66 1.34 3 3 3zm-1-9c0-.55.45-1 1-1s1 .45 1 1v6c0 .55-.45 1-1 1s-1-.45-1-1V6zm6 6c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z'/%3E%3C/svg%3E") !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
        background-size: 24px 24px !important;
    }
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] button[data-testid="stBaseButton-secondary"]:hover {
        background: #a00c25 !important;
    }
    /* Nameplate and header elements: white background; Live2D iframe element: transparent */
    [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(1),
    [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(3) {
        background: white !important;
    }
    [data-testid="stCustomComponentV1"] { margin: 0 !important; }
    /* Back-to-dashboard breadcrumb button: link style */
    .st-key-back_to_dashboard button {
        background: transparent !important; border: none !important; box-shadow: none !important;
        color: #C8102E !important; font-size: 14px !important; font-weight: 600 !important;
        padding: 0 !important; min-height: unset !important; height: auto !important;
    }
    .st-key-back_to_dashboard button:hover { text-decoration: underline !important; color: #a00c25 !important; }
    .st-key-back_to_dashboard [data-testid="stElementContainer"] { padding: 0 !important; margin: 0 !important; background: transparent !important; }
    /* Hidden sidebar nav trigger buttons — always present but invisible */
    .st-key-_sb_nav_send_money, .st-key-_sb_nav_pay_card,
    .st-key-_sb_nav_pay_loan,  .st-key-_sb_nav_pay_bill  { display: none !important; }
</style>
""", unsafe_allow_html=True)

# --- Streamlit UI ---
from datetime import datetime
_now = datetime.now()
_day_str = _now.strftime("%A, %d %B %Y")

# Sidebar (fixed position, pure HTML) — items mirror the 4 quick links
_sb_view = st.session_state.get("active_view")
_sb_home_cls  = "sb-item" + ("" if _sb_view else " active")
_sb_sm_cls    = "sb-item" + (" active" if _sb_view == "send_money" else "")
_sb_pc_cls    = "sb-item" + (" active" if _sb_view == "pay_card"   else "")
_sb_pl_cls    = "sb-item" + (" active" if _sb_view == "pay_loan"   else "")
_sb_pb_cls    = "sb-item" + (" active" if _sb_view == "pay_bill"   else "")
st.markdown(f"""
<nav class="bank-sidebar">
  <div class="sb-logo">
    <svg viewBox="0 0 24 24"><path d="M3 9l9-7 9 7v2H3V9zm2 3h14v8H5V12zm3 1v6h2v-6H8zm4 0v6h2v-6h-2zm4 0v6h2v-6h-2z"/></svg>
  </div>
  <div class="{_sb_home_cls}" data-nav="home" title="Dashboard">
    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/></svg>
  </div>
  <div class="{_sb_sm_cls}" data-nav="send_money" title="Send Money">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
  </div>
  <div class="{_sb_pc_cls}" data-nav="pay_card" title="Pay Card">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="5" width="20" height="14" rx="2"/><line x1="2" y1="10" x2="22" y2="10"/></svg>
  </div>
  <div class="{_sb_pl_cls}" data-nav="pay_loan" title="Pay Loan / Financing">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>
  </div>
  <div class="{_sb_pb_cls}" data-nav="pay_bill" title="Pay Bill">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="2" width="14" height="20" rx="2"/><line x1="9" y1="7" x2="15" y2="7"/><line x1="9" y1="11" x2="15" y2="11"/><line x1="9" y1="15" x2="13" y2="15"/></svg>
  </div>
  <div class="sb-spacer"></div>
</nav>
""", unsafe_allow_html=True)
# Inject sidebar click handlers via components.html (has allow-same-origin → can reach parent DOM)
# No inline onclick attributes so React #231 is avoided.
components.html("""
<style>html,body{margin:0;padding:0;overflow:hidden;}</style>
<script>
(function() {
  var p = window.parent.document;
  function _sbNav(key) {
    if (key === 'home') {
      var bk = p.querySelector('.st-key-back_to_dashboard button');
      if (bk) { bk.click(); return; }
      return; // already on dashboard
    }
    // Use always-present hidden nav trigger (works from any view)
    var b = p.querySelector('.st-key-_sb_nav_' + key + ' button');
    if (b) { b.click(); return; }
  }
  function _attachSbListeners() {
    p.querySelectorAll('.sb-item[data-nav]').forEach(function(el) {
      if (el._sbAttached) return;
      el._sbAttached = true;
      el.addEventListener('click', function() { _sbNav(el.getAttribute('data-nav')); });
    });
    // Chat action links (rendered in chat bubbles by render_action_links)
    p.querySelectorAll('.chat-action-link[data-nav]').forEach(function(el) {
      if (el._sbAttached) return;
      el._sbAttached = true;
      el.addEventListener('click', function(e) {
        e.preventDefault();
        _sbNav(el.getAttribute('data-nav'));
      });
    });
  }
  _attachSbListeners();
  new MutationObserver(_attachSbListeners).observe(p.body, { childList: true, subtree: true });
})();
</script>
""", height=0, scrolling=False)

# Topbar
st.markdown(f"""
<div class="bank-topbar">
    <span class="bank-brand"><span>Virtual</span>Bank</span>
    <span class="bank-topbar-meta">{_day_str} &nbsp;|&nbsp; Last login: <strong>Today 09:14 AM (GMT+8)</strong></span>
    <span class="bank-topbar-logout" onclick="window.location.href='/'">&#x238B;&nbsp; Log out</span>
    <span class="bank-topbar-avatar">JD</span>
</div>
""", unsafe_allow_html=True)

# Hidden nav trigger buttons — always rendered so sidebar JS can click them from any view
# Styled display:none via .st-key-_sb_nav_* CSS
_sb_nav_targets = ["send_money", "pay_card", "pay_loan", "pay_bill"]
for _snk in _sb_nav_targets:
    if st.button("_sb_nav", key=f"_sb_nav_{_snk}"):
        st.session_state["active_view"] = _snk
        st.rerun()

# ── Two-column layout: dashboard (left) + teller (right) ─────────
col_dashboard, col_teller = st.columns([1.5, 1], gap="small")
# ── LEFT: Banking dashboard / workflow surface ─────────────────────
with col_dashboard:
    # Spacer below fixed topbar
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    _active_view = st.session_state.get("active_view")

    if _active_view:
        # ── Workflow surface ─────────────────────────────────────────
        # Back navigation — simple button, no breadcrumb text (heading already says the page)
        if st.button("← Dashboard", key="back_to_dashboard"):
            st.session_state["active_view"] = None
            st.rerun()
        st.markdown('<div style="margin-bottom:4px"></div>', unsafe_allow_html=True)
        # Dispatch to the correct workflow renderer
        if _active_view == "send_money":
            render_send_money(MOCK_ACCOUNT)
        elif _active_view == "pay_card":
            render_pay_card(MOCK_ACCOUNT)
        elif _active_view == "pay_loan":
            render_pay_loan(MOCK_ACCOUNT)
        elif _active_view == "pay_bill":
            render_pay_bill(MOCK_ACCOUNT)
    else:
        # ── Dashboard (default view) ─────────────────────────────────
        # Welcome hero
        st.markdown("""
        <div class="bank-welcome">
            <div class="bank-welcome-avatar">
                <svg viewBox="0 0 24 24"><path d="M12 12c2.7 0 4.8-2.1 4.8-4.8S14.7 2.4 12 2.4 7.2 4.5 7.2 7.2 9.3 12 12 12zm0 2.4c-3.2 0-9.6 1.6-9.6 4.8v2.4h19.2v-2.4c0-3.2-6.4-4.8-9.6-4.8z"/></svg>
            </div>
            <div>
                <h2>Welcome back, John Doe</h2>
                <p>Your last login was on 17 Apr 2026, 09:14:03 AM (GMT+8)</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Account summary — clicking the arrow opens the account summary modal
        st.markdown('<div class="bank-section-title">Account Summary</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="bank-summary">
            <div>
                <div class="bank-summary-label">Current Account &nbsp;&bull;&nbsp; 1234-5678-9012</div>
                <div class="bank-summary-amount">RM 12,480.50</div>
                <div class="bank-summary-sub">Keep track of the money flowing in and out of your account.</div>
            </div>
            <div class="bank-summary-link" onclick="window.location.search='?modal=account_summary'" title="View details">
                <svg viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"/></svg>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick-links — st.button triggers in-session rerun (no page-reload / no session reset)
        st.markdown('<div class="bank-section-title">Quick Links</div>', unsafe_allow_html=True)
        _ql_cols = st.columns(4, gap="medium")
        _ql_items = [
            ("send_money", '<svg viewBox="0 0 24 24" fill="none" stroke="#C8102E" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>', "Send Money"),
            ("pay_card",   '<svg viewBox="0 0 24 24" fill="none" stroke="#C8102E" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="5" width="20" height="14" rx="2"/><line x1="2" y1="10" x2="22" y2="10"/></svg>', "Pay Card"),
            ("pay_loan",   '<svg viewBox="0 0 24 24" fill="none" stroke="#C8102E" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>', "Pay Loan /\nFinancing"),
            ("pay_bill",   '<svg viewBox="0 0 24 24" fill="none" stroke="#C8102E" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="2" width="14" height="20" rx="2"/><line x1="9" y1="7" x2="15" y2="7"/><line x1="9" y1="11" x2="15" y2="11"/><line x1="9" y1="15" x2="13" y2="15"/></svg>', "Pay Bill"),
        ]
        for _qi, (_vk, _svg, _lbl) in enumerate(_ql_items):
            with _ql_cols[_qi]:
                st.markdown(f'<div class="ql-icon"><div class="bank-action-icon">{_svg}</div></div>', unsafe_allow_html=True)
                if st.button(_lbl, key=f"nav_{_vk}", use_container_width=True):
                    st.session_state["active_view"] = _vk
                    st.rerun()
        st.markdown('<div style="margin-bottom:14px"></div>', unsafe_allow_html=True)

        # System notice — clicking opens maintenance modal
        st.markdown("""
        <div class="bank-notice" onclick="window.location.search='?modal=maintenance'" style="cursor:pointer;" title="View details">
            <div class="bank-notice-icon">
                <svg viewBox="0 0 24 24"><path d="M12 2L1 21h22L12 2zm0 4l7.5 13h-15L12 6zm-1 5v4h2v-4h-2zm0 6v2h2v-2h-2z"/></svg>
            </div>
            <div>
                <h4>System Maintenance Notice</h4>
                <p>VirtualBank will be temporarily unavailable on: 5, 6, 7, 11, 12 May 2026,
                   from 12:00 am – 6:00 am. Click to view details.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Mailbox + Promotions — clicking cards opens modals
        st.markdown("""
        <div class="info-cards-row">
            <div class="info-card">
                <div class="info-card-hdr">
                    <span class="info-card-title">Mailbox</span>
                    <span class="info-card-action" onclick="window.location.search='?modal=mailbox'" style="cursor:pointer;">View More</span>
                </div>
                <table class="mailbox-tbl">
                    <thead><tr><th>Date / Time</th><th>Subject</th></tr></thead>
                    <tbody>
                        <tr><td>13 Apr 2026<br>07:55 PM</td><td><span class="red-dot"></span>Card Alert</td></tr>
                        <tr><td>10 Apr 2026<br>09:08 PM</td><td>Card Transaction Alert</td></tr>
                        <tr><td>09 Apr 2026<br>02:26 PM</td><td>Card Paid</td></tr>
                        <tr><td>02 Apr 2026<br>01:53 AM</td><td>Card Transaction Alert</td></tr>
                        <tr><td>01 Apr 2026<br>07:44 PM</td><td><span class="red-dot"></span>Card Alert</td></tr>
                    </tbody>
                </table>
            </div>
            <div class="info-card">
                <div class="info-card-title" style="margin-bottom:12px">Promotions</div>
                <div class="promo-grid">
                    <div onclick="window.location.search='?modal=promotions'" style="cursor:pointer;">
                        <div class="promo-red">Get Your<br>Grand Rewards →</div>
                        <div class="promo-label">Get Your Grand Rewards...</div>
                    </div>
                    <div onclick="window.location.search='?modal=promotions'" style="cursor:pointer;">
                        <div class="promo-blue">Cashback &amp;<br>Petrol Deals →</div>
                        <div class="promo-label">Cashback &amp; Petrol Deals...</div>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ── RIGHT: Teller panel ───────────────────────────────────────────
with col_teller:
    # Teller panel header (no rounded corners — flush panel like mockup)
    st.markdown("""
    <div class="teller-panel-header">
        <div class="teller-online-dot"></div>
        <div>
            <h3>Haru — Virtual Teller</h3>
            <small>Ask me anything about your account</small>
        </div>
        <span class="teller-online-badge">● Online</span>
    </div>
    """, unsafe_allow_html=True)

    # Live2D avatar — rendered once; Streamlit won't remount if HTML is stable
    html_to_render = LIVE2D_HTML.replace("<!-- LIVE2D_DISPLAY_SCRIPT_MOUNT -->", f"<script>{live2d_display_script}</script>")
    components.html(html_to_render, height=460, width=460)
    # HARU nameplate below avatar (matches mockup teller-nameplate)
    st.markdown('<div class="teller-nameplate"><span class="teller-nameplate-tag">HARU</span></div>', unsafe_allow_html=True)

    # Persistent chat history — seed with Haru's greeting so the chat area always renders
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "teller", "text": "Hi! I'm Haru, your virtual banking assistant. How can I help you today? 😊"}
        ]

    # Chat container — this stVerticalBlock is styled as the scrollable chat area via CSS
    chat_container = st.container()

    # Always replay full history first
    with chat_container:
        for _m in st.session_state["messages"]:
            if _m["role"] == "user":
                st.markdown(
                    f'<div style="text-align:right;"><div class="role-label">You</div>'
                    f'<div class="chat-bubble-user">{_m["text"]}</div></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="role-label" style="margin-top:12px;">Haru</div>'
                    f'<div class="chat-bubble-teller">{render_action_links(_m["text"])}</div>',
                    unsafe_allow_html=True
                )

    # Auto-scroll the chat area to the bottom after every render and during streaming.
    # The interval is created on window.parent so it survives iframe replacement on rerenders.
    components.html("""
<style>html,body{margin:0;padding:0;overflow:hidden;}</style>
<script>
(function() {
  var p = window.parent;
  var pd = p.document;
  var SEL =
    '[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > ' +
    '[data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > ' +
    '[data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] > ' +
    '[data-testid="stLayoutWrapper"]';

  // Clear any stale interval from a previous render (interval ID on parent survives iframe swap)
  if (p._chatScrollInterval) {
    p.clearInterval(p._chatScrollInterval);
    p._chatScrollInterval = null;
  }

  // Tick runs in parent window timer queue — survives this iframe being destroyed
  p._chatScrollTick = function() {
    var el = pd.querySelector(SEL);
    if (!el) return;
    // Attach scroll listener once to respect manual scrolling up
    if (!el._chatScrollInit) {
      el._chatScrollInit = true;
      el._userScrolledUp = false;
      el.addEventListener('scroll', function() {
        el._userScrolledUp = (el.scrollHeight - el.scrollTop - el.clientHeight) > 60;
      });
    }
    if (!el._userScrolledUp) {
      el.scrollTop = el.scrollHeight;
    }
  };

  p._chatScrollTick();
  p._chatScrollInterval = p.setInterval(p._chatScrollTick, 200);
})();
</script>
""", height=0, scrolling=False)

    # Native chat input — communicates via Streamlit websocket, no sandbox issues
    val = st.chat_input("Type a message to Haru…", key="teller_chat_input")

    # PTT mic button — native Streamlit button, no sandbox issues
    # CSS positions it absolutely at bottom-right of the chat input area
    if st.button("🎤", key="ptt_btn", help="Push to Talk"):
        _audio_file = record_audio()
        if _audio_file:
            st.session_state['audio_file'] = _audio_file
            st.session_state['text_input'] = None
        st.rerun()

    # Resolve active input: typed chat > quick-link > PTT audio
    user_text = None
    if val:
        user_text = val
    elif st.session_state.get('text_input'):
        user_text = st.session_state.get('text_input')
        st.session_state['text_input'] = None
    elif st.session_state.get('audio_file'):
        _t_stt = time.time()
        user_text = stt_model.transcribe(st.session_state['audio_file'], fp16=True)["text"]
        _perf("STT transcription", time.time() - _t_stt)
        try:
            os.remove(st.session_state['audio_file'])
        except OSError:
            pass
        st.session_state['audio_file'] = None

    if user_text:
        # Detect prefill params for every workflow.
        # Only overwrite a key when the new parse is non-None so that a follow-up
        # message that doesn't mention a workflow doesn't wipe the original prefill.
        # Keys are cleared on dashboard navigation (see ?view=dashboard handler).
        _new_sm = _parse_send_money_prefill(user_text)
        _new_pc = _parse_pay_card_prefill(user_text)
        _new_pl = _parse_pay_loan_prefill(user_text)
        _new_pb = _parse_pay_bill_prefill(user_text)
        if _new_sm is not None: st.session_state["_sm_prefill"] = _new_sm
        if _new_pc is not None: st.session_state["_pc_prefill"] = _new_pc
        if _new_pl is not None: st.session_state["_pl_prefill"] = _new_pl
        if _new_pb is not None: st.session_state["_pb_prefill"] = _new_pb
        st.session_state["messages"].append({"role": "user", "text": user_text})
        with chat_container:
            st.markdown(
                f'<div style="text-align:right;"><div class="role-label">You</div>'
                f'<div class="chat-bubble-user">{user_text}</div></div>',
                unsafe_allow_html=True
            )

        write_teller_state(STATE_FILE, "thinking", "")
        with chat_container:
            st.markdown('<div class="role-label" style="margin-top:12px;">Haru</div>', unsafe_allow_html=True)
            responseContainer = st.empty()

        full_response = ""
        current_sentence = ""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        _t_llm = time.time()
        _first_token = True
        _sentence_idx = 0

        for chunk in chain.stream({"input": user_text}):
            if _first_token:
                _perf("LLM first token", time.time() - _t_llm)
                _first_token = False
            full_response += chunk
            current_sentence += chunk
            responseContainer.markdown(
                f'<div class="chat-bubble-teller">{render_action_links(full_response)}</div>',
                unsafe_allow_html=True
            )
            if re.search(r'[.!?]\s', current_sentence) or current_sentence.endswith(('.', '!', '?')):
                cleaned_sentence = clean_text(current_sentence)
                if cleaned_sentence:
                    _sentence_idx += 1
                    _t_tts = time.time()
                    executor.submit(run_tts_sync, cleaned_sentence)
                    _perf(f"TTS sentence {_sentence_idx} dispatched", time.time() - _t_tts)
                current_sentence = ""

        if current_sentence.strip():
            cleaned_sentence = clean_text(current_sentence)
            if cleaned_sentence:
                _sentence_idx += 1
                executor.submit(run_tts_sync, cleaned_sentence)

        _perf("LLM total stream", time.time() - _t_llm)
        st.session_state["messages"].append({"role": "teller", "text": full_response})
        write_teller_state(STATE_FILE, "idle", "")

