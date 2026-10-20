from flask import Flask, request, jsonify
from groq import Groq
import os
import json
import re

app = Flask(__name__)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

PROMPT_TEMPLATE = """
You are a data extraction assistant. Extract structured data from an LRU removal email.

The input contains the subject line and body of an email. Extract the following fields and return ONLY a single valid JSON object with no explanation, no markdown, no code blocks, no array brackets.

Fields to extract:
- tail: Aircraft tail/registration number. Starts with a country/airline registration prefix (N, TC, TF, F, SE, LN, EI, XA, etc.) followed by numbers and/or letters. Remove any dashes or spaces (example: "N-8843S" -> "N8843S", "TC-JHZ" -> "TCJHZ").
- location: Airport code. Usually appears after "@" symbol (example: @DEN -> DEN). May also appear as a standalone 3-letter IATA airport code.
- LRU: The LRU type of the unit REMOVED. Must be ONE single name from the LRU TYPE RULES below. Never return two LRU types, never use "/", ",", "&", "and" or "->" between types. For an upgrade (e.g. 40W HPT to 50W HPT, SAA to GSAA) return only the type that came OFF.
- removal_reason: Why the unit was removed or failed. Includes operational failures AND upgrade descriptions (examples: "upgraded to ModManSX", "upgrade to 50W HPT").
- off_sn: Serial number of the unit removed. Usually after "OFF:", "SN OFF:", "Removed SN:", or "Prim S/N OFF:".
- on_sn: Serial number of the unit installed. Usually after "ON:", "SN ON:", "Installed SN:", or "Prim S/N ON:".
- status: Operational status after the swap ONLY (examples: Fully Operational, Intermittent, Removed Only, Pending Test). Never put upgrade descriptions here.
- notes: Any additional information not captured in the above fields. If there is nothing to add, return an empty string "" (never "UNKNOWN").

==================== LRU TYPE RULES ====================
Allowed LRU names (use exactly as written):
"25W HPT", "40W HPT", "50W HPT", "A200 CWLU", "ACU", "APSU", "C-KRFU", "CWLU", "CWLU II",
"LF ACU", "LF SAA", "M-KRFU", "Max ACU", "Max SAA", "MDU-D", "Modman", "ModMan-S",
"Q05 GSAA", "Q16 GSAA", "SAA", "SMU", "SMU II", "SMU III", "SMU Mod20", "T-KRFU",
"TT CWLU", "TT SMU"

Decide the LRU in this order: (1) the OFF serial number format, (2) the airline, (3) the words in the email.

STEP 1 - Identify the airline from the tail:
- N + digits (N8843S, N275) = Southwest
- TC = Turkish | F (FHEPI, FGKXT) = Air France | LN, SE, EI = Norwegian | TF = Icelandair | XA = Aeromexico

STEP 2 - Only these LRU types exist on each airline. Never pick a type that is not on the tail's airline:
- Southwest: 40W HPT, 50W HPT, A200 CWLU, ACU, APSU, MDU-D, Max ACU, Max SAA, Q05 GSAA, SAA, SMU Mod20 (LF SAA / LF ACU only if "LF" is written)
- Turkish: APSU, C-KRFU, M-KRFU, Modman, ModMan-S, Q16 GSAA, TT CWLU, TT SMU
- Air France: 50W HPT, APSU, CWLU II, Modman, Q16 GSAA, SMU III, T-KRFU
- Norwegian: 50W HPT, A200 CWLU, ACU, CWLU II, MDU-D, SAA, SMU, SMU II
- Icelandair: 25W HPT, 50W HPT, A200 CWLU, ACU, CWLU, MDU-D, SAA, SMU
- Aeromexico: 40W HPT, ACU, CWLU, MDU-D, SAA, SMU II

STEP 3 - Serial number formats of the OFF unit (these override words in the email):
- "WTA-000xxx" (WTA prefix) = 40W HPT. A WTA serial is NEVER a 50W HPT.
- "SNxxxx" / "SN1xxxx" or a bare 4-5 digit number from 6400 to 23000 on an HPT = 50W HPT (except Turkish, see below)
- "Wxxxx" = 40W HPT (Aeromexico) or 25W HPT (Icelandair). "Txxxx" on Aeromexico = 40W HPT.
- "Txxxx" on Air France (T0105-T0317) = T-KRFU
- "XAxxxxx" or "XLxxxxx" = MDU-D
- "GDxxxx-xxx" = APSU
- "YIxxxxx" = ModMan-S
- 7-digit "492xxxx" or 8-digit "492011xx" = Modman
- Letter-coded KRFU serials like "M340799A0004", "MC40799A0006", "L740799B0001", "N440799A0004" = C-KRFU
- Turkish "SN1xxxx" (SN10817-SN13803) = M-KRFU
- Turkish "SN000xx"/"SN00xxx" = TT CWLU or TT SMU -> decide by the words CWLU/WAP vs SMU/server. SN00111 and higher = TT CWLU.
- 9-digit "2008xxxxx" = SMU III (Air France has SMU III only, never SMU II)
- 9-digit "2005xxxxx" or "2006xxxxx" = SMU Mod20 on Southwest, SMU II on Norwegian/Aeromexico
- 9-digit "462xxxxxx"-"478xxxxxx" (or 10-digit "37xxxxxxxx") = A200 CWLU
- Short 1-4 digit numbers are used by SAA, Max SAA, Q05 GSAA, Q16 GSAA, ACU, Max ACU, SMU (Norwegian/Icelandair) and CWLU/CWLU II -> use STEP 4.

STEP 4 - Antennas, ACUs, SMUs and CWLUs with short serials:
- Southwest 737 MAX tails are N87xx, N88xx and N18xx (4 digits). On those tails the antenna is "Max SAA" and the ACU is "Max ACU". All other Southwest tails use SAA / Q05 GSAA / ACU.
- Southwest antenna that is not a MAX tail: "GSAA", "Q05", PA current, mute relay or homing switch = Q05 GSAA; otherwise SAA. A zero-padded 4-digit serial (e.g. "0132") = Q05 GSAA.
- If an SAA is being upgraded/replaced with a GSAA ("upgrade to GSAA", "Phase 3 mod", "swapped with GSAA"), the removed LRU is "SAA" and the upgrade goes in removal_reason.
- Any GSAA on Turkish or Air France = Q16 GSAA. Never Q05 GSAA.
- HPT on Norwegian, Air France or Turkish = 50W HPT.
- SMU: Southwest = SMU Mod20, Air France = SMU III, Turkish = TT SMU, Norwegian/Icelandair short serial = SMU.
- CWLU: Southwest = A200 CWLU, Turkish = TT CWLU, Air France = CWLU II, Norwegian short serial = CWLU II, Icelandair/Aeromexico short serial = CWLU.
- KRFU on Air France = T-KRFU.
- Abbreviations: "MM" = Modman, "MMS"/"MMSX"/"ModManSX" = ModMan-S, "MDU" = MDU-D, "WAP" = CWLU family.
- If none of the allowed names fit (e.g. Radome, FDCP), return the raw name as written in the email.
========================================================

Rules:
- Return ONLY a single valid JSON object. No array, no wrapper, no explanation, no markdown, no code blocks.
- If a field other than notes cannot be found, return "UNKNOWN". Never return null. notes is the only field that may be an empty string.
- Do not guess or fabricate values. Only extract what is explicitly stated.
- Remove dashes and spaces from tail numbers only. Keep dashes in serial numbers.
- If the email contains Prim/Sec units, extract only the Primary (Prim) serial numbers and note "Secondary unit also present" in the notes field.
- If the email reports two DIFFERENT LRU types swapped, extract only the first one in the LRU, off_sn and on_sn fields and describe the other unit (type, OFF and ON serials) in notes.
- Upgrade language always goes in removal_reason, never in status.

Example input:
"N275: MDU Replaced@MKE for Performance Issues:
Prim S/N OFF: XA00440-51067, Sec S/N OFF: XA00440-51126
Prim S/N ON: XA00675-51884, Sec S/N ON: XA00675-51887
Fully Operational"

Example output:
{"tail": "N275", "location": "MKE", "LRU": "MDU-D", "removal_reason": "Performance Issues", "off_sn": "XA00440-51067", "on_sn": "XA00675-51884", "status": "Fully Operational", "notes": "Secondary unit also present - off_sn: XA00440-51126, on_sn: XA00675-51887"}

Example input:
"N8722: HPT Replaced@DAL. OFF: WTA-000215 ON: SN10467. Upgrade to 50W HPT. Fully Operational"

Example output:
{"tail": "N8722", "location": "DAL", "LRU": "40W HPT", "removal_reason": "Upgrade to 50W HPT", "off_sn": "WTA-000215", "on_sn": "SN10467", "status": "Fully Operational", "notes": ""}

Example input:
"TC-JHZ antenna replaced @IST for pointing errors. OFF: 194 ON: 0218. Fully Operational"

Example output:
{"tail": "TCJHZ", "location": "IST", "LRU": "Q16 GSAA", "removal_reason": "Pointing errors", "off_sn": "194", "on_sn": "0218", "status": "Fully Operational", "notes": ""}
"""


# ===========================================================================
# Deterministic LRU check (runs AFTER Groq). Serial formats, airline and
# keywords validated against the LRU Removals master (99% agreement).
# The LLM extracts the fields; this function makes the final LRU call.
# ===========================================================================
# ---------------------------------------------------------------------------
# Airline detection from tail number (tail already stripped of dashes/spaces)
# ---------------------------------------------------------------------------
def airline_from_tail(tail):
    t = re.sub(r"[^A-Z0-9]", "", str(tail).upper())
    if not t or t == "UNKNOWN":
        return None
    if t.startswith("TC"):
        return "Turkish"
    if t.startswith("TF"):
        return "Icelandair"
    if t.startswith("XA"):
        return "Aeromexico"
    if t.startswith(("LN", "SE", "EI", "OY")):
        return "Norwegian"
    if t.startswith("F"):
        return "Air France"
    if re.match(r"^N\d", t):
        return "Southwest"
    return None


# LRUs installed per airline (Reliability Dashboard Table 1 + LRU Removals master)
AIRLINE_LRUS = {
    "Southwest":  ["40W HPT", "50W HPT", "A200 CWLU", "ACU", "APSU", "LF ACU", "LF SAA",
                   "MDU-D", "Max ACU", "Max SAA", "Q05 GSAA", "SAA", "SMU Mod20"],
    "Turkish":    ["APSU", "C-KRFU", "M-KRFU", "Modman", "ModMan-S", "Q16 GSAA",
                   "TT CWLU", "TT SMU"],
    "Air France": ["50W HPT", "APSU", "CWLU II", "Modman", "Q16 GSAA", "SMU III", "T-KRFU"],
    "Norwegian":  ["50W HPT", "A200 CWLU", "ACU", "CWLU II", "MDU-D", "SAA", "SMU", "SMU II"],
    "Icelandair": ["25W HPT", "50W HPT", "A200 CWLU", "ACU", "CWLU", "MDU-D", "SAA", "SMU"],
    "Aeromexico": ["40W HPT", "ACU", "CWLU", "MDU-D", "SAA", "SMU II"],
}

# Southwest 737 MAX tails: N87xx, N88xx, N18xx (4 digits after N, optional letter suffix)
def is_southwest_max_tail(tail):
    return bool(re.match(r"^N(87|88|18)\d{2}[A-Z]{0,2}$", str(tail).upper()))


def _has(text, *words):
    return any(re.search(w, text, re.I) for w in words)


def _family(llm_lru, text):
    """Rough LRU family from the LLM guess + email text."""
    s = f"{llm_lru} {text}"
    if _has(s, r"\bHPT\b", r"\b\d{2}\s?W\b", r"transmitter"):
        return "HPT"
    if _has(s, r"KRFU"):
        return "KRFU"
    if _has(s, r"mod\s?man", r"\bMM(SX|S)?\b", r"modem\b(?!.*MDU)"):
        return "MODMAN"
    if _has(s, r"MDU"):
        return "MDU"
    if _has(s, r"APSU"):
        return "APSU"
    if _has(s, r"CWLU", r"\bWAP\b", r"wireless"):
        return "CWLU"
    if _has(s, r"\bSMU\b", r"server"):
        return "SMU"
    if _has(s, r"\bACU\b"):
        return "ACU"
    if _has(s, r"GSAA", r"\bSAA\b", r"antenna"):
        return "ANTENNA"
    return None


def classify_lru(llm_lru, tail, off_sn, on_sn="", email_text=""):
    """
    Returns the corrected LRU type, or the LLM value if no rule applies.
    Serial-number format of the REMOVED unit is the strongest signal, then
    airline (from tail), then keywords in the email.
    """
    airline = airline_from_tail(tail)
    sn = re.sub(r"\s+", "", str(off_sn).upper())
    # Fall back to the installed SN only when the removed SN is missing and the
    # swap is like-for-like (an upgrade puts a DIFFERENT LRU type on the aircraft)
    if sn in ("", "UNKNOWN", "NONE", "N/A") and not re.search(r"upgrad", str(email_text), re.I):
        sn = re.sub(r"\s+", "", str(on_sn).upper())
    text = str(email_text)
    llm_u = str(llm_lru).strip().upper()

    # Family comes from the LLM's LRU name. The rest of the email is only used to
    # find the family when the LLM returned UNKNOWN, so words like "antenna" or
    # "server" elsewhere in the email can't turn a Radome or router into an SAA/SMU.
    fam = _family(llm_lru, "")
    if fam is None:
        if llm_u not in ("", "UNKNOWN", "NONE", "N/A"):
            return llm_lru          # unit not in the LRU list -> leave Groq's raw name alone
        fam = _family("", text)


    # ---------- 1. Serial formats unique to one LRU type ----------
    if re.match(r"^WTA[-_]?\d{5,6}$", sn):
        return "40W HPT"
    if re.match(r"^GD\d{4,5}-\d{2,3}$", sn):
        return "APSU"
    if re.match(r"^X[AL]\d{4,6}(-\d+)?$", sn):
        return "MDU-D"
    if re.match(r"^YI\d{5,6}$", sn):
        return "ModMan-S"
    if re.match(r"^49\d{5,6}$", sn):                       # 492xxxx / 492011xx
        return "Modman"
    if re.match(r"^[LMN][A-C0-9]\d{4,5}[AB]\d{3,4}$", sn):  # e.g. M340799A0004, MC40799A0006
        return "M-KRFU" if str(llm_lru).upper().startswith("M-KRFU") else "C-KRFU"
    if re.match(r"^2008\d{5}$", sn):
        return "SMU III"
    if re.match(r"^20(05|06)\d{5}$", sn) or re.match(r"^20(05|06)\d{4}$", sn):
        return "SMU Mod20" if airline in (None, "Southwest") else "SMU II"
    if re.match(r"^4(6[2-9]|7\d)\d{6}$", sn) or re.match(r"^37\d{8}$", sn):
        return "A200 CWLU"

    # ---------- 2. Serial formats shared by a few types -> airline / keyword ----------
    m = re.match(r"^SN0*(\d+)$", sn)
    if m:
        n = int(m.group(1))
        if airline == "Turkish":
            if fam == "HPT":
                return "50W HPT"
            if n >= 10000:
                return "M-KRFU"
            if fam == "SMU":
                return "TT SMU"
            if fam == "CWLU" or n > 110:
                return "TT CWLU"
        elif n >= 6000 and fam in ("HPT", None):   # SN6422 ... SN22833
            return "50W HPT"

    if re.match(r"^T\d{4}$", sn):
        return "T-KRFU" if airline == "Air France" else "40W HPT"
    if re.match(r"^W\d{4,6}$", sn):
        return "25W HPT" if airline == "Icelandair" else "40W HPT"

    if fam == "HPT":
        # Bare 4-5 digit HPT serials (6422-22837) are 50W
        if re.match(r"^\d{4,5}$", sn) and 6000 <= int(sn) <= 25000:
            return "50W HPT"
        if airline == "Icelandair" and _has(text + str(llm_lru), r"25\s?W"):
            return "25W HPT"
        if airline in ("Norwegian", "Air France", "Turkish"):
            return "50W HPT"
        if airline == "Aeromexico":
            return "40W HPT"

    # ---------- 3. Short numeric serials: antenna / ACU / SMU I / CWLU ----------
    if fam == "ANTENNA" or fam == "ACU":
        is_acu = fam == "ACU"
        if airline == "Southwest":
            if _has(text + str(llm_lru), r"\bLF\b"):
                return "LF ACU" if is_acu else "LF SAA"
            if is_southwest_max_tail(tail) or _has(text + str(llm_lru), r"\bMAX\b"):
                return "Max ACU" if is_acu else "Max SAA"
            if is_acu:
                return "ACU"
            upgrade_to_gsaa = _has(text, r"upgrad\w*\s+(to\s+)?(a\s+)?GSAA", r"GSAA\s+upgrade",
                                   r"swapped\s+with\s+GSAA", r"replaced\s+with\s+(a\s+)?GSAA", r"phase\s*3")
            if upgrade_to_gsaa:
                return "SAA"                      # the unit coming OFF is the legacy SAA
            if _has(text + str(llm_lru), r"GSAA", r"Q05", r"\bPA\d?\b", r"mute relay", r"homing"):
                return "Q05 GSAA"
            if re.match(r"^0\d{3}$", sn):          # Q05 GSAA serials are zero-padded 4 digits
                return "Q05 GSAA"
            return "SAA"
        if airline in ("Turkish", "Air France"):
            return "Q16 GSAA"
        if airline in ("Norwegian", "Icelandair", "Aeromexico"):
            return "ACU" if is_acu else "SAA"

    if fam == "SMU":
        if airline == "Southwest":
            return "SMU Mod20"
        if airline == "Air France":
            return "SMU III"
        if airline == "Turkish":
            return "TT SMU"
        if airline in ("Norwegian", "Icelandair") and re.match(r"^\d{1,4}$", sn):
            return "SMU"

    if fam == "CWLU":
        if airline == "Turkish":
            return "TT CWLU"
        if airline == "Air France":
            return "CWLU II"
        if airline == "Southwest":
            return "A200 CWLU"
        if airline == "Norwegian" and re.match(r"^\d{1,4}$", sn):
            return "CWLU II"
        if airline in ("Icelandair", "Aeromexico") and re.match(r"^\d{1,4}$", sn):
            return "CWLU"

    if fam == "MDU":
        return "MDU-D"
    if fam == "KRFU":
        if airline == "Air France":
            return "T-KRFU"
    if fam == "MODMAN":
        if airline == "Turkish" and _has(text + str(llm_lru), r"\bS\b", r"SX", r"-S\b"):
            return "ModMan-S"
        return "Modman"

    return llm_lru


# Allowed LRU names, longest first so "Max SAA" wins over "SAA", "TT SMU" over "SMU", etc.
ALLOWED_LRUS = sorted([
    "25W HPT", "40W HPT", "50W HPT", "A200 CWLU", "ACU", "APSU", "C-KRFU", "CWLU", "CWLU II",
    "LF ACU", "LF SAA", "M-KRFU", "Max ACU", "Max SAA", "MDU-D", "Modman", "ModMan-S",
    "Q05 GSAA", "Q16 GSAA", "SAA", "SMU", "SMU II", "SMU III", "SMU Mod20", "T-KRFU",
    "TT CWLU", "TT SMU"], key=len, reverse=True)
_LRU_RE = re.compile(r"(?<![A-Za-z0-9-])(" + "|".join(re.escape(x) for x in ALLOWED_LRUS) + r")(?![A-Za-z0-9-])", re.I)
_CANON = {x.upper(): x for x in ALLOWED_LRUS}


def single_lru(value):
    """Guarantee exactly one LRU type. If the value is a list or names two or more
    allowed types (e.g. "40W HPT / 50W HPT"), keep the first one mentioned, which is
    the unit that came OFF in upgrade wording."""
    if isinstance(value, list):
        value = value[0] if value else "UNKNOWN"
    value = str(value).strip() or "UNKNOWN"
    found = [_CANON[m.group(1).upper()] for m in _LRU_RE.finditer(value)]
    if len(set(found)) >= 2:
        return found[0]
    if len(found) == 1 and re.search(r"[/,&+]|->|\band\b|\bto\b", value, re.I):
        return found[0]          # e.g. "SAA to GSAA", "Radome/SAA"
    return value


@app.route("/parse", methods=["POST"])
def parse_email():
    raw = ""
    try:
        data = request.get_json()
        email_text = data.get("email_text", "")

        if not email_text:
            return jsonify({"error": "No email_text provided"}), 400

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": PROMPT_TEMPLATE
                },
                {
                    "role": "user",
                    "content": email_text
                }
            ],
            model="openai/gpt-oss-120b",
            temperature=0.1
        )

        raw = chat_completion.choices[0].message.content.strip()

        # Strip markdown code blocks if present
        raw = re.sub(r"```json|```", "", raw).strip()

        # Find the first { and last } to extract just the JSON object
        start = raw.find('{')
        end = raw.rfind('}') + 1
        if start == -1 or end == 0:
            return jsonify({"error": "No JSON object found in response", "raw": raw}), 500

        json_str = raw[start:end]
        parsed = json.loads(json_str)

        # Final LRU type decision: serial format -> airline -> keywords
        llm_lru = single_lru(parsed.get("LRU", "UNKNOWN"))
        final_lru = classify_lru(
            llm_lru,
            parsed.get("tail", ""),
            parsed.get("off_sn", ""),
            parsed.get("on_sn", ""),
            f"{email_text} {parsed.get('removal_reason', '')}",
        )
        if final_lru != llm_lru:
            app.logger.info(f"LRU reclassified: {llm_lru} -> {final_lru} "
                            f"(tail={parsed.get('tail')}, off_sn={parsed.get('off_sn')})")
        parsed["LRU"] = single_lru(final_lru)

        # notes: blank instead of "UNKNOWN"
        notes = parsed.get("notes", "")
        if notes is None or str(notes).strip().upper() in ("UNKNOWN", "NONE", "N/A", "NULL"):
            parsed["notes"] = ""

        return jsonify(parsed)

    except json.JSONDecodeError as e:
        return jsonify({"error": "Failed to parse JSON", "details": str(e), "raw": raw}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def health():
    return "LRU Email Parser is running!", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
