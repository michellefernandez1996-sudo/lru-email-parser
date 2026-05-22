from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import json
import re

app = Flask(__name__)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

PROMPT_TEMPLATE = """
You are a data extraction assistant. Extract structured data from an LRU removal email.

The input contains the subject line and body of an email. Extract the following fields and return ONLY a single valid JSON object with no explanation, no markdown, no code blocks, no array brackets.

Fields to extract:
- tail: Aircraft tail/registration number. Starts with a country/airline registration prefix (N, TC, TF, F, SE, G, EC, etc.) followed by numbers and/or letters. Remove any dashes or spaces (example: "N-8843S" → "N8843S", "TK 583" → "TK583").
- location: Airport code. Usually appears after "@" symbol (example: @DEN → DEN). May also appear as a standalone 3-letter IATA airport code.
- LRU: The Line Replaceable Unit being swapped. Match to the closest name from this list:
    "40W HPT", "50W HPT", "A200 CWLU", "ACU", "APSU", "C-KRFU", "CWLU", "CWLU II",
    "Q05 GSAA", "Q06 GSAA", "Max ACU", "Max SAA", "MDU-D", "M-KRFU", "Modman",
    "ModMan-S", "SAA", "SMU", "SMU III", "SMU Mod20", "T-KRFU", "TT CWLU", "TT SMU"
    These may also appear shortened or abbreviated (examples: "MM" → "Modman", "ModManSX" → "ModMan-S").
    If no match is found, return the raw name as written in the email.
- removal_reason: Why the unit was removed or failed. Includes operational failures AND upgrade descriptions (examples: "upgraded to ModManSX", "upgrade to 50W HPT").
- off_sn: Serial number of the unit removed. Usually after "OFF:", "SN OFF:", "Removed SN:", or "Prim S/N OFF:".
- on_sn: Serial number of the unit installed. Usually after "ON:", "SN ON:", "Installed SN:", or "Prim S/N ON:".
- status: Operational status after the swap ONLY (examples: Fully Operational, Intermittent, Removed Only, Pending Test). Never put upgrade descriptions here.
- notes: Any additional information not captured in the above fields.

Rules:
- Return ONLY a single valid JSON object. No array, no wrapper, no explanation, no markdown, no code blocks.
- If a field cannot be found, return "UNKNOWN". Never return null or empty strings.
- Do not guess or fabricate values. Only extract what is explicitly stated.
- Remove dashes and spaces from tail numbers only. Keep dashes in serial numbers.
- If the email contains Prim/Sec units, extract only the Primary (Prim) serial numbers and note "Secondary unit also present" in the notes field.
- Upgrade language always goes in removal_reason, never in status.

Example input:
"N275: MDU Replaced@MKE for Performance Issues:
Prim S/N OFF: XA00440-51067, Sec S/N OFF: XA00440-51126
Prim S/N ON: XA00675-51884, Sec S/N ON: XA00675-51887
Fully Operational"

Example output:
{"tail": "N275", "location": "MKE", "LRU": "MDU-D", "removal_reason": "Performance Issues", "off_sn": "XA00440-51067", "on_sn": "XA00675-51884", "status": "Fully Operational", "notes": "Secondary unit also present - off_sn: XA00440-51126, on_sn: XA00675-51887"}

Email text to extract from:
{email_text}
"""

@app.route("/parse", methods=["POST"])
def parse_email():
    try:
        data = request.get_json()
        email_text = data.get("email_text", "")

        if not email_text:
            return jsonify({"error": "No email_text provided"}), 400

        prompt = PROMPT_TEMPLATE.format(email_text=email_text)
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Strip markdown code blocks if present
        raw = re.sub(r"```json|```", "", raw).strip()

        parsed = json.loads(raw)
        return jsonify(parsed)

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse JSON from AI response", "raw": raw}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def health():
    return "LRU Email Parser is running!", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
