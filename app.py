import os
import pandas as pd
import re
import gradio as gr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# =========================
# LOAD DATA
# =========================

sheet_id = "13g5gckP9kDbFef1nuIQoIghs7BpbyqOz05oWaRLa8tM"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

df = pd.read_csv(url)
print("Data loaded:", df.shape)


# =========================
# TEXT CLEANING
# =========================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_text"] = df["message_text"].apply(clean_text)


# =========================
# TACTIC MODEL
# =========================

tactic_columns = [
    "authority",
    "urgency",
    "fear",
    "reciprocity",
    "social_proof",
    "greed"
]

X = df["clean_text"]
y_tactics = df[tactic_columns]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_tactics,
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(max_features=3000)

X_train_vec = vectorizer.fit_transform(X_train)

tactic_model = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
)

tactic_model.fit(X_train_vec, y_train)

print("Tactic model trained")


# =========================
# SCAM TYPE MODEL
# =========================

y_type = df["scam_type"]

type_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

type_model.fit(
    vectorizer.transform(X),
    y_type
)


# =========================
# EXPLANATIONS
# =========================

tactic_explanations = {
    "authority": "Pretends to be police, government, bank, or another trusted organization.",
    "urgency": "Pressures you to act immediately or within a short deadline.",
    "fear": "Uses threats such as arrest, account blocking, or legal action.",
    "reciprocity": "Offers a benefit in exchange for money or personal information.",
    "social_proof": "Claims that other people have already participated or been selected.",
    "greed": "Promises money, prizes, jobs, or rewards that seem too good to be true."
}


scam_type_labels = {
    "digital_arrest": "Digital Arrest Scam",
    "fake_job": "Fake Job Offer Scam",
    "fake_scholarship": "Fake Scholarship Scam",
    "kyc_upi": "Fake KYC / Bank / UPI Scam",
    "lottery": "Lottery / Prize Scam",
    "courier": "Fake Courier / Parcel Scam",
    "legitimate": "No Scam Detected"
}


# =========================
# ANALYSIS
# =========================

def check_message(message):

    if not message or not message.strip():
        return (
            "⚠️  PLEASE ENTER A MESSAGE\n\n"
            "Paste a suspicious SMS, WhatsApp message, "
            "email, job offer, or bank alert to analyze it."
        )

    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])

    tactic_pred = tactic_model.predict(vec)[0]

    detected = [
        tactic_columns[i]
        for i in range(len(tactic_columns))
        if tactic_pred[i] == 1
    ]

    type_pred = type_model.predict(vec)[0]

    type_label = scam_type_labels.get(
        type_pred,
        type_pred
    )

    # LEGITIMATE
    if not detected and type_pred == "legitimate":
        return (
            "🟢  LOW RISK / LIKELY LEGITIMATE\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📌 Category: {type_label}\n\n"
            "No major manipulation tactics were detected.\n\n"
            "💡 Still stay cautious with unknown senders."
        )

    # NO TACTICS
    if not detected:
        return (
            "🟡  NO CLEAR MANIPULATION TACTICS\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📌 Predicted Category: {type_label}\n\n"
            "The model did not detect the listed manipulation tactics.\n\n"
            "💡 Stay cautious and verify suspicious claims independently."
        )

    # SCAM
    result = (
        "🔴  LIKELY SCAM\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📌 Predicted Scam Type\n"
        f"{type_label}\n\n"
        "🎯 Manipulation Tactics Detected\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
    )

    for tactic in detected:
        result += (
            f"\n🔸 {tactic.upper()}\n"
            f"   {tactic_explanations[tactic]}\n"
        )

    result += (
        "\n\n⚠️ Why this matters\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "Scammers often combine fear, urgency, authority, "
        "or rewards to pressure people into acting quickly "
        "without verification.\n\n"
        "🛡️ Safety Checklist\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "• Do not click suspicious links\n"
        "• Never share OTPs or PINs\n"
        "• Do not transfer money under pressure\n"
        "• Verify through official channels"
    )

    return result


# =========================
# PROFESSIONAL DESIGN
# =========================

css = """

body {
    background: linear-gradient(135deg, #eef4ff 0%, #f8fbff 50%, #eef8f5 100%);
}

.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
    padding: 20px !important;
}

#hero {
    text-align: center;
    padding: 35px 15px 25px;
}

#logo {
    font-size: 58px;
    margin-bottom: 5px;
}

#title {
    font-size: 46px;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0;
}

#tagline {
    font-size: 19px;
    margin-top: 10px;
    color: #5b6575;
}

#description {
    font-size: 15px;
    color: #6b7280;
    max-width: 700px;
    margin: 12px auto;
}

.panel {
    background: white;
    border-radius: 20px;
    padding: 22px;
    box-shadow: 0 8px 30px rgba(20, 40, 80, 0.08);
    border: 1px solid #e6ebf2;
}

#analyze {
    height: 58px;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 700;
}

#result textarea {
    font-size: 15px !important;
    line-height: 1.55 !important;
}

.section-title {
    text-align: center;
    font-size: 25px;
    font-weight: 700;
    margin: 35px 0 15px;
}

.info-card {
    background: white;
    border-radius: 16px;
    padding: 18px;
    border: 1px solid #e6ebf2;
    text-align: center;
    box-shadow: 0 5px 20px rgba(20, 40, 80, 0.05);
}

.footer {
    text-align: center;
    color: #6b7280;
    margin-top: 35px;
    padding: 20px;
    font-size: 13px;
}

"""


# =========================
# APP
# =========================

with gr.Blocks(css=css) as demo:

    # HERO
    gr.HTML(
        """
        <div id="hero">

            <div id="logo">🛡️</div>

            <div id="title">
                TacticGuard
            </div>

            <div id="tagline">
                AI-Powered Scam Message Analyzer
            </div>

            <div id="description">
                Detect psychological manipulation tactics hidden
                inside suspicious messages — not just fake or real.
            </div>

        </div>
        """
    )


    # MAIN ANALYZER
    with gr.Row():

        with gr.Column(elem_classes="panel"):

            gr.Markdown("### 🔍 Analyze a Suspicious Message")

            msg_input = gr.Textbox(
                label="Message",
                lines=9,
                placeholder=(
                    "Paste an SMS, WhatsApp message, "
                    "job offer, bank alert, email, or suspicious message here..."
                )
            )

            submit_btn = gr.Button(
                "🔎 ANALYZE MESSAGE",
                variant="primary",
                elem_id="analyze"
            )


        with gr.Column(elem_classes="panel"):

            gr.Markdown("### 📊 TacticGuard Analysis")

            output_box = gr.Textbox(
                label="Analysis Report",
                lines=17,
                interactive=False,
                elem_id="result"
            )


    # SAMPLE MESSAGES
    gr.HTML(
        '<div class="section-title">🧪 Try Sample Messages</div>'
    )

    gr.Examples(
        examples=[
            [
                "Your Aadhaar number is linked to a parcel containing illegal drugs. Video call our officer within 30 minutes."
            ],
            [
                "Congratulations! You have been selected for a Work From Home job at Amazon. Pay Rs 500 registration fee."
            ],
            [
                "Hi, are we still meeting for lunch today at 1pm?"
            ]
        ],
        inputs=msg_input,
        label="Select a sample message"
    )


    # HOW IT WORKS
    gr.HTML(
        '<div class="section-title">⚙️ How TacticGuard Works</div>'
    )

    with gr.Row():

        gr.HTML(
            """
            <div class="info-card">
                <h3>1️⃣ Input</h3>
                <p>Paste a suspicious message.</p>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="info-card">
                <h3>2️⃣ AI Analysis</h3>
                <p>Machine-learning models analyze the text.</p>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="info-card">
                <h3>3️⃣ Detect</h3>
                <p>Identify manipulation tactics and scam type.</p>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="info-card">
                <h3>4️⃣ Protect</h3>
                <p>Get practical safety guidance.</p>
            </div>
            """
        )


    # SAFETY SECTION
    gr.HTML(
        """
        <div class="section-title">
            🛡️ Stay Safe Online
        </div>

        <div class="info-card">

            <h3>Remember the basics</h3>

            <p>
            Never share <b>OTP • PIN • Password • Banking Details</b>
            with unknown people.
            </p>

            <p>
            If a message creates extreme urgency or fear,
            stop and verify it through an official source.
            </p>

        </div>
        """
    )


    # FOOTER
    gr.HTML(
        """
        <div class="footer">
            🛡️ <b>TacticGuard</b>
            &nbsp;•&nbsp;
            AI-Based Scam Awareness Project
            <br>
            Built for cybersecurity awareness and education.
        </div>
        """
    )


    submit_btn.click(
        fn=check_message,
        inputs=msg_input,
        outputs=output_box
    )


# =========================
# RENDER SERVER
# =========================

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)