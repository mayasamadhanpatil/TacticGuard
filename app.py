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
# CLEAN TEXT
# =========================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_text"] = df["message_text"].apply(clean_text)


# =========================
# MODEL 1: TACTIC DETECTION
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
# MODEL 2: SCAM TYPE
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
# LABELS
# =========================

tactic_explanations = {
    "authority": "Pretends to be police, government, or a trusted organization",
    "urgency": "Pressures you to act immediately",
    "fear": "Uses threats such as arrest, blocking, or legal action",
    "reciprocity": "Offers a benefit in exchange for money or information",
    "social_proof": "Claims that other people have already done it",
    "greed": "Promises money or rewards that seem too good to be true"
}

scam_type_labels = {
    "digital_arrest": "Digital Arrest Scam",
    "fake_job": "Fake Job Offer Scam",
    "fake_scholarship": "Fake Scholarship Scam",
    "kyc_upi": "Fake KYC / Bank / UPI Scam",
    "lottery": "Fake Lottery / Prize Scam",
    "courier": "Fake Courier / Parcel Scam",
    "legitimate": "No Scam Detected"
}


# =========================
# ANALYSIS FUNCTION
# =========================

def check_message(message):

    if not message or not message.strip():
        return "⚠️ Please enter a message to analyze."

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

    if not detected and type_pred == "legitimate":

        return (
            "🟢 LOW RISK / LIKELY LEGITIMATE\n\n"
            "No major manipulation tactics were detected.\n\n"
            "Still stay cautious with unknown senders."
        )

    if not detected:

        return (
            "🟡 NO CLEAR TACTICS DETECTED\n\n"
            f"Predicted category: {type_label}\n\n"
            "Stay cautious with unknown messages."
        )

    result = (
        "🔴 LIKELY SCAM\n\n"
        f"📌 Predicted Type:\n{type_label}\n\n"
        "🎯 Manipulation Tactics Detected:\n"
    )

    for tactic in detected:
        result += (
            f"\n• {tactic.upper()}\n"
            f"  {tactic_explanations[tactic]}\n"
        )

    result += (
        "\n\n⚠️ Why this matters:\n"
        "Scammers often combine fear, urgency, authority, "
        "or rewards to make people act without verifying.\n\n"
        "🛡️ Safety Tip:\n"
        "Do not click suspicious links, share OTPs/PINs, "
        "or transfer money. Verify independently through "
        "official channels."
    )

    return result


# =========================
# PROFESSIONAL UI
# =========================

css = """

body {
    background: #f4f7fb;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

#hero {
    text-align: center;
    padding: 30px 10px 20px;
}

#title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

#subtitle {
    font-size: 18px;
    color: #5f6b7a;
}

.card {
    border-radius: 18px !important;
    padding: 20px !important;
}

#analyze {
    height: 55px;
    font-size: 17px;
    font-weight: 700;
}

#result textarea {
    font-size: 15px !important;
}

.footer {
    text-align: center;
    color: #6b7280;
    margin-top: 25px;
    padding: 15px;
}

"""


with gr.Blocks(css=css) as demo:

    # HERO
    gr.Markdown(
        """
        <div id="hero">
            <div id="title">🛡️ TacticGuard</div>
            <div id="subtitle">
                AI-powered scam message analyzer
            </div>
            <p>
                Detect manipulation tactics behind suspicious messages —
                not just fake or real.
            </p>
        </div>
        """
    )

    # MAIN AREA
    with gr.Row():

        with gr.Column(elem_classes="card"):

            gr.Markdown("### 🔍 Analyze a Message")

            msg_input = gr.Textbox(
                label="Suspicious Message",
                lines=8,
                placeholder=(
                    "Paste a suspicious SMS, WhatsApp message, "
                    "job offer, bank alert, or email here..."
                )
            )

            submit_btn = gr.Button(
                "🔎 Analyze Message",
                variant="primary",
                elem_id="analyze"
            )

        with gr.Column(elem_classes="card"):

            gr.Markdown("### 📊 Analysis Result")

            output_box = gr.Textbox(
                label="TacticGuard Report",
                lines=16,
                interactive=False,
                elem_id="result"
            )

    # EXAMPLES
    gr.Markdown("### 🧪 Try Sample Messages")

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
        label="Click a sample to test"
    )

    # HOW IT WORKS
    gr.Markdown(
        """
        ---

        ### ⚙️ How TacticGuard Works

        **1️⃣ Message Input** → Paste a suspicious message

        **2️⃣ AI Analysis** → Machine-learning models analyze the text

        **3️⃣ Tactic Detection** → Detects psychological manipulation tactics

        **4️⃣ Scam Classification** → Predicts the possible scam category

        **5️⃣ Safety Guidance** → Provides practical precautions

        ---

        ### 🛡️ Stay Safe Online

        Never share **OTP, PIN, passwords or banking details** with
        unknown people. Always verify suspicious claims through
        official channels.
        """
    )

    gr.Markdown(
        """
        <div class="footer">
            🛡️ TacticGuard • AI-based Scam Awareness Project
        </div>
        """
    )

    submit_btn.click(
        fn=check_message,
        inputs=msg_input,
        outputs=output_box
    )


# =========================
# START SERVER
# =========================

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)