import os
import pandas as pd
import re
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
sheet_id = "13g5gckP9kDbFef1nuIQoIghs7BpbyqOz05oWaRLa8tM"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
df = pd.read_csv(url)
print("Data loaded:", df.shape)

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["message_text"].apply(clean_text)

# Model 1: tactic detection
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

# Model 2: scam type detection
y_type = df["scam_type"]

type_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

type_model.fit(vectorizer.transform(X), y_type)

# Explanations
tactic_explanations = {
    "authority": "pretends to be police, government, or a trusted organization",
    "urgency": "pressures you to act immediately, with a tight deadline",
    "fear": "threatens you with arrest, blocking, or legal action",
    "reciprocity": "offers something (job, prize, refund) in exchange for a fee or info",
    "social_proof": "claims others have already done this or been selected",
    "greed": "promises money or rewards that seem too good to be true",
}
scam_type_labels = {
    "digital_arrest": "Digital Arrest Scam (fake police/government threat)",
    "fake_job": "Fake Job Offer Scam",
    "fake_scholarship": "Fake Scholarship Scam",
    "kyc_upi": "Fake KYC/Bank/UPI Update Scam",
    "lottery": "Fake Lottery/Prize Scam",
    "courier": "Fake Courier/Parcel Scam",
    "legitimate": "Not a scam",
}

def check_message(message):
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])

    tactic_pred = tactic_model.predict(vec)[0]
    detected = [
        tactic_columns[i]
        for i in range(len(tactic_columns))
        if tactic_pred[i] == 1
    ]

    type_pred = type_model.predict(vec)[0]
    type_label = scam_type_labels.get(type_pred, type_pred)

    if not detected and type_pred == "legitimate":
        return "This message looks legitimate. Still stay cautious with unknown senders."

    if not detected:
        return "No manipulation tactics detected. Stay cautious with unknown senders."

    result = f"LIKELY SCAM — Predicted type: {type_label}\n\n"
    result += "Manipulation tactics detected:\n"

    for tactic in detected:
        result += f"• {tactic.upper()}: {tactic_explanations[tactic]}\n"

    result += "\nWhy this matters: scammers combine these tactics to pressure you into acting quickly.\n"
    result += "\nSafety tip: Do not click links, share OTPs/PINs, or transfer money. Verify independently through official channels."

    return result


custom_css = """
#title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    color: #1E2761;
}

#subtitle {
    text-align: center;
    font-size: 16px;
    color: #5B6472;
    margin-bottom: 20px;
}
"""

with gr.Blocks(css=custom_css) as demo:

    gr.Markdown(
        "# 🛡️ TacticGuard",
        elem_id="title"
    )

    gr.Markdown(
        "AI-powered scam message analyzer — detects manipulation tactics, not just fake/real",
        elem_id="subtitle"
    )

    with gr.Row():

        with gr.Column():

            msg_input = gr.Textbox(
                label="Paste a suspicious message here",
                lines=5,
                placeholder="e.g. Your Aadhaar is linked to a case, pay within 2 hours..."
            )

            submit_btn = gr.Button(
                "🔍 Analyze Message",
                variant="primary"
            )

        with gr.Column():

            output_box = gr.Textbox(
                label="Analysis Result",
                lines=12
            )

    gr.Examples(
        examples=[
            "Your Aadhaar number is linked to a parcel containing illegal drugs. Video call our officer within 30 minutes.",
            "Congratulations! You have been selected for a Work From Home job at Amazon. Pay Rs 500 registration fee.",
            "Hi, are we still meeting for lunch today at 1pm?",
        ],
        inputs=msg_input,
        label="Try an example"
    )

    submit_btn.click(
        fn=check_message,
        inputs=msg_input,
        outputs=output_box
    )


demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)
    
