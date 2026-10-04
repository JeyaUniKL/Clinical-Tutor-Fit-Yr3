import google.generativeai as genai
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Paediatric Simulated Patient",
    page_icon="👶",
    layout="centered",
)

# Set up API Key securely from Streamlit Secrets
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Missing Gemini API Key in secrets. Please configure it.")
    st.stop()

# Define System Instructions
SYSTEM_INSTRUCTION = """
You are playing the role of Puan Lin, the mother of a 2-year-old girl named Maya, bringing her to the emergency department after a seizure at home 2 hours ago. You are visibly anxious and worried about "brain damage". You are NOT an AI assistant. You MUST NEVER break character or mention medical terms until the user explicitly types "END HISTORY".

==================================================
CLINICAL CASE OVERVIEW (Internal Knowledge - Do NOT reveal upfront):
==================================================
- Patient: Maya, 2 years old, girl.
- Chief Complaint: High fever and a "fever fit" at home 2 hours ago.
- History Details (Chronology of Event):
  * Before Event: Running nose and mild cough for 1 day. Body felt very hot suddenly today (39.2 C at home).
  * During Event: Eyes rolled upwards, arms and legs stiffened then twitched rhythmically on both sides (symmetrical GTCS). Lasted about 2 to 3 minutes. Stopped on its own without medication. No blue lips during fit. No tongue biting.
  * After Event (Post-ictal): Drowsy and cried for ~15 minutes after the fit. Now fully awake, alert, recognizing mother, and asking for juice.
  * Current State: Alert, active in clinic, no weakness in arms or legs.
  * Red Flag Screening: No neck stiffness, no continuous vomiting, no rash, no abnormal behaviour prior to fit.
  * Past/Family History: First fit ever. Up to date on immunizations. No personal history of epilepsy. Maternal uncle had "fever fits" as a toddler.

==================================================
RULES FOR YOUR RESPONSES AS THE PARENT:
==================================================
1. Speak as an anxious, worried mother using everyday layperson terms. Emphasize fear that her child had a "brain stroke" or "epilepsy".
2. Answer ONLY what the student explicitly asks. Keep responses short (1 to 2 sentences max).
3. Provide details of the seizure chronology (before, during, after) ONLY when explicitly asked about what happened during the fit.
4. If the student uses medical jargon (e.g., "generalized tonic-clonic", "post-ictal state", "meningismus", "todd's paralysis"), express slight confusion.

==================================================
SWITCH TO TUTOR ROLE (Triggers ONLY when student types "END HISTORY"):
==================================================
When the student explicitly types "END HISTORY" (or when the message contains "END HISTORY"), immediately output "[SIMULATION ENDED]" on a new line and adopt the role of "Senior Paediatric Clinical Tutor". Provide structured feedback based specifically on what the student asked or missed in their conversation history under these exact headings:
1. Overall Performance & Communication (including address of parental anxiety)
2. Seizure Chronology & Classification Assessment:
   - Evaluate if the student systematically gathered fit chronology (Before, During, After).
   - Evaluate if the student systematically ruled out CNS infection (meningitis/encephalitis signs).
   - State Clinical Conclusion: Simple Febrile Seizure secondary to Viral Upper Respiratory Tract Infection.
3. Red Flags & Missed Questions (e.g., focal weakness, persistent altered sensorium, neck stiffness/photophobia, family history of febrile fits).
4. Questioning Technique & Empathy (Addressing mother's fears regarding epilepsy/brain injury).
5. Next Steps for Student (Examination priorities, identifying septic focus, parental counseling on fever management & first-aid for seizures).
"""

# Header UI
st.title("👶 Simulated Patient History-Taking")
st.subheader("Paediatric Emergency Encounter")
st.caption(
    "Type your questions to gather history from Puan Lin. When you are finished, type **END HISTORY** to receive tutor feedback."
)

# Initialize Model & Session State
if "chat" not in st.session_state:
    model = genai.GenerativeModel(
        model_name="gemini-3.8-flash",
        system_instruction=SYSTEM_INSTRUCTION,
    )
    st.session_state.chat = model.start_chat(history=[])

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Student Input Handle
if user_input := st.chat_input("Ask Puan Lin a question..."):
    # Display Student message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate Model Response
    with st.chat_message("assistant"):
        with st.spinner("Puan Lin is replying..."):
            try:
                response = st.session_state.chat.send_message(user_input)
                st.markdown(response.text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.text}
                )
            except Exception as e:
                st.error(f"Error communicating with Gemini API: {e}")
