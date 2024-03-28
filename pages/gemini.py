import streamlit as st
import google.generativeai as genai
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


@st.cache_resource
def load_models():
  print("Loading VN Whisper ...")

  VN_model_id = "vinai/PhoWhisper-small"

  VN_model = AutoModelForSpeechSeq2Seq.from_pretrained(
      VN_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
  )
  VN_model.to(device)

  VN_processor = AutoProcessor.from_pretrained(VN_model_id)

  vn_pipe = pipeline(
      "automatic-speech-recognition",
      model=VN_model,
      tokenizer=VN_processor.tokenizer,
      feature_extractor=VN_processor.feature_extractor,
      torch_dtype=torch_dtype,
      device=device,
  )

  print("Loading OpenAI Whisper ...")

  EN_model_id = "openai/whisper-small"

  EN_model = AutoModelForSpeechSeq2Seq.from_pretrained(
      EN_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
  )
  EN_model.to(device)

  EN_processor = AutoProcessor.from_pretrained(EN_model_id)

  en_pipe = pipeline(
      "automatic-speech-recognition",
      model=EN_model,
      tokenizer=EN_processor.tokenizer,
      feature_extractor=EN_processor.feature_extractor,
      max_new_tokens=128,
      chunk_length_s=30,
      batch_size=16,
      torch_dtype=torch_dtype,
      device=device,
  )

  return vn_pipe, en_pipe

def transcribe(audio, lang):
    vn_pipe, en_pipe = load_models()
    result = en_pipe(audio, generate_kwargs={"language": lang}) if lang != 'vietnamese' else vn_pipe(audio)
    return result['text']

genai.configure(api_key="AIzaSyDlhZuEB3Pawz6YMJERLfz-phMuWFKD0Xg")

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

prompt_parts = [
  "input: Customer and Employee Call Transcription Analysis Prompt Template\nPurpose: To analyze customer and employee call transcripts to understand satisfaction levels, identify pain points, and evaluate employee performance.\nFormat:\n\nTranscript",
  "output: Customer Analysis\n\nOverall Satisfaction Score: Rate the customer's satisfaction level on a scale of 1-5 (1 = Very Dissatisfied, 5 = Very Satisfied).\nReasons for Satisfaction: Identify specific statements or actions in the transcript that indicate the customer's satisfaction.\nReasons for Dissatisfaction: Identify specific statements or actions in the transcript that indicate the customer's dissatisfaction.\n\nPain Points Analysis\n\nProblem Areas: Describe the specific issues or challenges mentioned by the customer.\nImpact of Pain Points: Assess the severity and impact of the pain points on the customer's experience.\nPossible Solutions: Suggest potential solutions or actions that could address the identified pain points.\n\nEmployee Analysis\n\nCommunication Skills: Evaluate the employee's ability to communicate effectively and clearly.\nEmpathy and Understanding: Assess the employee's ability to understand and empathize with the customer's needs.\nProblem-Solving Skills: Determine the employee's ability to identify and resolve customer issues efficiently.\nProfessionalism and Courtesy: Evaluate the employee's overall demeanor and professionalism during the interaction.\n\nAdditional Notes\n\nOther Analysis: Include any additional insights or observations that may be relevant to the analysis.\nClassification: Categorize the interaction based on relevant factors (e.g., product inquiry, billing issue, support request).\nAction Items: List specific actions or recommendations that should be taken to improve customer satisfaction, address pain points, or enhance employee performance.\n\nEmployee Performance Rating:\n\nOverall Performance Score: Rate the employee's overall performance on a scale of 1-5 (1 = Needs Improvement, 5 = Excellent).\nStrengths: Identify specific areas where the employee excelled.\nAreas for Improvement: Provide constructive feedback on areas where the employee could improve their performance.",
]


def main():
    st.set_page_config(page_title="Customer's experience analysis", page_icon=":chart_with_upwards_trend:")
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>Gemini Man</h1>
            <h3>The perfect hitman to end your customer's pain points.</h3>
            <p>Sentimental analysis to better understand customer. Deliver a rating for both customer's opinion and the servicer's performance </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    page_bg_img = '''
    <style>
      body {
        background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
        background-size: cover;
      }
    </style>
    '''

    uploaded_file = st.file_uploader("Upload an audio file(MP3, WAV)", type=['mp3', 'wav'], accept_multiple_files=False)

    lang_option = st.selectbox(
    'Conversation language?',
    ('vietnamese', 'english'))

    st.write('You selected:', lang_option)

    if uploaded_file != None:
        audio_bytes = uploaded_file.read()
        transcription = transcribe(audio_bytes, lang_option)
        print(transcription)
        with st.expander("Transcription Preview"):
            st.write(transcription)
    text_input =  st.text_area("Is there any other information that you want to ask?")
    if st.button("Analysis"):
        template = text_input+prompt_parts[0]+transcription+prompt_parts[1]
        formatted_template = template.format(text_input=text_input)
        response = model.generate_content(formatted_template)
        analysis = response.text
        st.write("Analysis and Evaluate:")
        st.write(analysis)

if __name__ == "__main__":
    main()