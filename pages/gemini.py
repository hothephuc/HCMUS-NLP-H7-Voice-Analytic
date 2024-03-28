import streamlit as st
import google.generativeai as genai


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

prompt_parts[0]= 
response = model.generate_content(prompt_parts)
print(response.text)

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

    uploaded_file = st.file_uploader("Upload a csv file", type=['csv'])

    if uploaded_file != None:
        data = pd.read_csv(uploaded_file)
        with st.expander("Dataframe Preview"):
            st.write(data.head(5))
            
        text_input = st.text_area("Enter a prompt about the data (e.g., 'summarize key statistics', 'find trends'):")
        if st.button("Generate"):
            if text_input:
                with st.spinner("Generating response..."):
                    template = f"""
                        Analyze the data in the uploaded CSV file based on the prompt: {text_input}
                        The dataset: {data}
                    """
                    formatted_template = template.format(text_input=text_input)
                    #st.write(formatted_template)
            else:
                st.warning("Please enter a prompt")

            response = model.generate_content(formatted_template)
            analysis = response.text
            st.write("Analysis:")
            st.write(analysis)

if __name__ == "__main__":
    main()