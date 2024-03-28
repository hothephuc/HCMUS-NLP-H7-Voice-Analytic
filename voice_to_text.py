# Start by making sure the `assemblyai` package is installed.
# If not, you can install it by running the following command:
# pip install -U assemblyai
#
# Note: Some macOS users may need to use `pip3` instead of `pip`.

import assemblyai as aai

# Replace with your API key
aai.settings.api_key = "c99ac5c26f21406a8ac3e687bff27787"

# URL of the file to transcribe
#FILE_URL = "https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3"

# You can also transcribe a local file by passing in a file path
# FILE_URL = './path/to/file.mp3'

FILE_URL = "/home/phuc/university_of_science/third_year/second_sem/nlp/project/HCMUS-NLP-H7-Voice-Analytic/sample/SampleCall1.mp3"

config = aai.TranscriptionConfig(speaker_labels=True)

transcriber = aai.Transcriber()
transcript = transcriber.transcribe(
  FILE_URL,
  config=config
)

f = open("/home/phuc/university_of_science/third_year/second_sem/nlp/project/HCMUS-NLP-H7-Voice-Analytic/example/call_1.txt", "a")

for utterance in transcript.utterances:
  f.write(f"Speaker {utterance.speaker}: {utterance.text} \n")
