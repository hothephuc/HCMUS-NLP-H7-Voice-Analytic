import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

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

def transcribe(audio, lang='vn'):
    result = en_pipe(audio, generate_kwargs={"language": lang}) if lang != 'vn' else vn_pipe(audio)
    return result['text']

# print("Trancripting ...")
# transcription = transcribe('sample/VN_Sample1.wav', 'vn')

# print(transcription)