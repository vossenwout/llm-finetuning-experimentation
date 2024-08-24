import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, before_log, after_log
import logging
import json

from tqdm import tqdm

from groq import Groq

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv("config/.env.secret")

# MODEL = "mixtral-8x7b-32768"
MODEL_LLAMA = "llama3-70b-8192"
MODEL_LLAMA_8B = "llama3-8b-8192"
MODEL_MIXTRAL = "mixtral-8x7b-32768"

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


@retry(
    stop=stop_after_attempt(7),  # Stop after 5 attempts
    wait=wait_exponential(min=1, max=100),  # Wait exponentially between retries
)
def ask_groq(question, model):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        model=model,
    )
    return chat_completion.choices[0].message.content


def load_text_from_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


def chunk_text(text, max_chunk_size=2000):
    chunks = []
    current_chunk = ""
    for line in text.split("\n"):
        if len(current_chunk) + len(line) > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = ""
        current_chunk += line + "\n"
    chunks.append(current_chunk)
    return chunks


def format_interview_chunk_clean_prompt(chunk):
    prompt = f"""The following is a the transcript of an interview with TRUMP. However in the interview the text of the same speaker sometimes spans multiple lines. I want you to clean the transcript by concatenating the text of the same speaker such that every new line represents a new speaker.
    
    Example 1 
    INPUT:
    Speaker A: I don't think that's fair
    What you said about that girl isn't right.
    TRUMP: You know you are right. 
    
    OUTPUT:
    Speaker A: I don't think that's fair. What you said about that girl isn't right.
    TRUMP: You know you are right.

    INPUT:
    Speaker A: I don't think that's fair
    Speaker A: What you said about that girl isn't right.

    OUTPUT:
    Speaker A: I don't think that's fair. What you said about that girl isn't right.

    interview:
    {chunk}
    """
    return prompt


def clean_interview(interview_file_path):
    interview_text = load_text_from_file(interview_file_path)
    interview_chunks = chunk_text(interview_text)
    cleaned_interview = ""
    for chunk in tqdm(interview_chunks):
        cleaned_chunk = ask_groq(
            format_interview_chunk_clean_prompt(chunk), model=MODEL_LLAMA
        )
        cleaned_interview += cleaned_chunk
    return cleaned_interview


def save_text_to_file(text, file_path):
    with open(file_path, "w") as file:
        file.write(text)


file_path = "finetune_dataset/datasets/trump/raw_interviews/ccn_town_hall.txt"

clean_interview_text = clean_interview(file_path)

print(clean_interview_text)

save_text_to_file(
    clean_interview_text,
    "finetune_dataset/datasets/trump/clean_interviews/ccn_town_hall.txt",
)
