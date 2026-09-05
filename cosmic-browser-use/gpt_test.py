import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
    model=os.getenv("SUMMARY_LLM_MODEL", "gpt-5.6-luna"),
    input="Classify this email as 'Urgent' or 'Routine': [Email Content]",
    reasoning={"effort": "high"},
    text={"verbosity": "low"} 
)

print(response.output_text)
