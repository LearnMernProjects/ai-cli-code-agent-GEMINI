from openai import OpenAI
from dotenv import load_dotenv
import requests
import time

load_dotenv()
client = OpenAI()

def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    # Retry logic with delay
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                return f"The weather in {city} is {response.text.strip()}"
            else:
                return f"Unable to fetch weather data for {city}."
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            else:
                return f"Could not fetch weather for {city} after {max_retries} attempts: {str(e)}"

def main():
    while True:
        user_query = input("WHat do you want to ask the bot? ")
  
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": user_query}
            ]
        )

        print("botResponse:", response.choices[0].message.content)


