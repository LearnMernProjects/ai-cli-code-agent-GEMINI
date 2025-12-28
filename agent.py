import cmd
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os
import json
import requests

from main import get_weather
load_dotenv()

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/",
    api_key="AIzaSyBTiydPIe4hceZno-JwnmQRaCEG2Xw7JRA",
)
def run_command(cmd: str):
    result = os.system(cmd)
    return result

SYSTEM_PROMPT = """
You are an expert AI Assistant in revolving user queries using chain of thought ,You work on START, PLAN and OUTPUT steps.
You need to plan what's need to be done. The PLAN can be multiple steps. Once you think enough PLAN what needs to be done. The PLAN can be multiple steps.
Once you think enough PLAN has been done, finally you can give an OUTPUT.
You can also call a tool iff required from the list ofavailable tools.
For every call you must wait for the observe step which is the output from the called tool.
Rules:
-Strictly Follow the given JSON output format 
-Only run 1 step at a time
-The sequence of steps is START (where user gives an input), PLAN(That can be multiple times) and finally OUTPUT (which is going to be displayed to the user).
OUTPUT JSON format:
{{"step":"START"|"PLAN"|"TOOL" | "OUTPUT"| "TOOL","content":"string", "tool":"tool_name and string", "input":"input to tool", "output":"output from tool"}}

Available Tools:
- get_weather(); Takes city name as an input string and return the weather information as a string for particular city.
- run_command(); Takes a command as an input string and return the output on user's system and return the output from that command.
1. Weather Tool:
Example 1:
START:hey, can you solve 2+4*5/32
PLAN:{{"step":"PLAN","content":"seeems like user is interested in math problem"}}
PLAN:{{"step":"PLAN","content":"Yes , The BODMAS is correct thing to be done here"}}
PLAN:{{"step":"PLAN","content":"we must first multiply 4*5 which is 20 and continue giving till the final answer comes"}}
PLAN:{{"step":"PLAN","content":"we must divide 20 with 32 and continue giving till the final answer comes"}}
PLAN:{{"step":"PLAN","content":"we must add 2 in final result and continue giving till the final answer comes"}}
PLAN:{{"step":"TOOL","tool":"get_weather", "input":"new york"}}
PLAN:{{"step":"TOOL","tool":"get_weather", "output":"Fetching the weather of New York is "}}
OUTPUT:{{"step":"OUTPUT", "content":"the weather of New York is "}}


OUTPUT:{{"step":"OUTPUT","content":"3.5"}}

Example 2:


START:What is the weather in New York?
PLAN:{{"step":"PLAN","content":"seeems like user is interested in getting weather of New York"}}
PLAN:{{"step":"PLAN","content":"Lets see we have available tool"}}
PLAN:{{"step":"PLAN","content":"Great we have available tool for given query"}}
PLAN:{{"step":"OBSERVE","content":"I neet to call get_weather tool for getting live weather information"}}
PLAN:{{"step":"TOOL","content":"CALL get_weather() tool for new york"}}
PLAN:{{"step":"PLAN","content":"Great got the weather information from tool"}}
PLAN:{{"step":"PLAN","content":"Now I can give final output to user, weather in that city is"}}

OUTPUT:{{"step":"OUTPUT","content":"3.5"}}
"""
class MyOutputFormat(BaseModel):
    step: str = Field(... , description="The ID ofthe step in the process. Example:- START, PLAN, TOOL, OUTPUT")
    content: Optional[str] = Field( None, description="The content or message associated with the step.")
    tool: Optional[str] = Field( None, description="The name of the tool being used in this step.")
    input: Optional[str] = Field( None, description="The input provided to the tool.")
    output: Optional[str] = Field( None, description="The output received from the tool.")
# Message history setup
message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
user_query = input("Write your query: ")
message_history.append({"role": "user", "content": user_query})

available_tools = {
    "get_weather": lambda city: get_weather(city),
    "run_command": lambda command: str(run_command(command)),
}

while True:
    response = client.chat.completions.parse(
        model="gemini-2.5-flash",
        response_format=MyOutputFormat,
        messages=message_history
    )
    raw_result = response.choices[0].message.content
    message_history.append({"role": "assistant", "content": raw_result})

    # Extract first valid JSON object from raw_result
    parsed_result = response.choices[0].message.parsed
    
    

    if parsed_result is None:
        print("ERROR: Could not parse response as JSON")
        print("Raw response:", raw_result[:200])
        break

    if parsed_result.step == "START":
        print("done:", parsed_result.content)
        continue
    if parsed_result.step == "PLAN":
        print("getting result:", parsed_result.content)
        continue
    if parsed_result.step == "TOOL":
        tool_to_call = parsed_result.tool
        tool_input = parsed_result.input
        print(f"calling tool: {tool_to_call} with input: {tool_input}")
        try:
            tool_response = available_tools[tool_to_call](tool_input)
        except Exception as e:
            tool_response = f"Error calling tool: {str(e)}"
        message_history.append({"role": "user", "content": json.dumps({
            "step": "OBSERVE",
            "output": tool_response
        })})
        continue
    if parsed_result.step == "OUTPUT":
        print("final result:", parsed_result.content)
        break
    else:
        # Handle unexpected steps or continue if needed
        print(f"Step: {parsed_result.step}")
        continue
