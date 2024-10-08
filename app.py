from dotenv import load_dotenv
import chainlit as cl

load_dotenv()

# Import the function from movie_functions.py
from movie_functions import *

import random

# Note: If switching to LangSmith, uncomment the following, and replace @observe with @traceable
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable
# client = wrap_openai(openai.AsyncClient())

from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI
 
client = AsyncOpenAI()

gen_kwargs = {
    "model": "gpt-4o",
    "temperature": 0.2,
    "max_tokens": 500
}

SYSTEM_PROMPT = """\
You are a movie buff and you keep track of what the latest movies are.
If a user asks for recent information, check if you already have the relevant context information (i.e. now playing movies or showtimes for movies).
If you do, then output the contextual information.
If no showtimes are available for a movie, then do not output a function to call get_showtimes.
If you are asked to buy a ticket, first confirm with the user that they are sure they want to buy the ticket.
Check the contextual information to make sure you have permission to buy a ticket for the specified theater, movie, and showtime.
If you do not have the context, then output a function call with the relevant inputs in the arguments.
If you need to fetch more information, then pick the relevant function and only output the function call. 
Call functions using Python syntax in plain text, no code blocks.

You have access to the following functions:
- get_now_playing_movies()
- get_showtimes(title, location)
- buy_ticket(theater, movie, showtime)
- get_reviews(movie_id)
- pick_random_movie(movies)

When outputting the function for get_showtimes, do not include the variable names.
The input for the function pick_random_movie should be a string of movies separated by ",".
"""

# SYSTEM_PROMPT_WITH_CONTEXT = """\
# You are a movie buff and you keep track of what the latest movies are.
# If your last response was with a function call such as the following:
# - get_now_playing_movies()
# - get_showtimes(title, location)
# - buy_ticket(theater, movie, showtime)
# - get_reviews(movie_id)

# Review the message history for context in api_fetch_context. If you have the information, respond with it.
# """

@observe
@cl.on_chat_start
def on_chat_start():    
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)

@observe
async def generate_response(client, message_history, gen_kwargs):
    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(messages=message_history, stream=True, **gen_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)
    
    await response_message.update()

    return response_message

        # now_playing_movies = await get_now_playing_movies()

last_user_message = ""

@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})
    last_user_message = message.content
    
    response_message = await generate_response(client, message_history, gen_kwargs)

    # Check response for functions
    while any(keyword in response_message.content for keyword in [
        'get_now_playing_movies(',
        'get_showtimes(',
        'buy_ticket(',
        'get_reviews(',
        'pick_random_movie(',
        ]):

        if "get_now_playing_movies" in response_message.content:
            now_playing = get_now_playing_movies()
            message_history.append({"role": "system", "content": f"Fetched Context: {now_playing}"})
        elif "get_showtimes" in response_message.content:
            function_call = response_message.content
            start = function_call.find('(') + 1
            end = function_call.find(')')

            # Extract the arguments substring
            arguments = function_call[start:end]

            # Split the arguments by comma and strip any whitespace
            title, location = [arg.strip() for arg in arguments.split(',')]
            print(f"title: {title}, location: {location}")
            showtimes = get_showtimes(title, location)
            message_history.append({"role": "system", "content": f"Fetched Context: {showtimes}"})
        elif "buy_ticket" in response_message.content:
            function_call = response_message.content
            start = function_call.find('(') + 1
            end = function_call.find(')')

            # Extract the arguments substring
            arguments = function_call[start:end]
            theater, movie, showtime = [arg.strip() for arg in arguments.split(',')]
            response_message.content = buy_ticket(theater, movie, showtime)
            await response_message.update()
            break
        elif "get_reviews" in response_message.content:
            response_message.content = "Fuck this shit"
            await response_message.update()
        elif "pick_random_movie" in response_message.content:
            function_call = response_message.content
            start = function_call.find('(') + 1
            end = function_call.find(')')

            # Extract the arguments substring
            arguments = function_call[start:end]
            random_movie = random.choice(arguments.split(','))
            print(f"random_movie:{random_movie}")
            message_history.append({"role": "system", "content": f"Random movie picked is: {random_movie}"})

        response_message = await generate_response(client, message_history, gen_kwargs)

    # Update message history as normal if there is no function in the latest response from the chatbot
    message_history.append({"role": "assistant", "content": response_message.content})

    # message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)

if __name__ == "__main__":
    cl.main()
