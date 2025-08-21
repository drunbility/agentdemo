import asyncio
from agents import GuardrailFunctionOutput,Agent, Runner,InputGuardrail
from agents.exceptions import InputGuardrailTripwireTriggered
from dotenv import  load_dotenv
from pydantic import BaseModel

load_dotenv()

class HomeWorkOutput(BaseModel):
    is_homework: bool
    reasoning: str

guardrail_agent= Agent(
    model="litellm/gemini/gemini-2.5-flash",
    name="Guardrail Check",
    instructions="Check if the user is asking about homework.",
    output_type= HomeWorkOutput,
)

history_tutor_agent = Agent(
    model="litellm/gemini/gemini-2.5-flash",
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)

math_tutor_agent = Agent(
    model="litellm/gemini/gemini-2.5-flash",
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

async def homework_guardrail(ctx,agent,input_data):
    result = await Runner.run(guardrail_agent,input_data,context=ctx.context)
    final_output = result.final_output_as(HomeWorkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )

triage_agent = Agent(
    model="litellm/gemini/gemini-2.5-flash",
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)



async def main():

    try:
        result = await Runner.run(triage_agent, "who was the first president of the united states?")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail blocked this input:",e)
    # try:
    #     result = await Runner.run(triage_agent, "生命的意义是啥")
    #     print(result.final_output)
    # except InputGuardrailTripwireTriggered as e:
    #     print("Guardrail blocked this input:",e)

if __name__ == "__main__":
    asyncio.run(main())

