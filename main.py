from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents.exceptions import InputGuardrailTripwireTriggered
from pydantic import BaseModel
import asyncio
from dotenv import load_dotenv

load_dotenv()

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

guardrail_agent = Agent(
    model="litellm/gemini/gemini-2.5-flash",
    name="GuardrailCheck",
    instructions="检查用户是否在询问作业。",
    output_type=HomeworkOutput,
)

math_tutor_agent = Agent(
    model="litellm/gemini/gemini-2.5-flash",
    name="MathTutor",
    handoff_description="数学问题专家",
    instructions="你提供帮助数学问题。解释你的推理每一步，并包括例子",
)

history_tutor_agent = Agent(
    model="litellm/gemini/gemini-2.5-flash",
    name="HistoryTutor",
    handoff_description="历史问题专家",
    instructions="你提供帮助历史问题。解释重要事件和上下文清楚。",
)


async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=final_output.is_homework,
    )

triage_agent = Agent(
    model="litellm/gemini/gemini-2.5-flash",
    name="Triage",
    instructions="你根据用户的问题确定使用哪个代理",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)

async def main():
    # Example 1: History question
    # try:
    #     result = await Runner.run(triage_agent, "法国首都是哪里")
    #     print(result.final_output)
    # except InputGuardrailTripwireTriggered as e:
    #     print("Guardrail blocked this input:", e)

    # Example 2: General/philosophical question
    try:
        result = await Runner.run(triage_agent, "生命的意义是啥?")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail blocked this input:", e)

if __name__ == "__main__":
    asyncio.run(main())