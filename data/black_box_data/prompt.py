base_prompt = '''
You will be given a multiple-choice question with a set of answer options. Your task is to allocate tokens between these choices based on your belief or evidence for each option. You have a total of 100 tokens to distribute. The more tokens you allocate to an option, the more confident you are in that option.

You must follow these rules:

The tokens assigned to all options cannot add up to more than 100.
The tokens must be whole numbers.
Provide your token allocation in the following format:
{
  "A": [number of tokens],
  "B": [number of tokens],
  "C": [number of tokens],
  "D": [number of tokens],
  "E": [number of tokens],
  "F": [number of tokens]
}
'''