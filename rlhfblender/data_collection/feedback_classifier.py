import os
import json
import re
from groq import Groq ,AsyncGroq
from typing import Dict, Any

api_key = "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"
os.environ["GROQ_API_KEY"] = api_key
# Initialize the Groq client with the API key
client = AsyncGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Function to classify feedback using the Groq client
async def classify_feedback(text_input, all_feedback):
    # Construct the detailed prompt for the model
    prompt = f"""
    Analyze the following feedback provided for an agent's performance. This agent operates within the **Google Research Football (GRF)** environment. The agent's task is to perform football actions in a simulated environment, interacting with both teammates and opponents. It can control multiple players on the field, respond to game scenarios, and learn through reinforcement learning techniques. The agent's performance is assessed based on how effectively it achieves goals like scoring, defending, passing, and maintaining possession.

    The GRF environment provides raw observations such as ball position, player positions, movement directions, and more, allowing the agent to make informed decisions. It also provides a detailed action set that the agent can use to interact with the environment.

    Your task is to classify the feedback into one of the specific categories listed below. Each category is defined with clear characteristics to guide your classification. After identifying the category, assign a score from 1 to 10 based on the relevance and importance of the feedback. The score should reflect how critical this feedback is for improving or evaluating the agent's performance.

    Additionally, **extract and map the actions directly from the provided action set** (returning them as a list of integers). The feature selection should be derived from the observation component, and the feature importance should be a list of floats, where each float corresponds to a feature in the feature selection list.

    Additionally, provide goal preferences and corresponding action preferences:
    - goal_preferences: A list of dictionaries, where each dictionary contains information about a goal and its priority or preference level. The preference should be a int starting from 1 and then incrementing, with 1 being the highest preference.
    - action_preferences: A list of integers representing the priority of actions based on the goals. Use the action from action list we will get which is basically a list of integers corresponding to the mapped actions to select relevant actions to solve the goal and order them based on there priority meaning the actions that are more likely to achieve the goal should be ordered first.

    Additionally, analyze the following feedback for the current episode and compare it with other episodes. Provide a relative ranking of all episodes, based on the feedback. Assign a preference value between 0 and 1 for each episode, where higher values indicate higher preference.
    Current episode feedback: {text_input}
    All episodes feedback: {all_feedback}    
    

    ### Feedback Categories:

    1. Critique: Provide a detailed analysis of the agent's performance. Highlight both strengths and weaknesses with specific examples. This category involves a balanced evaluation that emphasizes constructive feedback.
       - Example: "The agent's ball control and passing accuracy were excellent, especially in tight spaces. However, its defensive positioning needs improvement, as it often left gaps in the backline."
       - Example: "The agent's dribbling skills were exceptional, allowing for impressive runs past defenders. However, its finishing ability needs improvement, as it missed several clear goal-scoring opportunities."
       - Example: "The agent's defensive tackles were strong and timely, disrupting the opponent's play effectively. However, its positional awareness was lacking at times, leading to unmarked opponents in dangerous areas."
       - Example: "The agent's passing range was outstanding, often delivering accurate long balls to switch the play. However, its decision-making in the final third needs refinement, as it sometimes chose the wrong option when in a scoring position."
       - Example: "The agent's teamwork was excellent, often making selfless runs to create space for others. However, its finishing under pressure could improve, as it struggled to convert chances in high-stakes situations."

    2. Suggestion: Offer specific recommendations aimed at improving the agent's behavior, strategy, or performance. The suggestions should be actionable and targeted at particular areas needing enhancement.
       - Example: "The agent should focus on improving its off-the-ball movement, particularly when making runs into the box."
       - Example: "The agent should work on its finishing technique during training, particularly focusing on shooting accuracy from various angles and distances."
       - Example: "The agent needs to enhance its defensive positioning by studying game footage to better understand when to step up and when to drop back to maintain team shape."
       - Example: "The agent should prioritize communication with teammates on the field, especially during set pieces, to ensure everyone is clear on their roles and responsibilities."
       - Example: "The agent could benefit from developing its ability to switch play quickly, utilizing longer passes to exploit spaces on the opposite side of the field."       

    3. Observation: Provide neutral descriptions of notable events, patterns, or behaviors observed during the agent's performance. This category should be free from subjective judgment and focus on what was noticed.
       - Example: "At 2:15 in the episode, the agent consistently used the offside trap strategy, causing the opposing team to commit multiple offside offenses."
       - Example: "During the first half, the agent completed 85% of its passes, demonstrating good distribution and maintaining possession for the team."
       - Example: "At 30:45 in the match, the agent made a crucial interception, preventing a potential goal-scoring opportunity for the opposing team."
       - Example: "The agent frequently dropped deep to receive the ball, particularly during buildup play, indicating a willingness to support the defense."
       - Example: "Throughout the match, the agent made several overlapping runs on the flank, creating space for teammates to exploit in attack."
       - Example: "During set pieces, the agent consistently positioned itself near the edge of the box, ready to react to any loose balls that might come its way."

    4. Comparison: Contrast the agent's current performance with previous episodes, other agents, or a known standard. Highlight areas of improvement or decline.
       - Example: "Compared to the previous episode, the agent's goalkeeping skills have significantly improved. It now makes better decisions on when to rush out versus staying on the goal line."
       - Example: "When compared to other wingers in the league, the agent's dribbling success rate is among the top 10, showcasing its effectiveness in one-on-one situations."
       - Example: "Compared to previous episodes, the agent's goal-scoring frequency has increased, netting three goals in the last four matches, while it scored only two in the first half of the season."
       - Example: "Unlike the previous match, where the agent completed only 50% of its aerial duels, it has now consistently won over 70% of them, demonstrating significant improvement in this area."
       - Example: "Relative to its last few appearances, the agent's involvement in key passes has decreased significantly, dropping from 4 key passes per game to just 1 in the latest match."       

    5. Mission: Describe the expected behavior or outcomes versus what actually occurred during the agent's performance. This should reflect discrepancies between anticipated and actual actions.
       - Example: "I expected the agent to prioritize ball possession, but it often opted for long, risky passes. This aggressive style led to more scoring opportunities than anticipated."
       - Example: "I anticipated the agent would focus on creating space for teammates with off-the-ball runs, but it frequently stood still, limiting the team's attacking options."
       - Example: "I expected the agent to take more shots from inside the box, but it hesitated to pull the trigger and instead passed the ball, resulting in fewer scoring chances than anticipated."
       - Example: "Should have created multiple scoring opportunities for teammates, but only managed one assist during the match."
       - Example: "Should have completed more than 80% of passes, but ended the game with only 65% accuracy, impacting the team's overall possession."
       - Example: "Expected to lead the defense effectively, but the agent was often caught out of position, allowing the opposing team to exploit gaps."

    6. Prioritization: Rank the aspects of the agent's performance that need the most attention or improvement. This category is about identifying critical areas for development.
       - Example: "1. Improve defensive positioning. 2. Enhance decision-making under pressure."
       - Example: "Improve finishing accuracy. 2. Enhance off-the-ball movement. 3. Increase defensive contributions."
       - Example: "1. Strengthen passing accuracy in the final third. 2. Develop better communication with teammates. 3. Focus on maintaining possession under pressure. "
       - Example: "1. Enhance tackling and interception skills. 2. Improve positioning during set pieces. 3. Increase involvement in build-up play. "
       - Example: "1. Boost aerial ability in both attacking and defensive situations. 2. Refine decision-making when to shoot versus pass. 3. Work on stamina and overall fitness levels."

    7. Miscellaneous: Use this category for feedback that does not fit into the above categories but is still relevant to the agent's performance or development.

    ### Actions Mapping (returned as a list of integers):
    The following is the action set the agent uses in the GRF environment. **Use the feedback to select the relevant actions from this list and return them as a list of integers:**
    - 0: Idle (no action)
    - 1: Run to the left
    - 2: Run to the top-left
    - 3: Run to the top
    - 4: Run to the top-right
    - 5: Run to the right
    - 6: Run to the bottom-right
    - 7: Run to the bottom
    - 8: Run to the bottom-left
    - 9: Long pass
    - 10: High pass
    - 11: Short pass
    - 12: Shot
    - 13: Sprint
    - 14: Release direction
    - 15: Release sprint
    - 16: Sliding
    - 17: Dribble
    - 18: Release dribble
    - 19: Built-in AI action

    ### Observations and Feature Selection:
    Derive the feature selection from the following observations in the GRF environment:
    - Ball information (position, direction, rotation, ownership)
    - Left team (positions, direction, tiredness, yellow cards, activity, roles)
    - Right team (same as left team)
    - Controlled player (active player, sticky actions, designated player)
    - Match state (score, steps left, game mode)
    - Screen (frame with RGB pixel vectors if rendering is enabled)

    The `feature_importance` should be a list of floats where each float corresponds to the feature in the `feature_selection`.

    ### Instructions:

    - Classify the feedback into one of the categories above.
    - Extract the following information from the feedback:
      {{
        "category": "classification type",
        "score": integer from 1 to 10,
        "feedback": "summary of the feedback",
        "action": **list of integers corresponding to the mapped actions**,
        "goal": it is a dictionary Ex:- ["goal_type": "description of the goal", "description": "detailed goal description if needed"],
        "feature_selection": observations component based on the provided details,
        "feature_importance": list of floats where each float explains the corresponding feature in feature_selection,
        "action_preferences": list of integers,
        "goal_preferences": list of dictionaries
        "preferences": [list of float values representing relative ranking of all episodes]
        "episode_ids": [list of corresponding episode IDs in the same order as preferences]
      }}

    -Ensure that the "preferences" field contains a list of float values, and the "episode_ids" field contains the corresponding episode IDs.

    - Example structure of the dictionary to return:
    {{
      "category": "Critique",
      "score": 8,
      "feedback": "The agent's positioning needs improvement, but ball control was excellent.",
      "action": [13, 12],
      "goal": ["goal_type": "Improve attacking positioning", description": "The agent needs to focus on better marking and positioning during attacking."],
      "feature_selection": ["ball_position", "left_team_direction", "right_team_roles"],
      "feature_importance": [0.8, 0.7, 0.9],
      "action_preferences": [12,13],
      "goal_preferences": [{{"goal": "scoring goal", "priority": 1}}, {{"goal": "improved passing", "priority": 2}}],
      "preferences": [0.8, 0.6, 0.4]
      "episode_ids": [1, 2, 3]
    }}

    - No additional explanation or text should be provided, just the JSON-formatted response.
    - Dont provide any explanation or text your answer should be only JSON format.
    - If the feedback is not related to the agent's performance or development, return the following JSON:
    {{
      "category": "Miscellaneous",
      "score": 0,
      "feedback": "The feedback is not related to the agent's performance or development.",
      "action": [],
      "goal": [],
      "feature_selection": [],
      "feature_importance": [],
      "action_preferences": [],
      "goal_preferences": [],
      "preferences": [],
      "episode_ids": []
    }}

    ### Feedback to Analyze:
    {text_input}
    """

    # Send the prompt to the Groq model
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-70b-8192",  # Use the appropriate model
    )

    # Retrieve the response from the model
    return chat_completion.choices[0].message.content



def clean_json_string(json_str: str) -> str:
    """Clean the JSON string by replacing single quotes and removing extraneous characters."""
    cleaned_json = json_str.replace("'", '"')
    cleaned_json = re.sub(r'(?<=[}\]"''])\s*(?:,\s*)?(?=[{\["''])', ',', cleaned_json)
    cleaned_json = re.sub(r',\s*}', '}', cleaned_json)
    cleaned_json = re.sub(r',\s*]', ']', cleaned_json)
    cleaned_json = re.sub(r'#.*', '', cleaned_json)
    cleaned_json = cleaned_json.strip()
    return cleaned_json

async def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'

    match = re.search(json_pattern, response_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        cleaned_json_str = clean_json_string(json_str)
        try:
            return json.loads(cleaned_json_str)
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed. Error: {e}")
    else:
        print("No valid JSON found in the response.")    
    
    return await create_default_response()

async def create_default_response() -> Dict[str, Any]:
    return {
  "category": "Miscellaneous",
  "score": 0,
  "feedback": "The feedback is not related to the agent's performance or development.",
  "action": [],
  "goal": [],
  "feature_selection": [],
  "feature_importance": [],
  "action_preferences": [],
  "goal_preferences": [],
  "preferences": [],
  "episode_ids": []
    }  
    
    
