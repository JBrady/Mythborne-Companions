import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from datetime import datetime # Import datetime

# --- Generate a unique log filename ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"crew_run_{timestamp}.log"
# --- End of filename generation ---

load_dotenv() # Load environment variables from .env file (ensure OPENAI_API_KEY is set)

# Configure the OpenAI LLM (using gpt-4.5-preview for initial testing, can upgrade later)
llm = ChatOpenAI(model="gpt-4.5-preview",
                 openai_api_key=os.getenv("OPENAI_API_KEY"),
                 temperature=0.7)

# --- Agent Definitions for "Mythborne Companions" ---

# 1. Project Manager Agent
project_manager = Agent(
    role='Lead Project Manager (Mobile Games)',
    goal='Oversee the Mythborne Companions project, coordinate tasks, manage priorities & risks, ensure alignment with vision & Pi Network constraints.', # Updated Name
    backstory=(
        'An experienced project manager from the mobile gaming industry, skilled in Agile methodologies '
        'and coordinating multi-disciplinary teams. Highly organized, communicative, and focused on '
        'efficient execution and risk management. Understands the unique challenges of developing '
        'within emerging ecosystems like Pi Network. Manages project docs and workflow for Mythborne Companions.' # Updated Name
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# 2. Game Designer Agent
game_designer = Agent(
    role='Senior Game Designer (Collectible Creature & Social Sims)',
    goal='Refine and detail gameplay mechanics, systems, player experience, and compliant Pi integration for Mythborne Companions. Maintain the GDD.', # Updated Name
    backstory=(
        'A creative F2P mobile designer passionate about collectible creatures and social simulation loops. '
        'Experienced in systems balancing, UX for casual audiences, and ethical Pi Network crypto integration for Mythborne Companions. ' # Updated Name
        'Updates the GDD based on feedback and specs features for other team members.'
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# 3. Lead Programmer Agent
lead_programmer = Agent(
    role='Lead Software Engineer (Mobile Web Games - HTML5/JS)',
    goal='Define technical architecture, assess feature feasibility, plan Pi SDK integration, choose web technologies, and oversee technical implementation for Mythborne Companions.', # Updated Name
    backstory=(
        'A seasoned engineer specializing in performant HTML5/JS games (e.g., Phaser, PixiJS). Expert in SDK integration, '
        'clean code, scalability, and secure crypto interactions, especially regarding Mythborne Companions and Pi Network. ' # Updated Name
        'Plans tech stack and generates core system structures based on GDD specs.'
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# 4. Lead Artist Agent
lead_artist = Agent(
    role='Lead Game Artist (Mobile Web)',
    goal='Define visual style, create concept art, establish art pipeline, ensure cohesive and optimized visuals for Mythborne Companions on Pi Browser.', # Updated Name
    backstory=(
        'A versatile mobile game artist lead skilled in character/environment/UI design and animation. Creates appealing, '
        'performant web visuals suitable for the Pi Browser, fitting the "Mythborne Companions" theme. ' # Updated Name
        'Can adapt to incorporating Pi branding subtly. Specs assets and collaborates with UI/UX.'
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# 5. UI/UX Designer Agent
ui_ux_designer = Agent(
    role='UI/UX Designer (Mobile Web)',
    goal='Design an intuitive, accessible, and visually appealing UI/UX flow for Mythborne Companions, optimized for Pi Browser.', # Updated Name
    backstory=(
        'A user-centric mobile web UI/UX specialist. Expertise in information architecture, interaction design, '
        'usability, and creating clean interfaces within web platform constraints, particularly for Mythborne Companions.' # Updated Name
        ' Collaborates closely with design and art.'
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True
)


# --- Define Tasks Based on Project Manager's Plan ---

# Task 1 (PM Output Item 1)
task_define_core_loop = Task(
    description=(
        'Based on the initial concept for Mythborne Companions, clearly define the core gameplay loop. '
        'Detail the primary player objectives, core actions (e.g., exploring, collecting, crafting, battling/mini-games), '
        'progression mechanics (e.g., leveling up companions, unlocking features), and how players interact '
        'meaningfully with their Mythborne companions.'
    ),
    expected_output=(
        'A document section detailing the core gameplay loop for Mythborne Companions, covering player objectives, '
        'core actions, progression systems, and key companion interactions.'
    ),
    agent=game_designer # Assigned by PM
)

# Task 2 (PM Output Item 2)
task_define_pi_integration = Task(
    description=(
        'Specify the exact Pi token integrations for Mythborne Companions. Detail potential ways players might earn Pi '
        '(if allowed by Pi Network policies, specify this constraint) and clear ways they can spend Pi within the game '
        '(e.g., for cosmetic items, speeding up timers, special crafting recipes, companion slots). '
        'Crucially, ensure these mechanics strictly align with current Pi Network technical guidelines and platform policies.'
    ),
    expected_output=(
        'A specification document detailing proposed Pi earning (if applicable) and spending mechanics. '
        'Include example costs/rewards in Pi. Explicitly state how alignment with Pi Network policies will be maintained.'
    ),
    agent=game_designer # Assigned by PM
)

# Task 3 (PM Output Item 3)
task_check_feasibility = Task(
    description=(
        'Evaluate the technical feasibility of integrating the proposed Pi Network functionalities '
        '(wallet interaction, transaction handling for Pi spending/earning actions defined in the previous task) '
        'into Mythborne Companions using the Pi SDK/APIs. Consider potential blockchain constraints (e.g., transaction times, fees), '
        'API limitations, security best practices, and the impact on the likely mobile web technology stack (HTML5/JS). '
        'Outline potential challenges and recommended technical approaches or solutions.'
    ),
    expected_output=(
        'A technical feasibility report covering Pi SDK/API integration for the defined mechanics. '
        'Highlight potential challenges (latency, security, policy compliance), constraints, '
        'and recommended technical solutions or necessary workarounds.'
    ),
    agent=lead_programmer # Assigned by PM
)

# Task 4 (PM Output Item 4)
task_define_art_style = Task(
    description=(
        'Establish initial visual art style guidelines for Mythborne Companions. Provide text descriptions '
        'and optionally list key visual references or inspirations (e.g., specific games, art styles) that define the '
        'desired look and feel for the Mythborne companions (creatures), game environments, and UI elements. '
        'The style should be appealing, consistent with the "Mythborne" theme, suitable for the Pi Network community, '
        'and technically feasible for performant mobile web rendering.'
    ),
    expected_output=(
        'A brief document outlining the proposed art style direction (e.g., "Stylized 2D Cartoon", "Semi-Realistic Fantasy"). '
        'Include descriptions of the desired look for companions, environments, and UI. List 2-3 visual reference points if possible.'
    ),
    agent=lead_artist # Assigned by PM
)

# Task 5 (PM Output Item 5)
task_map_core_ux_flows = Task(
    description=(
        'Map out the initial high-level UI/UX user flows for 2-3 core features of Mythborne Companions. '
        'Focus on: (1) Companion Management (viewing stats, feeding/interacting, initiating evolution if applicable) '
        'and (2) a key flow involving Pi token transactions (e.g., purchasing an item from a shop with Pi). '
        'Describe the steps the user takes and the screens they interact with, prioritizing intuitive navigation '
        'and a smooth experience on mobile web.'
    ),
    expected_output=(
        'A document describing the step-by-step user flow for Companion Management and a Pi Transaction screen. '
        'Use text descriptions for each step and screen involved. Focus on clarity and ease of use.'
    ),
    agent=ui_ux_designer # Assigned by PM
)

# --- Create the Crew with the New Task List ---
mythborne_crew = Crew(
    agents=[project_manager, game_designer, lead_programmer, lead_artist, ui_ux_designer],
    tasks=[
        # List the new tasks in a logical sequence
        task_define_core_loop,
        task_define_pi_integration,
        task_check_feasibility,
        task_define_art_style,
        task_map_core_ux_flows
    ],
    process=Process.sequential, # Run these tasks one after the other
    verbose=True,
    output_log_file='crew_run_internal.log' # Saves to a specific file
)

# --- Kick Off the Crew's Work ---
print("###################################################")
print("## Starting Mythborne Companions Crew Run (Phase 1 Tasks)...")
print("## Logging verbose output to: {log_filename}") # Indicate log file name
print("###################################################")
result = mythborne_crew.kickoff()

# --- Print the Final Result ---
print("\n\n###################################################")
print("## Crew Run Completed!")
print("## Full verbose log saved to: {log_filename}")
print("###################################################")
print("\nFinal Output (from UI/UX Designer - Last Task):\n") # Output is from the last task in sequential mode
print(result)