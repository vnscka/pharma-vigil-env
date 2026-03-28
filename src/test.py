from ade_environment import ADEEnvironment
from models import Action

# Initialize the environment
env = ADEEnvironment()

# Loop over tasks
for task_id in [1, 2, 3]:
    # Reset environment for the current task
    obs = env.reset(task_id=task_id, seed=42)
    print(f"Task {task_id} | initial observation: {obs}")

    # Corrected Action creation (use required field names)
    action = Action(
        severity='serious',
        causality=0.8,
        escalate=True,
        rec_action='request_followup',  # corrected field name
        is_signal=(task_id == 3)        # True for task 3, False for others
    )

    # Step the environment
    next_obs, reward, done, info = env.step(action)

    # Print results
    print(
        f"Task {task_id} | reward={reward} | raw={info.get('raw_score', 'N/A')} | "
        f"penalties={info.get('penalties_applied', 'N/A')} | done={done}"
    )
    print(f"Next observation: {next_obs}\n")