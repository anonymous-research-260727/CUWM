from eval.agent_wm_gen import Agent, setup_logger, parse_args_and_logger
from datetime import timezone, timedelta
import argparse
import os
from datetime import datetime
from tqdm import tqdm

if __name__ == "__main__":
    args, logger = parse_args_and_logger()
    # args.data_folder = "new_prompt/test_prompt"
    agent = Agent(args, logger)
    
    folders = []
    for root, dirs, files in os.walk(os.path.join(agent.ROOT, args.data_folder)):
        if "prev.png" in files:
            folders.append(root)
    
    for prompt_file in [
        "prompt_action_only.txt",
        "prompt_action_with_description.txt",
        "prompt_nl_gt.txt",
        "prompt_nl_pred_gpt-5.2-chat-20251211.txt",
        "prompt_nl_pred_checkpoint-450-merged.txt",
        "prompt_nl_pred_base.txt",
    ]:
        for folder in tqdm(folders):
            print(f"Processing: {folder}, prompt_file: {prompt_file}, model_key: {args.model_key}")
            prompt_name = prompt_file.replace(".txt", "")
            output_name = f"generated_next/next_{args.model_key}_{prompt_name}.png"
        
            os.makedirs(os.path.join(folder, 'generated_next'), exist_ok=True)
            
            if os.path.exists(os.path.join(folder, output_name)):
                logger.info(f"Output already exists for {folder}, skipping.")
                continue
            
            agent._generate_next_image(
                edit_path = os.path.join(folder, 'prev.png'),
                prompt_path = os.path.join(folder, prompt_file),
                output_path = os.path.join(folder, output_name)
            )    
    
    
    