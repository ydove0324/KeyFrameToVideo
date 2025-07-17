import wandb
import sys

def view_wandb_run():
    run_path = "./logs/wandb/offline-run-20250715_131519-k1wn84jb"
    print(f"Loading wandb run from: {run_path}")
    api = wandb.Api()
    run = api.run(run_path)
    history = run.history()
    return history

if __name__ == "__main__":
    history = view_wandb_run()
    print(history)