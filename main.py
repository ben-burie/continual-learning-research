import sys

from scripts.generate_data import main as generate_data
from scripts.train import main as train_base
from scripts.continual_train import main as train_continual
from scripts.ewc_continual_train import main as train_ewc
from scripts.distillation_continual_train import main as train_distillation
from scripts.evaluate import main as evaluate

def call_selection(selection):
    match selection:
        case 1: # Generate training data
            generate_data()
        case 2: # Train from scratch
            train_base()
        case 3: # Continual train - no ewc
            train_continual()
        case 4: # Continual train - ewc
            train_ewc()
        case 5: # Continual train - distillation + exemplar replay + bias correction
            train_distillation()
        case 6: # Evaluation
            evaluate()
        case 7: # Exit
            sys.exit()
        case _:
            raise Exception(f"{selection} is an invalid selection.")

def main():
    print("Welcome to COMPSCI 432 Final Project - EWC Implementation for a Voice Command System")
    print("Options:")
    print("1. Generate Training Data")
    print("2. Train From Scratch")
    print("3. Add New Command - Continual Train (NO EWC)")
    print("4. Add New Command - Continual Train Using EWC Strategy")
    print("5. Add New Command - Continual Train Using Distillation + BiC Strategy")
    print("6. Evaluate A Model")
    print("7. Exit")

    while True:
        try:
            selection = int(input("\nSelection: "))
            break
        except ValueError:
            print("Please enter a number.")
    try:
        call_selection(selection)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()