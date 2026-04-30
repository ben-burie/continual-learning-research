import sys

from scripts.generate_data import main as generate_data
from scripts.train import main as train_base
from scripts.continual_train import main as train_continual
from scripts.ewc_continual_train import main as train_ewc
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
        case 5: # Evaluation
            evaluate()
        case 6: # Exit
            sys.exit()
        case _:
            raise Exception(f"{selection} is an invalid selection.")

def main():
    print("Welcome to COMPSCI 432 Final Project - EWC Implementation for a Voice Command System")
    print("Options:")
    print("1. Generate Training Data")
    print("2. Train From Scratch")
    print("3. Add New Command - Continual Train (NO EWC)")
    print("4. Add New Command - Continual Train Using EWC Stratgey")
    print("5. Evaluate A Model")
    print("6. Exit")

    selection = int(input("\nSelection: "))
    call_selection(selection)

if __name__ == "__main__":
    main()