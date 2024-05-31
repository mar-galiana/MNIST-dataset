from constants import MENU_TEXT
from evaluation_logic import EvaluationLogic


def main():
    """
    Main function to initialize and start the application logic. This function creates an instance of the
    EvaluationLogic class
    """

    evaluation_logic = EvaluationLogic(MENU_TEXT)
    evaluation_logic.start_application_logic()


if __name__ == '__main__':
    # Entry point of the script
    main()
