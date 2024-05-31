from data_manager import DataManager
from constants import MIN_NUMBER_SAMPLES
from model_manager import ModelManager, Classifiers


class EvaluationLogic:
    """
    EvaluationLogic class to manage the application's logic.

    This class handles user interaction for model selection and executes the model evaluation process.

    Attributes:
        menu_text (str): The text to display as the menu.
    """

    def __init__(self, menu_text):
        """
        Initializes the EvaluationLogic.

        Args:
            menu_text (str): The text to display as the menu.
        """
        self.menu_text = menu_text

    def get_model_chosen(self):
        """
        Displays the menu for the user to choose a model.

        This method repeatedly asks the user to enter an option until a valid
        choice (between 1 and 4) is made.

        Returns:
            int: The chosen model option.
        """
        option = 1
        option_entered = False
        print(self.menu_text)

        while not option_entered:
            try:
                # Prompt the user to enter an option
                option = int(input("Enter option (1-4): "))

                # Validate the entered option
                if 0 < option <= 4:
                    option_entered = True
                else:
                    print("Invalid choice. Please enter a number between 1 and 4.")
            except ValueError:
                print("Invalid input. Please try again.")

        return option

    @staticmethod
    def get_number_samples():
        """
        Prompts the user to enter the number of samples for model evaluation.

        This method repeatedly asks the user to enter a valid number of samples
        until a correct input is provided.

        Returns:
            int: The number of samples entered by the user.
        """
        num_samples = 0
        is_input_correct = False

        while not is_input_correct:
            try:
                # Prompt the user to enter the number of samples
                num_samples = int(input("Enter number of samples to use in the model's evaluation: "))

                # Validate the number of samples
                if num_samples < MIN_NUMBER_SAMPLES:
                    print("Invalid input. Please enter a number higher than 1.")
                else:
                    is_input_correct = True
            except ValueError:
                print("Invalid input. Please enter a number.")

        return num_samples

    @staticmethod
    def execute_model_evaluation(clf_type):
        """
        This method loads the data and the selected classifier then it executes the model evaluation process.

        Args:
            clf_type (Classifiers): The classifier type to be evaluated.
        """
        num_samples = EvaluationLogic.get_number_samples()
        data_manager = DataManager(num_samples)
        model_manager = ModelManager(clf_type, data_manager)

        # Load the model
        model_manager.load_model()

        # Display the model's performance
        model_manager.show_model_performance()

    def start_application_logic(self):
        """
        Starts the application logic for model evaluation.

        This method displays the menu, handles user choices, and executes
        the appropriate model evaluation based on the user's input.
        """
        show_menu, exit_menu = True, False

        classifier_dict = {classifier.value: classifier for classifier in Classifiers}

        while not exit_menu:
            # Get the chosen model option from the user
            option = self.get_model_chosen()

            # Exit the menu loop if the user has chosen the 4th option
            if option == 4:
                exit_menu = True
            else:
                # Execute model evaluation for the chosen classifier type
                clf_type = classifier_dict.get(option)
                self.execute_model_evaluation(clf_type)
