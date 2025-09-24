# Load a pickle file and display its contents in a readable format.
# Used for debugging and testing purposes.

import pickle

class PKLReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self.load_data()

    def load_data(self):
        """Load data from a pickle file."""
        try:
            with open(self.filepath, 'rb') as file:
                data = pickle.load(file)
            return data
        except FileNotFoundError:
            print(f"File not found: {self.filepath}")
            return None
        except Exception as e:
            print(f"Error loading file: {e}")
            return None

    def format_data(self, data=None, indent=0):
        """Format the loaded data for pretty printing."""
        if data is None:
            data = self.data

        if isinstance(data, dict):
            formatted_data = ""
            for key, value in data.items():
                formatted_data += " " * indent + f"{key}:\n"
                formatted_data += self.format_data(value, indent + 2)
            return formatted_data
        elif isinstance(data, list):
            formatted_data = ""
            for item in data:
                formatted_data += " " * indent + "- "
                formatted_data += self.format_data(item, indent + 2).strip() + "\n"
            return formatted_data
        elif isinstance(data, tuple):
            formatted_data = "("
            formatted_data += ", ".join(self.format_data(item, 0).strip() for item in data)
            formatted_data += ")\n"
            return formatted_data
        else:
            return " " * indent + str(data) + "\n"

    def display_data(self):
        """Print the formatted data."""
        if self.data is not None:
            print(self.format_data())
        else:
            print("No data to display.")

if __name__ == "__main__":
    # Adjust path to a profile.pkl you want to inspect
    reader = PKLReader('profiles/QLearningBot/profile.pkl')
    reader.display_data()
