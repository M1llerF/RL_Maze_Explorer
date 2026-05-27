# Load a pickle file and display its contents in a readable format.
# Used for debugging and testing purposes.

import pickle

class PKLReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self.loadData()

    def loadData(self):
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

    def formatData(self, data=None, indent=0):
        """Format the loaded data for pretty printing."""
        if data is None:
            data = self.data

        if isinstance(data, dict):
            formattedData = ""
            for key, value in data.items():
                formattedData += " " * indent + f"{key}:\n"
                formattedData += self.formatData(value, indent + 2)
            return formattedData
        elif isinstance(data, list):
            formattedData = ""
            for item in data:
                formattedData += " " * indent + "- "
                formattedData += self.formatData(item, indent + 2).strip() + "\n"
            return formattedData
        elif isinstance(data, tuple):
            formattedData = "("
            formattedData += ", ".join(self.formatData(item, 0).strip() for item in data)
            formattedData += ")\n"
            return formattedData
        else:
            return " " * indent + str(data) + "\n"

    def displayData(self):
        """Print the formatted data."""
        if self.data is not None:
            print(self.formatData())
        else:
            print("No data to display.")

if __name__ == "__main__":
    # Adjust path to a profile.pkl you want to inspect
    reader = PKLReader('profiles/QLearningBot/profile.pkl')
    reader.displayData()
