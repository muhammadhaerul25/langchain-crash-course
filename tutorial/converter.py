import pandas as pd

class Converter:
    @staticmethod
    def csv_to_json(csv_path, json_path, orient="records", indent=4):
        """
        Convert CSV file to JSON file.

        :param csv_path: Path to input CSV file
        :param json_path: Path to output JSON file
        :param orient: JSON format (default: 'records')
        :param indent: Indentation level for pretty JSON (default: 4)
        """
        df = pd.read_csv(csv_path)
        df.to_json(json_path, orient=orient, indent=indent, force_ascii=False)

        print(f"Successfully converted '{csv_path}' to '{json_path}'")


csv_path = "documents/transaksi.csv"
json_path = "documents/transaksi.json"
Converter.csv_to_json(csv_path, json_path)