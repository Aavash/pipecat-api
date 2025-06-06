import os
import re


def rename_files(folder_path, pattern, new_name_format, preview=False):
    """
    Renames files in the specified folder based on a regex pattern and a new name format.

    Args:
        folder_path (str): Path to the folder containing the files.
        pattern (str): Regex pattern with capture groups to match filenames.
        new_name_format (str): New filename format using Python-style formatting, e.g., 'lesson_{0}_{1}.md'.
        preview (bool): If True, only preview the changes without renaming.

    Example:
        pattern = r"lesson_(\d+)_Suffix_value(_?\d*)\.md"
        new_name_format = "lesson_{0}_adv_python.md"
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return

    regex = re.compile(pattern, re.IGNORECASE)
    renamed_count = 0

    for filename in os.listdir(folder_path):
        match = regex.match(filename)
        if match:
            groups = match.groups()
            try:
                new_name = new_name_format.format(*groups)
            except IndexError:
                print(f"Error: Not enough groups in pattern for '{filename}'")
                continue

            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)

            if preview:
                print(f"[Preview] {filename} → {new_name}")
            else:
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} → {new_name}")
                    renamed_count += 1
                except Exception as e:
                    print(f"Error renaming {filename}: {str(e)}")

    if not preview:
        print(f"\nRenaming complete! {renamed_count} files were renamed.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "readme_output")

    # Customize this pattern and format as needed
    rename_files(
        folder_path=output_folder,
        pattern=r"lesson_(\d+)_Suffix_value(_?\d*)\.md",
        new_name_format="lesson_{0}_adv_python.md",
    )
