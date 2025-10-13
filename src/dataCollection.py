import os

def process_fruit_folder(main_folder, fruit_name, destination_folder):
    count = 0
    # List all items in the main folder
    items = os.listdir(main_folder)

    # Filter out only the directories whose names start with the specified fruit name
    fruit_directories = [item for item in items if os.path.isdir(os.path.join(main_folder, item)) and item.startswith(fruit_name)]

    # Iterate through each fruit directory and list its files
    for fruit_directory in fruit_directories:
        fruit_folder_path = os.path.join(main_folder, fruit_directory)
        
        # List all files in the fruit directory
        files_in_fruit_folder = [file for file in os.listdir(fruit_folder_path) if os.path.isfile(os.path.join(fruit_folder_path, file))]

        # Print or perform any operation on each file
        for i, file_name in enumerate(files_in_fruit_folder):
            _, ext = os.path.splitext(file_name)
            new_name = f"{i + 1}_{fruit_name.lower()}{ext}"

            old_path = os.path.join(fruit_folder_path, file_name)
            new_path = os.path.join(destination_folder, new_name)

            # Rename the file
            os.rename(old_path, new_path)
            


if __name__ == "__main__":
    main_folder_path = "/Users/meganpitts/Desktop/Intro to AI/Final Project GitHub/Fruit-Images-Dataset/Training"
    destination_folder_path = "/Users/meganpitts/Desktop/Intro to AI/Final Project GitHub/B351-final-project/data/train/watermelon"
    target_fruit_name = "Watermelon"  # Change this to the desired fruit name
    
    process_fruit_folder(main_folder_path, target_fruit_name, destination_folder_path)

