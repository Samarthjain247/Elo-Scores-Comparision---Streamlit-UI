import os
import cv2
import streamlit as st
import pandas as pd
from random import shuffle, choice
import csv

# Define checkpoints
checkpoints = ['ckpt1', 'ckpt2', 'ckpt3', 'ckpt4', 'ckpt5', 'ckpt6', 'ckpt7', 'ckpt8']

# Directory path where all cases folders are located
base_folder = '/Users/samarthjain/Documents/ALL_Cases'

# Directory path to save CSV file
csv_folder = '/Users/samarthjain/Documents/csv_checkpoints'

# Initialize ELO scores
elo_scores = {checkpoint: 1500 for checkpoint in checkpoints}

# Function to generate all combinations of checkpoint pairs and images
def generate_combinations(category_folder):
    all_combinations = []
    num_checkpoints = len(checkpoints)
    
    # Get a list of all images in the first checkpoint directory
    images = os.listdir(os.path.join(base_folder, category_folder, checkpoints[0]))
    
    # Generate all unique pairs of checkpoints and combine with images
    for image_name in images:
        for i in range(num_checkpoints):
            for j in range(i + 1, num_checkpoints):
                all_combinations.append((checkpoints[i], checkpoints[j], image_name))
    
    # Shuffle the list of combinations randomly
    shuffle(all_combinations)
    return all_combinations

# Function to read choices from CSV file
def read_choices_from_csv(csv_folder):
    choices = []
    csv_file = os.path.join(csv_folder, 'image_comparison_results.csv')
    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                choice = {
                    'CheckpointA': row.get('CheckpointA', ''),
                    'CheckpointB': row.get('CheckpointB', ''),
                    'Image_Name': row.get('Image_Name', ''),
                    'Preferred_Checkpoint': row.get('Preferred_Checkpoint', ''),
                    'Serial_Number': row.get('Serial_Number', ''),
                    'Category': row.get('Category', '')
                }
                choices.append(choice)
    return choices

# Function to read and resize images
def read_and_resize_image(img_path, height=1000):
    if not os.path.exists(img_path):
        return None
    
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    h, w, _ = img.shape
    aspect_ratio = w / h
    new_w = int(height * aspect_ratio)
    img = cv2.resize(img, (new_w, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Function to append a single choice to CSV file
def save_to_csv(choice, csv_folder):
    csv_file = os.path.join(csv_folder, 'image_comparison_results.csv')
    
    if not os.path.exists(csv_file):
        # Create a new file with headers if it does not exist
        with open(csv_file, mode='w', newline='') as file:
            fieldnames = ['Serial_Number', 'CheckpointA', 'CheckpointB', 'Image_Name', 'Preferred_Checkpoint', 'Category']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
    
    # Determine the starting serial number
    serial_number = 1
    if os.path.isfile(csv_file):
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                serial_number = int(row.get('Serial_Number', '0')) + 1
    
    # Add serial number and write the choice to the CSV
    choice['Serial_Number'] = serial_number
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Serial_Number', 'CheckpointA', 'CheckpointB', 'Image_Name', 'Preferred_Checkpoint', 'Category'])
        writer.writerow(choice)

# Function to clear CSV file data
def clear_csv_data(csv_folder):
    csv_file = os.path.join(csv_folder, 'image_comparison_results.csv')
    try:
        os.remove(csv_file)
        st.session_state.choices = []
        st.session_state.current_pair_index = 0
        st.success("CSV file data cleared successfully.")
    except OSError as e:
        st.error(f"Error clearing CSV file: {e}")

# Function to update ELO scores
def update_elo(winner, loser, K=32):
    if winner not in elo_scores:
        elo_scores[winner] = 1500
    if loser not in elo_scores:
        elo_scores[loser] = 1500
    
    # Calculate expected scores
    expected_winner = 1 / (1 + 10 ** ((elo_scores[loser] - elo_scores[winner]) / 400))
    expected_loser = 1 / (1 + 10 ** ((elo_scores[winner] - elo_scores[loser]) / 400))
    
    # Update ELO scores
    elo_scores[winner] += K * (1 - expected_winner)
    elo_scores[loser] += K * (0 - expected_loser)

# Function to process comparison results from CSV and update ELO scores
def process_comparison_results(csv_folder, category=None):
    temp_elo_scores = {checkpoint: 1500 for checkpoint in checkpoints}
    file_path = os.path.join(csv_folder, 'image_comparison_results.csv')
    if os.path.exists(file_path):
        comparison_results = pd.read_csv(file_path)
        if category:
            # Filter rows based on the processed image category name
            category_filter = f"processed_images_{category}"
            comparison_results = comparison_results[comparison_results['Category'].str.contains(category_filter, case=False, na=False)]
            st.write(f"Filtered results for category {category_filter}:")
            st.write(comparison_results)  # Debug: Print filtered results
        for index, row in comparison_results.iterrows():
            checkpoint_a = row['CheckpointA']
            checkpoint_b = row['CheckpointB']
            preferred_checkpoint = row['Preferred_Checkpoint']
            
            if preferred_checkpoint == checkpoint_a:
                temp_elo_scores = update_elo_score(checkpoint_a, checkpoint_b, temp_elo_scores)
            elif preferred_checkpoint == checkpoint_b:
                temp_elo_scores = update_elo_score(checkpoint_b, checkpoint_a, temp_elo_scores)
    
    return temp_elo_scores

def update_elo_score(winner, loser, scores, K=32):
    if winner not in scores:
        scores[winner] = 1500
    if loser not in scores:
        scores[loser] = 1500
    
    # Calculate expected scores
    expected_winner = 1 / (1 + 10 ** ((scores[loser] - scores[winner]) / 400))
    expected_loser = 1 / (1 + 10 ** ((scores[winner] - scores[loser]) / 400))
    
    # Update ELO scores
    scores[winner] += K * (1 - expected_winner)
    scores[loser] += K * (0 - expected_loser)
    
    return scores

# Function to randomly select a category folder
def get_random_category_folder(base_folder):
    categories = [folder for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder))]
    return choice(categories)

# Add a "Clear CSV Data" button at the top
if st.button("Clear CSV Data"):
    clear_csv_data(csv_folder)

# Read existing choices from CSV
existing_choices = read_choices_from_csv(csv_folder)

# Initialize choices list and last saved index
if 'choices' not in st.session_state:
    st.session_state.choices = []
if 'current_pair_index' not in st.session_state:
    st.session_state.current_pair_index = 0

# Add buttons for each category
category_buttons = {
    "Broken": st.sidebar.button("Broken"),
    "Scratches": st.sidebar.button("Scratches"),
    "Chipped": st.sidebar.button("Chipped"),
    "PPO": st.sidebar.button("PPO")
}

# Handle category filtering
selected_category = None
for category, button_pressed in category_buttons.items():
    if button_pressed:
        selected_category = category
        st.session_state.current_pair_index = 0  # Reset the index when category changes
        break

# Process existing comparison results to update ELO scores
if selected_category:
    st.write(f"Selected Category: {selected_category}")
    display_elo_scores = process_comparison_results(csv_folder, category=selected_category)
else:
    display_elo_scores = process_comparison_results(csv_folder)

# Display the number of comparisons made so far
if existing_choices:
    last_serial_number = int(existing_choices[-1]['Serial_Number'])
else:
    last_serial_number = 0

st.sidebar.write(f"Number of comparisons made so far: {last_serial_number}")

# Randomly select a category folder
selected_category_folder = get_random_category_folder(base_folder)

# Generate all combinations excluding already saved pairs
all_combinations = generate_combinations(selected_category_folder)
if existing_choices:
    all_combinations = [(c1, c2, img) for c1, c2, img in all_combinations
                        if not any((c1 == choice['CheckpointA'] and c2 == choice['CheckpointB'] and img == choice['Image_Name']) or
                                   (c1 == choice['CheckpointB'] and c2 == choice['CheckpointA'] and img == choice['Image_Name'])
                                   for choice in existing_choices)]

# Initialize or increment pair index
if 'current_pair_index' not in st.session_state:
    st.session_state.current_pair_index = 0

if all_combinations:
    current_pair_index = st.session_state.current_pair_index
    if current_pair_index < len(all_combinations):
        checkpoint1, checkpoint2, image_name = all_combinations[current_pair_index]

        img1_path = os.path.join(base_folder, selected_category_folder, checkpoint1, image_name)
        img2_path = os.path.join(base_folder, selected_category_folder, checkpoint2, image_name)

        img1 = read_and_resize_image(img1_path)
        img2 = read_and_resize_image(img2_path)

        if img1 is not None and img2 is not None:
            col1, col2 = st.columns([6, 1])
            with col1:
                st.image(img1, caption=f"Image {current_pair_index + 1}A", use_column_width=True)
            with col2:
                if st.button(f"Prefer Image {current_pair_index + 1}A"):
                    choice = {
                        'CheckpointA': checkpoint1,
                        'CheckpointB': checkpoint2,
                        'Image_Name': image_name,
                        'Preferred_Checkpoint': checkpoint1,
                        'Category': selected_category_folder  # Ensure this is correct
                    }
                    update_elo(checkpoint1, checkpoint2)
                    st.session_state.current_pair_index += 1
                    save_to_csv(choice, csv_folder)
                    st.experimental_rerun()

            col3, col4 = st.columns([6, 1])
            with col3:
                st.image(img2, caption=f"Image {current_pair_index + 1}B", use_column_width=True)
            with col4:
                if st.button(f"Prefer Image {current_pair_index + 1}B"):
                    choice = {
                        'CheckpointA': checkpoint1,
                        'CheckpointB': checkpoint2,
                        'Image_Name': image_name,
                        'Preferred_Checkpoint': checkpoint2,
                        'Category': selected_category_folder  # Ensure this is correct
                    }
                    update_elo(checkpoint2, checkpoint1)
                    st.session_state.current_pair_index += 1
                    save_to_csv(choice, csv_folder)
                    st.experimental_rerun()

            # Display image name at the bottom
            st.write(f"Current Image: {image_name}")

            # Add a "Tie" button
            if st.button("Tie"):
                choice = {
                    'CheckpointA': checkpoint1,
                    'CheckpointB': checkpoint2,
                    'Image_Name': image_name,
                    'Preferred_Checkpoint': 'Tie',
                    'Category': selected_category_folder  # Ensure this is correct
                }
                st.session_state.current_pair_index += 1
                save_to_csv(choice, csv_folder)
                st.experimental_rerun()

            # Display ELO scores
            sorted_elo_scores = dict(sorted(display_elo_scores.items(), key=lambda item: item[1], reverse=True))
            st.sidebar.write("Current ELO Scores:")
            for checkpoint, score in sorted_elo_scores.items():
                st.sidebar.write(f"{checkpoint}: {score}")

        else:
            st.error("Error reading images. Please check the image paths.")
        
    else:
        st.info("Image comparison completed!")
else:
    st.info("No image combinations available for comparison.")
