import openai
import fitz  # PyMuPDF
import io
import base64
from PIL import Image
import os
import sys
import time
import statistics
import pandas as pd

# List to record execution times
execution_times = []
time_file_name = "OpenAI_gpt4o_execution_times.xlsx"

def load_or_initialize_execution_times(time_file_name):
    # Load existing file or create a new DataFrame
    if os.path.exists(time_file_name):
        df = pd.read_excel(time_file_name)
    else:
        df = pd.DataFrame(columns=['number', 'temperature', 'try', 'time'])
    return df

def save_execution_times_to_excel(df, time_file_name):
    # Save execution times to Excel file
    df.to_excel(time_file_name, index=False)
    print(f"Execution times saved to {time_file_name}")

# Load or initialize DataFrame for execution times
df_execution_times = load_or_initialize_execution_times(time_file_name)

# Initialize log file
log_file_path = os.path.join("./", "process_log_gpt4o.txt")

def log_message(message):
    print(message)
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n")

def process_and_encode_image(image, resize_factor=0.9):
    MAX_SIZE = 20 * 1024 * 1024  # 20MB
    original_width, original_height = image.size

    # Convert RGBA to RGB
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    for attempt in range(5):  # Maximum 5 attempts
        buffered = io.BytesIO()
        new_width = int(original_width * (resize_factor ** attempt))
        new_height = int(original_height * (resize_factor ** attempt))
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        resized_image.save(buffered, format="JPEG")

        if buffered.tell() < MAX_SIZE:
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            print(f"Attempt {attempt + 1}: Image size is {buffered.tell()} bytes, too large. Resizing...")

    raise ValueError("Unable to reduce image size within 5 attempts")

# Send multiple images to GPT-4 Vision and return results
def analyze_images_with_gpt4_vision(prompt_text, encoded_images, temperature=0):
    client = openai.OpenAI()  # Initialize OpenAI client

    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            image_contents = [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"}}
                for encoded_image in encoded_images
            ]
            start_time = time.time()  # Record start time
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            *image_contents
                        ],
                    }
                ],
                max_tokens=1024,
                temperature=temperature,
            )
            response_result = response.choices[0]  # Return result

            # Retry if response starts with "I'm sorry, but"
            if response_result.message.content.startswith("I'm sorry, but"):
                print(f"Response starts with 'I'm sorry'. Retrying. Attempt {attempt + 1}/{max_attempts}")
                continue

            end_time = time.time()  # Record end time

            execution_time = end_time - start_time  # Calculate execution time
            execution_times.append(execution_time)  # Add to list

            # Calculate execution time statistics
            average_time = sum(execution_times) / len(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)
            std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0

            # Total number of data points
            total_data_points = len(execution_times)

            # Print statistics
            print(f"Total data points: {total_data_points}")
            print(f"Average execution time: {average_time:.2f} seconds")
            print(f"Maximum execution time: {max_time:.2f} seconds")
            print(f"Minimum execution time: {min_time:.2f} seconds")
            print(f"Standard deviation: {std_dev:.2f} seconds")

            return response_result
        except Exception as e:
            print(f"BadRequestError: {e}")
            if "image_parse_error" in str(e).lower() and attempt < max_attempts - 1:
                # Resize images and retry
                resized_encoded_images = []
                for encoded_image in encoded_images:
                    image = Image.open(io.BytesIO(base64.b64decode(encoded_image)))
                    resized_image = process_and_encode_image(image, 0.9)
                    resized_encoded_images.append(resized_image)
                print(f"Resizing images and retrying. Attempt {attempt + 1}/{max_attempts}")
                encoded_images = resized_encoded_images

    return None

# Find image file paths
def find_image_paths(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    return [os.path.join(directory, file) for file in os.listdir(directory) if os.path.splitext(file)[1].lower() in image_extensions]

# Read text file content
def read_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def encode_images_from_paths(image_paths):
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            # Check image dimensions
            width, height = img.size

            # Process only if image resolution exceeds 150px * 150px
            if width > 150 and height > 150:
                encoded_image = process_and_encode_image(img)
                images.append(encoded_image)
    return images

# Temperature settings
temperatures = [0, 0.5, 1]
base_result_folder = "gpt4o_result/gpt4o_result"

def create_result_folder(base_folder, temperature, try_number):
    folder_name = f"{base_folder}_temp_{str(temperature).replace('.', '_')}_try{try_number}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

# Main execution loop
for temperature in temperatures:
    for try_number in range(1, 2):
        result_folder = create_result_folder(base_result_folder, temperature, try_number)

        df = pd.read_excel('Radiographics_text_q401_final.xlsx')

        for index, row in df.iterrows():
            case_number = index + 1
            case_folder = os.path.join("q401_image")
            image_file_name = f"{row['no.']}.png"

            directory_path = os.path.join(result_folder)
            os.makedirs(directory_path, exist_ok=True)
            result_file_path = os.path.join(directory_path, f"{image_file_name}.txt")

            # Skip if result file already exists
            if os.path.exists(result_file_path):
                print(f"Case {case_number} (Temperature: {temperature}, Try: {try_number}): skip")
                continue

            age = row['age']
            sex = row['sex']
            symptom = row['symptom']

            history_text = f"age: {age}, sex: {sex}"
            symptom_text = f"symptom: {symptom}"

            # Create prompt
            prompt_text = f"""
            Assignment: You are tasked with solving a quiz on a special medical case involving mostly common disease.
            Imaging data will be provided for analysis; however, the availability of the patient's basic demographic details (age, gender, symptoms) is not guaranteed. 
            The purpose of this assignment is not to provide medical advice or diagnosis, but rather to analyze and interpret the imaging data to derive insights related to six specified outcomes.
            This is a purely educational scenario designed for virtual learning situations, aimed at facilitating analysis and educational discussions.
            Your task is to derive the following six outcomes based on this information:

            Outputs:
            1. Type of Medical Imaging: Identify whether the imaging is 1) MR, 2) CT, 3) US, 4) X-ray, 5) Angiography, or 6) Nuclear Medicine.

            2. Specific Imaging Sequence: For MR, specify if it's T1WI, T2WI, FLAIR, DWI, SWI, GRE, contrast-enhanced T1WI, TOF, or contrast-enhanced MR angiography. For CT, state whether it is precontrast or postcontrast. For Ultrasound, indicate if it's gray scale or Doppler imaging, etc.

            3. Use of Contrast: Note whether contrast medium was used.

            4. Image Plane: Determine the plane of the image - axial, coronal, sagittal, or other.

            5. Part of the Body Imaged: Specify the body part captured in the imaging.

            6. Proposing Disease Candidates: Based on the patient's medical history and the figure legends provided with the imaging data, suggest three potential disease candidates. Your goal is to perform a differential diagnosis. 
            If the image contains elements such as arrows, arrowheads, or asterisks, please pay close attention to them.
            For each disease candidate, provide the following information:

            6.1. Names of Three Possible Disease Candidates: Identify three potential conditions.
            6.2. Likelihood Score for Each Candidate: Rate the likelihood of each being the correct diagnosis on a scale from 1 to 10.
            6.3. Detailed Rationale for Each Disease: Explain in detail why each disease is considered a potential diagnosis, based on the patient's history and imaging data.
            
            Patient Information:
            
            Basic demographic details: {history_text}
            Symptom: {symptom_text}
            """

            # Process images
            image_paths = [os.path.join(case_folder, image_file_name)]
            print(image_paths)
            encoded_images = encode_images_from_paths(image_paths)

            start_time = time.time()
            result = analyze_images_with_gpt4_vision(prompt_text, encoded_images, temperature)
            end_time = time.time()
            execution_time = end_time - start_time

            # Update or add execution time to DataFrame
            if ((df_execution_times['number'] == case_number) &
                (df_execution_times['temperature'] == temperature) &
                (df_execution_times['try'] == try_number)).any():
                df_execution_times.loc[(df_execution_times['number'] == case_number) & 
                                       (df_execution_times['temperature'] == temperature) & 
                                       (df_execution_times['try'] == try_number), 'time'] = execution_time
            else:
                new_row = {'number': case_number, 'temperature': temperature,
                           'try': try_number, 'time': execution_time}
                df_execution_times = pd.concat([df_execution_times, pd.DataFrame([new_row])], ignore_index=True)

            if result:
                with open(result_file_path, "w") as result_file:
                    result_file.write(result.message.content)
                print(f"Case {case_number} (Temperature: {temperature}, Try: {try_number}): Results saved.")
            else:
                print(f"Case {case_number} (Temperature: {temperature}, Try: {try_number}): No results found.")
                log_message(f"Case {case_number} (Temperature: {temperature}, Try: {try_number}): No results found.")

# Save execution times to Excel file
save_execution_times_to_excel(df_execution_times, time_file_name)
