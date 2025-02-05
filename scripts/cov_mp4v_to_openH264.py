import os
import subprocess

def convert_mp4v_to_openh264(input_folder, output_folder):
    print(f"Starting conversion in folder: {input_folder}")
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp4'):
            # Check if both thumbnail and rendered video exist
            thumbnail_path = os.path.join('data', 'thumbnails', f"{os.path.splitext(filename)[0]}.jpg")
            final_output_path = os.path.join(output_folder, filename)
            
            if os.path.exists(thumbnail_path) and os.path.exists(final_output_path):
                print(f"Skipping {filename} - already converted with thumbnail")
                continue
                
            input_path = os.path.join(input_folder, filename)
            temp_output_path = os.path.join(output_folder, f"temp_{filename}")
            
            # Generate thumbnail first
            thumbnail_command = [
                'ffmpeg', '-i', input_path,
                '-vf', 'thumbnail,scale=300:-1',
                '-frames:v', '1',
                thumbnail_path
            ]
            subprocess.run(thumbnail_command, capture_output=True)
            
            command = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264',
                '-loglevel', 'error', 
                temp_output_path
            ]
            subprocess.run(command, capture_output=True)
            
            os.replace(temp_output_path, final_output_path)
            print(f"Completed conversion for: {filename}")
