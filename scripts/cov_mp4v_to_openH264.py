import os
import subprocess



def convert_mp4v_to_openh264(input_folder, output_folder):

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all mp4 files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            print(f"Converting {filename}")
            if "_openh264" in filename:
                print(f"Skipping already converted file: {filename}")
                continue

            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_openh264.mp4")
                        
            # FFmpeg command to convert MP4V to OpenH264
            command = [
                "ffmpeg",
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", "medium",   
                "-crf", "23",                
                "-c:a", "aac",       
                "-b:a", "128k",      
                output_path
            ]

            # Execute the FFmpeg command
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            #subprocess.run(command, check=True)
            if result.returncode == 0:
                print(f"Successfully converted {filename}")
                os.remove(input_path)
                print(f"Deleted original file: {input_path}")
            else:
                print(f"Failed to convert {filename}")


    print("Conversion completed.")





