import os
import subprocess



def convert_mp4v_to_openh264(input_folder, output_folder):

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all mp4 files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_openh264.mp4")

            # FFmpeg command to convert MP4V to OpenH264
            command = [
                "ffmpeg",
                "-i", input_path,
                "-c:v", "libopenh264",
                "-crf", "23",
                "-preset", "medium",
                "-c:a", "copy",
                output_path
            ]

            # Execute the FFmpeg command
            subprocess.run(command, check=True)

    print("Conversion completed.")


#     command = [
#     "ffmpeg",
#     "-i", input_path,
#     "-c:v", "libx264",   
#     "-crf", "23",
#     "-preset", "medium",
#     "-c:a", "aac",       
#     "-b:a", "128k",      
#     output_path
# ]


