import os 

folder_path = './features/clip_text_features'
condition = lambda x : x.endswith('.npz')

file_list = os.listdir(folder_path)

for filename in file_list:
    if condition(filename):
        # Modify the filename as needed
        new_filename = 'qid' + filename
        # Construct the full paths
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        # Rename the file
        os.rename(old_path, new_path)

        print(f'Renamed: {filename} to {new_filename}')
print('File renaming completed.')
