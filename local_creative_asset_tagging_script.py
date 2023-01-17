import os
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz


def preprocess_image(directory, filename):
    """"preprocesses image for text dection by grayscaling and bluring image"""
    import cv2
    import os
    
    #' edit image for text detection
    #'    load image
    image_file = os.path.join(directory, filename)
    #'    convert image to grayscale
    img_gryscl = cv2.imread(image_file, 0)
    #'    blur grayscale image
    img_blur = cv2.blur(src = img_gryscl, ksize=(10, 10))

    #' save edited image for processing
    directory_new = r'C://Local Projects/Creative Asset Tagging/Edited Images/'
    os.chdir(directory_new)
    filename_new = 'opencv_' + filename
    cv2.imwrite(filename_new, img_blur)

    #' set new path
    path = os.path.join(directory_new, filename_new)

    return path


def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    import os
    import pandas as pd
    
    #' Set environment for Google to point to for authentication
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']='C:\\Local Projects\Creative Asset Tagging\lorenzo-342017-195048f1c304.json'

    client = vision.ImageAnnotatorClient()

    df_final = pd.DataFrame(columns = ['filename', 'ExtractedText'])

    # path = os.path.abspath('C://Local Projects/Creative Asset Tagging/Images for Processing/PEPSI Ad Examples/Rockstar4PackDigital624x1132NoLogo.png')
  
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    # find text lines within image
    image_text = texts[0].description
    image_text_for_df = io.StringIO(image_text)
    df_image_text = pd.read_csv(image_text_for_df, names=['ExtractedText'])
    df_image_text['key'] = 0
    
    # find filename from specified path
    filename_string = os.path.basename(path)
    fname_string_for_df = io.StringIO(filename_string)
    df_filename = pd.read_csv(fname_string_for_df, names=['filename'])
    df_filename['key'] = 0
    
    # create final dataframe with filename and associated text
    df_combined = df_filename.merge(df_image_text, how="outer", on = 'key')
    df_combined.drop('key', 1, inplace=True)
    df_final = df_combined.groupby('filename')['ExtractedText'].apply(' '.join).reset_index()
    
    return df_final


# create empty dataframe for extracted text
df_extracted_text = pd.DataFrame(columns = ['filename', 'ExtractedText'])

# assign directory
directory = 'C://Local Projects/Creative Asset Tagging/Images for Processing'

# iterate over files in directory
for filename in os.listdir(directory):
    image_file = preprocess_image(directory, filename)
    if os.path.isfile(image_file):
        df_final = detect_text(image_file)
        df_extracted_text = df_extracted_text.append(df_final)

# write dataframe to csv
df_extracted_text.to_csv('C://Local Projects/Creative Asset Tagging/preprocessed_extracted_text.csv', index = False)

# Expected Alt Text was created manually after writing the previous dataframe to a csv
# File contains outputs from 1st and 2nd passes as well as a manually created column of what the expected outputs should be for each image
# define path and file name to read for accuracy score
path = 'C://Local Projects/Creative Asset Tagging/'
filename = 'Expected_Alt_Text.xlsx'

# read csv file with expected outputs to dataframe
df_alttxt = pd.read_excel(path + filename)

# assign columns from dataframe to train and test lists
train_lst = df_alttxt['Expected_Alt_Text'].tolist()
test_orig_lst = df_alttxt['GAPI_Output'].tolist()
test_preproc_lst = df_alttxt['GAPI_PreProcessed_Output'].tolist()

# create empty list of ratio scores
mat_ratio_orig = []

# loop through train and test lists to retrieve fuzzy match ratio score for initial results
for i in range(len(train_lst)):
    mat_ratio_orig.append(fuzz.ratio(train_lst[i].lower(), test_orig_lst[i].lower()))

# calculate average ratio for initial results
np.average(mat_ratio_orig)

# create empty list for ratio scores for preprocessed results
mat_ratio_preproc = []

# loop through train and test lists to retrieve fuzzy match ratio score for preprocessed results
for i in range(len(train_lst)):
    mat_ratio_preproc.append(fuzz.ratio(train_lst[i].lower(), test_preproc_lst[i].lower()))

# calculate average ratio for preprocessed results
np.average(mat_ratio_preproc)
