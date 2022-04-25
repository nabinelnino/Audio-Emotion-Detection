from fastapi import FastAPI,UploadFile 
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import numpy as np

from modal.utils import load_net, SplitWavAudioMubin, preprocess
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences



emotions = {
    0 : 'neutral',
    1 : 'calm',
    2 : 'happy',
    3 : 'sad',
    4 : 'angry',
    5 : 'fearful',
    6 : 'disgust',
    7 : 'suprised'   
}

file_path = "OSR_us_000_0036_8k - Copy.wav"
folder_path = 'C:\\Users\\elnin\\Desktop\\Capstone\\final\\temp'

model = load_net()

app = FastAPI()

origins = [
    "http://localhost",
    "null",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def Upload(file: UploadFile):
    '''
    returns filename with emotion prediction for uploaded file
    '''
    split_wav = SplitWavAudioMubin(folder_path, file.filename,file.file)
    split_wav.multiple_split(sec_per_split=10)
    total_predictions = []
    file_list = [f for f in os.listdir(folder_path) if f != file.filename]
    for filename in file_list:
        f = os.path.join(folder_path, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
            x = preprocess(f) # 'output.wav' file preprocessing.
            print(x.shape)
            x_padded = pad_sequences(x, maxlen=400)
            predictions = model.predict(x_padded, use_multiprocessing=True)
        
            pred_list = list(predictions)
            pred_np = np.squeeze(np.array(pred_list).round(3).tolist(), axis=0).tolist() # Get rid of 'array' & 'dtype' statments.
            total_predictions.append(pred_np)
    result = {v: total_predictions[0][k] for k,v in emotions.items()}
    [os.remove(os.path.join(folder_path, f)) for f in file_list]
    return {"message":f"uploaded {file.filename}", "predictions": result}



