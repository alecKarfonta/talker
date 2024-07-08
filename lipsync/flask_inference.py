from glob import glob
import json
import pickle
import threading
import torch
from time import  strftime
import os, sys
import tempfile
from flask import Flask, jsonify, request, render_template, send_file,redirect,send_from_directory
from uuid import uuid4 
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
import numpy as np
#import yappi
#download models, script will not download over existing
os.system("bash scripts/download_models.sh")

from src.facerender.pirender_animate import AnimateFromCoeff_PIRender


import platform

app = Flask(__name__,static_folder="cache")

# this is the same as the arguments here 
# https://github.com/OpenTalker/SadTalker/blob/cd4c0465ae0b54a6f85af57f5c65fec9fe23e7f8/inference.py#L100
class SadTalker_Settings:
    ref_eyeblink=None
    ref_pose=None
    checkpoint_dir='./checkpoints'
    result_dir='./cache'
    pose_style=0
    batch_size=2
    size=256
    expression_scale=1.0
    input_yaw=None
    input_pitch=None
    input_roll=None
    enhancer=None
    background_enhancer=None
    cpu=False
    face3dvis=False
    still=False
    preprocess='crop'
    verbose=False
    old_version=False
    net_recon='resnet50'
    init_path=None
    use_last_fc=False
    bfm_folder='./checkpoints/BFM_Fitting/'
    bfm_model='BFM_model_front.mat'
    focal=1015.0
    center=112.0
    camera_d=10.0
    z_near=5.0
    z_far=15.0
    device='cuda'
    # these are extra
    face_folder='./faces'
    coeff_data=None

global_settings=SadTalker_Settings()
os.makedirs(global_settings.face_folder, exist_ok=True)

current_root_path = os.path.split(sys.argv[0])[0]
sadtalker_paths = init_path(global_settings.checkpoint_dir, os.path.join(current_root_path, 'src/config'), global_settings.size, global_settings.old_version, global_settings.preprocess)
preprocess_model = CropAndExtract(sadtalker_paths, global_settings.device)
audio_to_coeff = Audio2Coeff(sadtalker_paths,  global_settings.device)    
#animate_from_coeff = AnimateFromCoeff(sadtalker_paths, global_settings.device)
animate_from_coeff = AnimateFromCoeff_PIRender(sadtalker_paths, global_settings.device)


preprocess_data_dict = {}

######### sadtalker_main ################
# need to handle file objects in args
def sadtalker_main(str_wavfile, str_imgpath, settings=SadTalker_Settings(), preprocess_data=None, args=None):
    #torch.backends.cudnn.enabled = False
    pic_path = str_imgpath
    audio_path = str_wavfile
    save_dir = '/tmp/sadtalker_run_' + strftime("%Y_%m_%d_%H.%M.%S")
    os.makedirs(save_dir, exist_ok=True)
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)

    if preprocess_data == None and str_imgpath not in preprocess_data_dict:

        print('3DMM Extraction for source image')
        first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, settings.preprocess,\
                                                                                 source_image_flag=True, pic_size=settings.size)
        preprocess_data_dict[str_imgpath] = {'first_coeff_path':first_coeff_path,'crop_pic_path':crop_pic_path,'crop_info':crop_info}
    elif preprocess_data == None and str_imgpath in preprocess_data_dict:

        print(f"Using saved preprecessed data at {str_imgpath}")
        preprocess_data = preprocess_data_dict[str_imgpath]
        first_coeff_path = preprocess_data['first_coeff_path']
        crop_pic_path = preprocess_data['crop_pic_path']
        crop_info = preprocess_data['crop_info']
        
    else:
        first_coeff_path = preprocess_data['first_coeff_path']
        crop_pic_path = preprocess_data['crop_pic_path']
        crop_info = preprocess_data['crop_info']
    # end here
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    # not implemented (needs file reduction)
    if settings.ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(settings.ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(settings.ref_eyeblink, ref_eyeblink_frame_dir, settings.preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None
    # not implmented (needs file reduction)
    if settings.ref_pose is not None:
        if settings.ref_pose == settings.ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(settings.ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(settings.ref_pose, ref_pose_frame_dir, settings.preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path=None

    #audio2ceoff
    batch = get_data(first_coeff_path, audio_path, settings.device, ref_eyeblink_coeff_path, still=settings.still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, settings.pose_style, ref_pose_coeff_path,return_filepaths=False)

    # 3dface render (not implemented) (needs file reduction)
    if settings.face3dvis:
        print('3D face visualization')
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(settings.args, settings.device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
    
    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                settings.batch_size, settings.input_yaw, settings.input_pitch, settings.input_roll,
                                expression_scale=settings.expression_scale, still_mode=settings.still, 
                                preprocess=settings.preprocess, size=settings.size,create_files=False, facemodel=args.facerender)
   

    # over riding audio path with temp save file from flask until ffmpeg code uses pipes
    if not isinstance(str_wavfile,str):
        audio_path.seek(0) # need to reset file pointer to start from last reads
        temp_junk =tempfile.NamedTemporaryFile().name
        audio_path.save(temp_junk)
        data['audio_path'] = temp_junk # override with argument
    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                enhancer=settings.enhancer, background_enhancer=settings.background_enhancer, preprocess=settings.preprocess, img_size=settings.size)

    return result

#########################

@app.get("/")
async def root():
    return render_template('index.html')

def temp_save(obj_file,str_filename):
    obj_file.save(str_filename)

@app.route('/run_sadtalker', methods = ['POST'])
async def run_sadtalker(): 
    if request.files['face_file'].filename == '' :
        return "No Face File"
    if request.files['wav_file'].filename == '' : 
        return "No Wav File"
    try:
        final_file=sadtalker_main(request.files['wav_file'],request.files['face_file'])
        return send_file(final_file,"application/octet-stream")
    except ValueError as ve:
        return "Error: " + str(ve)
    

@app.route('/upload_face', methods = ['POST'])
async def upload_face(): 
    if request.files['face_file'].filename == '' :
        return "No Face File"    
    face_name=request.form.get('name',None)
    if face_name == None:
        return "No Face Name Supplied"
    face_name=os.path.basename(face_name)
    face_dir='faces/'+ face_name 
    try:
        os.makedirs(face_dir)
        temp_first_coeff_path, temp_crop_pic_path, temp_crop_info =  preprocess_model.generate(request.files['face_file'], face_dir, "crop",\
                                                                                 source_image_flag=True, pic_size=global_settings.size,return_filepaths=False)
        
        
        f=open(face_dir + '/face.sadface','wb')
        pickle.dump({'first_coeff_path':temp_first_coeff_path,'crop_pic_path':temp_crop_pic_path,'crop_info':temp_crop_info},f,protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
    except FileExistsError as e:
        return "Error: A Face with that name exists"
        
    return '{"status":"success"}'

@app.route('/generate_avatar_message', methods = ['POST'])
async def generate_avatar_message():

    if request.files['wav_file'].filename == '' : 
        return "No Wav File"
    face_name=request.form.get('name',None)
    if face_name == None:
        return "No Face Name Supplied"
           
    face_name=os.path.basename(face_name)
    face_dir=global_settings.face_folder  +'/'+face_name
    # these will be replaced with the data in the face.sadface file at some point
    f=open(face_dir+'/face.sadface','rb')
    preprocess_data=pickle.load(f)
    f.close()

    final_file=sadtalker_main(request.files['wav_file'],"",global_settings,preprocess_data, args=request.args)
    

    return send_file(final_file,"application/octet-stream")

@app.get("/view_system_face")
async def view_system_face():
    face_name=request.args.get("name",None)
    if face_name == None:
        return 'No face name supplied'
    face_file=face_name + ".sadface"
    if os.path.isfile(global_settings.face_folder + "/" + face_file):        
        f = open(global_settings.face_folder + "/" + face_file,"r+")
        sadface_data = json.loads(f.read())
        f.close()
        return send_file(sadface_data['crop_pic_path'],"application/octet-stream")
    else:
        return "Face not Found"

@app.get("/get_system_faces")
async def get_system_faces():
    faces=glob(global_settings.face_folder + "/*.sadface")
    for idx, face in enumerate(faces):
        faces[idx]=os.path.basename(face).replace(".sadface","")
    return faces

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7666)