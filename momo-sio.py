import copy
from pathlib import Path
import sys
import datetime
import os
import base64
import io

from aiohttp import web

sys.path.append("./taming-transformers")

from PIL import Image
from typing import Optional, List
from logic import VQGANCLIPRun

session = {}

output_base = "sio-outputs"
outputdir = "outputs"

def getOutputPath(outputdir):
    return Path(output_base, outputdir)

async def generate_image(
        text_input: str = "the first day of the waters",
        vqgan_ckpt: str = "vqgan_imagenet_f16_16384",
        num_steps: int = 300,
        image_x: int = 300,
        image_y: int = 300,
        init_image: Optional[Image.Image] = None,
        image_prompts: List[Image.Image] = [],
        continue_prev_run: bool = False,
        seed: Optional[int] = None,
        mse_weight: float = 0,
        mse_weight_decay: float = 0,
        mse_weight_decay_steps: int = 0,
        tv_loss_weight: float = 1e-3,
        use_scrolling_zooming: bool = False,
        translation_x: int = 0,
        translation_y: int = 0,
        rotation_angle: float = 0,
        zoom_factor: float = 1,
        transform_interval: int = 10,
        use_cutout_augmentations: bool = True,
) -> None:

    ### Init -------------------------------------------------------------------
    run = VQGANCLIPRun(
        text_input=text_input,
        vqgan_ckpt=vqgan_ckpt,
        num_steps=num_steps,
        image_x=image_x,
        image_y=image_y,
        seed=seed,
        init_image=init_image,
        image_prompts=image_prompts,
        continue_prev_run=continue_prev_run,
        mse_weight=mse_weight,
        mse_weight_decay=mse_weight_decay,
        mse_weight_decay_steps=mse_weight_decay_steps,
        tv_loss_weight=tv_loss_weight,
        use_scrolling_zooming=use_scrolling_zooming,
        translation_x=translation_x,
        translation_y=translation_y,
        rotation_angle=rotation_angle,
        zoom_factor=zoom_factor,
        transform_interval=transform_interval,
        use_cutout_augmentations=use_cutout_augmentations,
    )

    ### Load model -------------------------------------------------------------

    if continue_prev_run is True:
        run.load_model(
            prev_model=session["model"],
            prev_perceptor=session["perceptor"],
        )
        prev_run_id = session["run_id"]

    else:
        # Remove the cache first! CUDA out of memory
        if "model" in session:
            del session["model"]

        if "perceptor" in session:
            del session["perceptor"]

        session["model"], session["perceptor"] = run.load_model()
        prev_run_id = None

    # Generate random run ID
    # Used to link runs linked w/ continue_prev_run
    # ref: https://stackoverflow.com/a/42703382/13095028
    # Use URL and filesystem safe version since we're using this as a folder name
    run_id = session["run_id"] = base64.urlsafe_b64encode(
        os.urandom(6)
    ).decode("ascii")

    run_start_dt = datetime.datetime.now()

    ### Model init -------------------------------------------------------------
    if continue_prev_run is True:
        run.model_init(init_image=session["prev_im"])
    elif init_image is not None:
        run.model_init(init_image=init_image)
    else:
        run.model_init()

    ### Iterate ----------------------------------------------------------------
    session['keep_running'] = True

    while session['keep_running']:
        _, im = run.iterate()
        
        # turn image into PNG bytes
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        
        # send PNG bytes to client
        await sio.emit('generate_image_update', buf.getvalue())
        
        # save current image as last image in session
        session["prev_im"] = im
        try:
            await sio.call('uthere?', to=session['sid'], timeout=5)
        except: #timeout exception called when client disconnects
            session['keep_running'] = False
            print('Nobody there, stopping.')
    # end of session, return to ready
    session['continue_prev_run'] = True
    await set_gen_state(GenState.Ready)
            
def convert_to_prompts(dict):
    result = ""
    for key, value in dict.items():
        result += f"{key}:{value}|"
    return result[:-1] # don't return last pipe

from enum import Enum
class GenState(str, Enum):
    Submitted = "submitted"
    Ready = "ready"
    Running = "running"
    Waking = "waking"
    
async def set_gen_state(newState):
    print(f"new state: {newState}")
    session['genState'] = newState
    await send_current_data()

import socketio

sio = socketio.AsyncServer(async_mode='aiohttp', logger=False, cors_allowed_origins="*", ignore_queue=True)
app = web.Application()
sio.attach(app)

app.add_routes([web.static('/', "../vqgan-js", show_index=True)])

session['continue_prev_run'] = False
session['genState'] = GenState.Ready

@sio.on('generate')
async def start_run(sid, data):
    session['sid'] = sid # track original requester, stop if they disconnect.
    
    print(data)
    # convert preset to data
    size_x = data['size_x']
    size_y = data['size_y']
    text_input = convert_to_prompts(data['prompts'])
    session['current_data'] = data
    await set_gen_state(GenState.Running)
    # Start the run!
    sio.start_background_task(generate_image, text_input, "vqgan_imagenet_f16_16384", -1, size_x, size_y, None, [], session['continue_prev_run'], 99, 0, 0, 0, 1e-3, False, 0, 0, 0, 1, 10, True);

@sio.on('reset')
async def reset_run(sid):
    session['continue_prev_run'] = False
    print("reset")

@sio.on('pause')
async def pause_run(sid):
    if(session['keep_running']):
        session['keep_running'] = False
    else:
        await set_gen_state(GenState.Ready)

@sio.event
async def connect(sid, environ, auth):
    print('connect ', sid)
    if 'prev_im' in session:
        print('sending previous image')
        # turn image into PNG bytes
        buf = io.BytesIO()
        session['prev_im'].save(buf, format='PNG')
        # send PNG bytes to client
        await sio.emit('generate_image_update', buf.getvalue())
    await send_current_data()
        
async def send_current_data():
    # send current state
    payload = {}
    if 'genState' in session:
        payload['genState'] = session['genState']
        
    # may not have current_data if nothing submitted yet
    if 'current_data' in session:
        payload['current_data'] = session['current_data']
        
    print(f"sending payload {payload}")
    
    await sio.emit('current_data', payload)

@sio.event
async def disconnect(sid):
    print('disconnect ', sid)

if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0', port=8501)
    session['genState'] = GenState.Ready