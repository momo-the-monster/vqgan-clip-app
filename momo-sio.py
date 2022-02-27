import urllib
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
        image_x: int = 260,
        image_y: int = 355,
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

        if "stack" in session:
            del session["stack"]

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
    
    if init_image is not None:
        session['input_image'] = init_image

    ### Model init -------------------------------------------------------------
    if continue_prev_run is True:
        run.model_init(init_image=session["prev_im"])
    elif init_image is not None:
        run.model_init(init_image=init_image)
    else:
        run.model_init()

    ### Iterate ----------------------------------------------------------------
    step_counter = 0
    session['keep_running'] = True

    try:
        while session['keep_running']:
            _, im = run.iterate()
            
            # turn image into PNG bytes
            buf = io.BytesIO()
            im.save(buf, format='PNG')
            
            # send PNG bytes to client
            await sio.emit('generate_image_update', buf.getvalue())
            
            # save current image as last image in session
            session["prev_im"] = im
            stack_element = {"image":im, "step":step_counter}
            if("stack" in session):
                session["stack"].insert(0,stack_element)
            else:
                session["stack"] = [stack_element]
            # purge images over x
            if(len(session["stack"]) > 7):
                session["stack"].pop()
            step_counter += 1
            try:
                await sio.call('uthere?', to=session['sid'], timeout=5)
            except: #timeout exception called when client disconnects
                session['keep_running'] = False
                print('Nobody there, stopping.')
    except RuntimeError:
        await send_error("Out of Memory, try reducing Image Size")
    except:
        await send_error("Something went wrong, restarting")
    finally:
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
session["stack"] = []
session['input_image'] = None

@sio.on('generate')
async def start_run(sid, data):
    session['sid'] = sid # track original requester, stop if they disconnect.
    
    # print(data)
    # convert preset to data
    size_x = data['size_x']
    size_y = data['size_y']
    text_input = convert_to_prompts(data['prompts'])
    session['current_data'] = data
    
    if 'image_prompt' in data:
        imageBinaryBytes = data['image_prompt']
        imageStream = io.BytesIO(imageBinaryBytes)
        inputImage = Image.open(imageStream)
    
    await set_gen_state(GenState.Running)
    # Start the run!
    sio.start_background_task(generate_image, text_input, "vqgan_imagenet_f16_16384", -1, size_x, size_y, inputImage, [], session['continue_prev_run'], 99, 0, 0, 0, 1e-3, False, 0, 0, 0, 1, 10, True);

@sio.on('reset')
async def reset_run(sid):
    session['continue_prev_run'] = False
    session["stack"] = []
    print("reset")

@sio.on('pause')
async def pause_run(sid):
    if(session['keep_running']):
        session['keep_running'] = False
    else:
        await set_gen_state(GenState.Ready)

stack_output_dir = "E:\sync-test\Sync\sio"

@sio.on('save')
async def save(sid):
    if("stack" in session):
        i = 0
        run_id = session["run_id"]
        for entry in session["stack"]:
            step = entry["step"]
            image = entry["image"] 
            save_dir = Path(stack_output_dir, run_id)
            if not save_dir.exists():
                os.mkdir(save_dir)
            save_location = Path(save_dir, f"{run_id}_{step}.png")
            print(save_location.absolute())
            image.save(save_location.absolute())
            i += 1
        
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
        
    # print(f"sending current-state {payload}")
    
    await sio.emit('current_data', payload)

async def send_error(error):
    payload = {}
    payload['message'] = error
    await sio.emit('error', payload)

@sio.event
async def disconnect(sid):
    print('disconnect ', sid)

if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0', port=8501)