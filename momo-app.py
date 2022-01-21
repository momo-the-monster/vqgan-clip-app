"""
This script is organized like so:
+ `if __name__ == "__main__" sets up the Streamlit UI elements
+ `generate_image` houses interactions between UI and the CLIP image 
generation models
+ Core model code is abstracted in `logic.py` and imported in `generate_image`
"""
import streamlit as st
from pathlib import Path
import sys
import datetime
import shutil
import json
import os
import base64

sys.path.append("./taming-transformers")

from PIL import Image
from typing import Optional, List
from omegaconf import OmegaConf
import imageio
import numpy as np

# Catch import issue, introduced in version 1.1
# Deprecate in a few minor versions
try:
    import cv2
except ModuleNotFoundError:
    st.warning(
        "Version 1.1 onwards requires opencv. Please update your Python environment as defined in `environment.yml`"
    )

from logic import VQGANCLIPRun

# Optional
try:
    import git
except ModuleNotFoundError:
    pass


def generate_image(
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
            prev_model=st.session_state["model"],
            prev_perceptor=st.session_state["perceptor"],
        )
        prev_run_id = st.session_state["run_id"]

    else:
        # Remove the cache first! CUDA out of memory
        if "model" in st.session_state:
            del st.session_state["model"]

        if "perceptor" in st.session_state:
            del st.session_state["perceptor"]

        st.session_state["model"], st.session_state["perceptor"] = run.load_model()
        prev_run_id = None

    # Generate random run ID
    # Used to link runs linked w/ continue_prev_run
    # ref: https://stackoverflow.com/a/42703382/13095028
    # Use URL and filesystem safe version since we're using this as a folder name
    run_id = st.session_state["run_id"] = base64.urlsafe_b64encode(
        os.urandom(6)
    ).decode("ascii")

    run_start_dt = datetime.datetime.now()

    ### Model init -------------------------------------------------------------
    if continue_prev_run is True:
        run.model_init(init_image=st.session_state["prev_im"])
    elif init_image is not None:
        run.model_init(init_image=init_image)
    else:
        run.model_init()

    ### Iterate ----------------------------------------------------------------
    step_counter = 0
    frames = []

    try:
        # Try block catches st.script_runner.StopExecution, no need of a dedicated stop button
        # Reason is st.form is meant to be self-contained either within sidebar, or in main body
        # The way the form is implemented in this app splits the form across both regions
        # This is intended to prevent the model settings from crowding the main body
        # However, touching any button resets the app state, making it impossible to
        # implement a stop button that can still dump output
        # Thankfully there's a built-in stop button :)
        while True:
            # While loop to accomodate running predetermined steps or running indefinitely
            status_text.text(f"Running step {step_counter}")

            _, im = run.iterate()

            if num_steps > 0:  # skip when num_steps = -1
                step_progress_bar.progress((step_counter + 1) / num_steps)
            else:
                step_progress_bar.progress(100)

            # At every step, display and save image
            im_display_slot.image(im, caption="Output image", output_format="PNG", width=None, use_column_width="never")
            st.session_state["prev_im"] = im

            # ref: https://stackoverflow.com/a/33117447/13095028
            # im_byte_arr = io.BytesIO()
            # im.save(im_byte_arr, format="JPEG")
            # frames.append(im_byte_arr.getvalue()) # read()
            frames.append(np.asarray(im))

            step_counter += 1

            if (step_counter == num_steps) and num_steps > 0:
                break

        # Stitch into video using imageio
        writer = imageio.get_writer("temp.mp4", fps=24)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        # Save to output folder if run completed
        runoutputdir = getOutputPath(outputdir) / (
            run_start_dt.strftime("%Y%m%dT%H%M%S") + "-" + run_id
        )
        runoutputdir.mkdir()

        # Save final image
        im.save(runoutputdir / f"{run_id}.png", format="PNG")
        im.save(runoutputdir / f"output.PNG", format="PNG") # leaving in for compatibility for now

        # Save init image
        if init_image is not None:
            init_image.save(runoutputdir / "init-image.jpg", format="JPEG")

        # Save image prompts
        for count, image_prompt in enumerate(image_prompts):
            image_prompt.save(
                runoutputdir / f"image-prompt-{count}.jpg", format="JPEG"
            )

        # Save animation
        shutil.copy("temp.mp4", runoutputdir / "anim.mp4")

        # Save metadata
        details = {
            "run_id": run_id,
            "num_steps": step_counter,
            "planned_num_steps": num_steps,
            "text_input": text_input,
            "init_image": False if init_image is None else True,
            "image_prompts": False if len(image_prompts) == 0 else True,
            "continue_prev_run": continue_prev_run,
            "prev_run_id": prev_run_id,
            "seed": run.seed,
            "Xdim": image_x,
            "ydim": image_y,
            "vqgan_ckpt": vqgan_ckpt,
            "start_time": run_start_dt.strftime("%Y%m%dT%H%M%S"),
            "end_time": datetime.datetime.now().strftime("%Y%m%dT%H%M%S"),
            "mse_weight": mse_weight,
            "mse_weight_decay": mse_weight_decay,
            "mse_weight_decay_steps": mse_weight_decay_steps,
            "tv_loss_weight": tv_loss_weight,
        }

        if use_scrolling_zooming:
            details.update(
                {
                    "translation_x": translation_x,
                    "translation_y": translation_y,
                    "rotation_angle": rotation_angle,
                    "zoom_factor": zoom_factor,
                    "transform_interval": transform_interval,
                }
            )
        if use_cutout_augmentations:
            details["use_cutout_augmentations"] = True

        if "git" in sys.modules:
            try:
                repo = git.Repo(search_parent_directories=True)
                commit_sha = repo.head.object.hexsha
                details["commit_sha"] = commit_sha[:6]
            except Exception as e:
                print("GitPython detected but not able to write commit SHA to file")
                print(f"raised Exception {e}")

        with open(runoutputdir / "details.json", "w") as f:
            json.dump(details, f, indent=4)

        status_text.text("Done!")  # End of run

    except st.script_runner.StopException as e:
        # Dump output to dashboard
        print(f"Received Streamlit StopException")
        status_text.text("Execution interruped, dumping outputs ...")
        writer = imageio.get_writer("temp.mp4", fps=24)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        # TODO: Make the following DRY
        # Save to output folder if run completed
        runoutputdir = getOutputPath(outputdir) / (
            run_start_dt.strftime("%Y%m%dT%H%M%S") + "-" + run_id
        )
        runoutputdir.mkdir()

        # Save final image
        im.save(runoutputdir / "output.PNG", format="PNG")

        # Save init image
        if init_image is not None:
            init_image.save(runoutputdir / "init-image.JPEG", format="JPEG")

        # Save image prompts
        for count, image_prompt in enumerate(image_prompts):
            image_prompt.save(
                runoutputdir / f"image-prompt-{count}.JPEG", format="JPEG"
            )

        # Save animation
        shutil.copy("temp.mp4", runoutputdir / "anim.mp4")

        # Save metadata
        details = {
            "run_id": run_id,
            "num_steps": step_counter,
            "planned_num_steps": num_steps,
            "text_input": text_input,
            "init_image": False if init_image is None else True,
            "image_prompts": False if len(image_prompts) == 0 else True,
            "continue_prev_run": continue_prev_run,
            "prev_run_id": prev_run_id,
            "seed": run.seed,
            "Xdim": image_x,
            "ydim": image_y,
            "vqgan_ckpt": vqgan_ckpt,
            "start_time": run_start_dt.strftime("%Y%m%dT%H%M%S"),
            "end_time": datetime.datetime.now().strftime("%Y%m%dT%H%M%S"),
            "mse_weight": mse_weight,
            "mse_weight_decay": mse_weight_decay,
            "mse_weight_decay_steps": mse_weight_decay_steps,
            "tv_loss_weight": tv_loss_weight,
        }

        if use_scrolling_zooming:
            details.update(
                {
                    "translation_x": translation_x,
                    "translation_y": translation_y,
                    "rotation_angle": rotation_angle,
                    "zoom_factor": zoom_factor,
                    "transform_interval": transform_interval,
                }
            )
        if use_cutout_augmentations:
            details["use_cutout_augmentations"] = True

        if "git" in sys.modules:
            try:
                repo = git.Repo(search_parent_directories=True)
                commit_sha = repo.head.object.hexsha
                details["commit_sha"] = commit_sha[:6]
            except Exception as e:
                print("GitPython detected but not able to write commit SHA to file")
                print(f"raised Exception {e}")

        with open(runoutputdir / "details.json", "w") as f:
            json.dump(details, f, indent=4)

        status_text.text("Done!")  # End of run

def get_new_outputdir():
    from random_word import RandomWords
    return '_'.join(RandomWords().get_random_words(limit=2, maxLength=5, maxDictionaryCount=1))

def get_latest_outputdir():
    import os, glob
    the_glob = glob.glob(os.path.join("new-output", '*/'))
    if the_glob:
        latest = max(the_glob, key=os.path.getmtime)
        if latest:
            return os.path.basename(Path(latest))

def getOutputPath(outputdir):
    return Path('new-output', outputdir)

def listDirs(rootdir):
    dirs = []
    for dir in Path(rootdir).iterdir():
        if(dir.is_dir()):
            dirs.append(dir.name)
    return dirs

if __name__ == "__main__":
    defaults = OmegaConf.load("defaults.yaml")

    st.set_page_config(page_title="Momo VQGAN-CLIP playground")
    
    # DEBUG print session state
    # st.session_state

    outputdir = get_latest_outputdir()
    if not outputdir:
        outputdir = get_new_outputdir()
        os.mkdir(getOutputPath(outputdir))

    # Determine what weights are available in `assets/`
    weights_dir = Path("assets").resolve()
    available_weight_ckpts = list(weights_dir.glob("*.ckpt"))
    available_weight_configs = list(weights_dir.glob("*.yaml"))
    available_weights = [
        i.stem
        for i in available_weight_ckpts
        if i.stem in [j.stem for j in available_weight_configs]
    ]

    # i.e. no weights found, ask user to download weights
    if len(available_weights) == 0:
        st.warning("No weights found in `assets/`, refer to `download-weights.sh`")
        st.stop()

    # Set vqgan_imagenet_f16_1024 as default if possible
    if "vqgan_imagenet_f16_1024" in available_weights:
        default_weight_index = available_weights.index("vqgan_imagenet_f16_1024")
    else:
        default_weight_index = 0
        
    # Show image display above form
    im_display_slot = st.empty()

    # progress under image display
    step_progress_bar = st.progress(0)

    # Start of input form
    with st.form("form-inputs"):
        
        # Main Page elements
        text_input = st.text_input(
            "Text prompt",
            help="VQGAN-CLIP will generate an image that best fits the prompt",
        )

        submitted = st.form_submit_button("Run!")
        
        # Sidebar elements

        continue_prev_run = st.sidebar.checkbox(
            "Continue previous run",
            value=defaults["continue_prev_run"],
            help="Use existing image and existing weights for the next run. If yes, ignores 'Use starting image'",
        )

        num_steps = st.sidebar.select_slider(
            "steps",
            options=[-1, 20,50,100,200,500],
            value=50
        )
        
        image_x = st.sidebar.number_input(
            "Xdim", value=defaults["Xdim"], help="Width of output image, in pixels"
        )
        image_y = st.sidebar.number_input(
            "ydim", value=defaults["ydim"], help="Height of output image, in pixels"
        )
        set_seed = st.sidebar.checkbox(
            "Set seed",
            value=defaults["set_seed"],
            help="Check to set random seed for reproducibility. Will add option to specify seed",
        )

        seed_widget = st.sidebar.empty()
        if set_seed is True:
            # Use text_input as number_input relies on JS
            # which can't natively handle large numbers
            # torch.seed() generates int w/ 19 or 20 chars!
            seed_str = seed_widget.text_input(
                "Seed", value=str(defaults["seed"]), help="Random seed to use"
            )
            try:
                seed = int(seed_str)
            except ValueError as e:
                st.error("seed input needs to be int")
        else:
            seed = None

        use_custom_starting_image = st.sidebar.checkbox(
            "Use starting image",
            value=defaults["use_starting_image"],
            help="Check to add a starting image to the network",
        )

        starting_image_widget = st.sidebar.empty()
        if use_custom_starting_image is True:
            init_image = starting_image_widget.file_uploader(
                "Upload starting image",
                type=["png", "jpeg", "jpg"],
                accept_multiple_files=False,
                help="Starting image for the network, will be resized to fit specified dimensions",
            )
            # Convert from UploadedFile object to PIL Image
            if init_image is not None:
                init_image: Image.Image = Image.open(init_image).convert(
                    "RGB"
                )  # just to be sure
        else:
            init_image = None

        use_image_prompts = st.sidebar.checkbox(
            "Add image prompt(s)",
            value=defaults["use_image_prompts"],
            help="Check to add image prompt(s), conditions the network similar to the text prompt",
        )

        image_prompts_widget = st.sidebar.empty()
        if use_image_prompts is True:
            image_prompts = image_prompts_widget.file_uploader(
                "Upload image prompts(s)",
                type=["png", "jpeg", "jpg"],
                accept_multiple_files=True,
                help="Image prompt(s) for the network, will be resized to fit specified dimensions",
            )
            # Convert from UploadedFile object to PIL Image
            if len(image_prompts) != 0:
                image_prompts = [Image.open(i).convert("RGB") for i in image_prompts]
        else:
            image_prompts = []

        radio = st.sidebar.radio(
            "Model weights",
            available_weights,
            index=default_weight_index,
            help="Choose which weights to load, trained on different datasets. Make sure the weights and configs are downloaded to `assets/` as per the README!",
        )

        use_mse_reg = st.sidebar.checkbox(
            "Use MSE regularization",
            value=defaults["use_mse_regularization"],
            help="Check to add MSE regularization",
        )
        mse_weight_widget = st.sidebar.empty()
        mse_weight_decay_widget = st.sidebar.empty()
        mse_weight_decay_steps = st.sidebar.empty()

        if use_mse_reg is True:
            mse_weight = mse_weight_widget.number_input(
                "MSE weight",
                value=defaults["mse_weight"],
                # min_value=0.0, # leave this out to allow creativity
                step=0.05,
                help="Set weights for MSE regularization",
            )
            mse_weight_decay = mse_weight_decay_widget.number_input(
                "Decay MSE weight by ...",
                value=defaults["mse_weight_decay"],
                # min_value=0.0, # leave this out to allow creativity
                step=0.05,
                help="Subtracts MSE weight by this amount at every step change. MSE weight change stops at zero",
            )
            mse_weight_decay_steps = mse_weight_decay_steps.number_input(
                "... every N steps",
                value=defaults["mse_weight_decay_steps"],
                min_value=0,
                step=1,
                help="Number of steps to subtract MSE weight. Leave zero for no weight decay",
            )
        else:
            mse_weight = 0
            mse_weight_decay = 0
            mse_weight_decay_steps = 0

        use_tv_loss = st.sidebar.checkbox(
            "Use TV loss regularization",
            value=defaults["use_tv_loss_regularization"],
            help="Check to add MSE regularization",
        )
        tv_loss_weight_widget = st.sidebar.empty()
        if use_tv_loss is True:
            tv_loss_weight = tv_loss_weight_widget.number_input(
                "TV loss weight",
                value=defaults["tv_loss_weight"],
                min_value=0.0,
                step=1e-4,
                help="Set weights for TV loss regularization, which encourages spatial smoothness. Ref: https://github.com/jcjohnson/neural-style/issues/302",
                format="%.1e",
            )
        else:
            tv_loss_weight = 0

        use_scrolling_zooming = st.sidebar.checkbox(
            "Scrolling/zooming transforms",
            value=False,
            help="At fixed intervals, move the generated image up/down/left/right or zoom in/out",
        )
        translation_x_widget = st.sidebar.empty()
        translation_y_widget = st.sidebar.empty()
        rotation_angle_widget = st.sidebar.empty()
        zoom_factor_widget = st.sidebar.empty()
        transform_interval_widget = st.sidebar.empty()
        if use_scrolling_zooming is True:
            translation_x = translation_x_widget.number_input(
                "Translation in X", value=0, min_value=0, step=1
            )
            translation_y = translation_y_widget.number_input(
                "Translation in y", value=0, min_value=0, step=1
            )
            rotation_angle = rotation_angle_widget.number_input(
                "Rotation angle (degrees)",
                value=0.0,
                min_value=0.0,
                max_value=360.0,
                step=0.05,
                format="%.2f",
            )
            zoom_factor = zoom_factor_widget.number_input(
                "Zoom factor",
                value=1.0,
                min_value=0.1,
                max_value=10.0,
                step=0.02,
                format="%.2f",
            )
            transform_interval = transform_interval_widget.number_input(
                "Iterations per frame",
                value=10,
                min_value=0,
                step=1,
                help="Note: Will multiply by num steps above!",
            )
        else:
            translation_x = 0
            translation_y = 0
            rotation_angle = 0
            zoom_factor = 1
            transform_interval = 1

        use_cutout_augmentations = st.sidebar.checkbox(
            "Use cutout augmentations",
            value=True,
            help="Adds cutout augmentatinos in the image generation process. Uses up to additional 4 GiB of GPU memory. Greatly improves image quality. Toggled on by default.",
        )

        # End of form

    status_text = st.empty()
    status_text.text("Pending input prompt")

    vid_display_slot = st.empty()
    debug_slot = st.empty()

    if "prev_im" in st.session_state:
        im_display_slot.image(
            st.session_state["prev_im"], caption="Output image", output_format="PNG"
        )
        
    with st.expander("Target Directory"):
        # Separate form for making new output dir
        with st.form("form-output-dir"):
            new_written_outputdir = st.text_input("Name", value=outputdir)
    
            if(st.form_submit_button("Create")):
                os.mkdir(Path("new-output", new_written_outputdir))
                outputdir = new_written_outputdir

    # Create dir if needed
    realdir = getOutputPath(outputdir)
    if not realdir.exists():
        realdir.mkdir()
        
    with st.expander("Previous Runs"):        
        gallerydir = st.select_slider("Gallery Dir", listDirs(realdir.parent), value=realdir.name)
        st.write(gallerydir)
        cols = st.columns(3)
        col = 0
        for subdir in getOutputPath(gallerydir).iterdir():
            if subdir.is_dir():
                with cols[col]:
                    st.image(str(subdir/"output.PNG"), caption=subdir.name)
                    if st.button("choose", key=subdir.name):
                        print(f"you chose me! ({subdir.name})")
                    # increment and wrap col
                    col = (col + 1) % len(cols)
        # Previous Runs
        gallery_0 = st.empty()
        gallery_1 = st.empty()
        gallery_2 = st.empty()

    # with st.expander("Expand for README"):
    #     with open("README.md", "r") as f:
    #         # Preprocess links to redirect to github
    #         # Thank you https://discuss.streamlit.io/u/asehmi, works like a charm!
    #         # ref: https://discuss.streamlit.io/t/image-in-markdown/13274/8
    #         markdown_links = [str(i) for i in Path("docs/").glob("*.md")]
    #         images = [str(i) for i in Path("docs/images/").glob("*")]
    #         readme_lines = f.readlines()
    #         readme_buffer = []
    # 
    #         for line in readme_lines:
    #             for md_link in markdown_links:
    #                 if md_link in line:
    #                     line = line.replace(
    #                         md_link,
    #                         "https://github.com/tnwei/vqgan-clip-app/tree/main/"
    #                         + md_link,
    #                     )
    # 
    #             readme_buffer.append(line)
    #             for image in images:
    #                 if image in line:
    #                     st.markdown(" ".join(readme_buffer[:-1]))
    #                     st.image(
    #                         f"https://raw.githubusercontent.com/tnwei/vqgan-clip-app/main/{image}"
    #                     )
    #                     readme_buffer.clear()
    #         st.markdown(" ".join(readme_buffer))
    # 
    # with st.expander("Expand for CHANGELOG"):
    #     with open("CHANGELOG.md", "r") as f:
    #         st.markdown(f.read())

    if submitted:
        # debug_slot.write(st.session_state) # DEBUG
        status_text.text("Loading weights ...")
        generate_image(
            # Inputs
            text_input=text_input,
            vqgan_ckpt=radio,
            num_steps=num_steps,
            image_x=int(image_x),
            image_y=int(image_y),
            seed=int(seed) if set_seed is True else None,
            init_image=init_image,
            image_prompts=image_prompts,
            continue_prev_run=continue_prev_run,
            mse_weight=mse_weight,
            mse_weight_decay=mse_weight_decay,
            mse_weight_decay_steps=mse_weight_decay_steps,
            use_scrolling_zooming=use_scrolling_zooming,
            translation_x=translation_x,
            translation_y=translation_y,
            rotation_angle=rotation_angle,
            zoom_factor=zoom_factor,
            transform_interval=transform_interval,
            use_cutout_augmentations=use_cutout_augmentations,
        )

        vid_display_slot.video("temp.mp4")
        # debug_slot.write(st.session_state) # DEBUG
