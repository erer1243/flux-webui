#!/usr/bin/env python3
try:
    import torch
    import os
    import sys
    import base64
    import gc
    from io import BytesIO
    from threading import Thread
    from queue import Queue
    from diffusers import FluxPipeline
    from bottle import get, post, run, request
    from time import time
except ImportError:
    me = os.path.realpath(__file__)
    dir = os.path.dirname(me)
    sys.exit(os.system(f"{dir}/venv/bin/python '{me}'"))

HUGGING_FACE_TOKEN = "FILL ME IN"
all_imgs = []
cur_imgs = []
status = "idle"
prompt_queue = Queue()


def img_thread_main():
    while True:
        try:
            gc.collect()
            img_thread_body()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)
            global status
            status = "idle"


def img_thread_body():
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        token=HUGGING_FACE_TOKEN,
    )
    pipe.enable_sequential_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    while True:
        prompt, side_length, batch_size, batch_count = prompt_queue.get()
        print("Processing prompt", prompt)
        gen_images(
            pipe,
            prompt,
            batch_size=batch_size,
            batch_count=batch_count,
            side_length=side_length,
        )

        global status
        status = "idle"


def update_progress(step, step_count, batch, batch_count):
    global status
    percent = 100 * (step_count * batch + step) / (step_count * batch_count)
    status = f'{int(percent)}% (step {step} of {step_count}, batch {batch+1} of {batch_count})'


def gen_images(
    pipe,
    prompt,
    batch_size=1,
    batch_count=1,
    side_length=1024,
    step_count=50,
    guidance_scale=3.5,
    max_sequence_length=512,
):
    global cur_imgs, all_imgs
    cur_imgs.clear()

    title = f"<h2>{prompt}</h2>"
    cur_imgs.append(title)
    all_imgs.append(title)

    update_progress(0, step_count, 0, batch_count)
    for batch in range(batch_count):
        images = pipe(
            batch_size * [prompt],
            height=side_length,
            width=side_length,
            guidance_scale=guidance_scale,
            num_inference_steps=step_count,
            max_sequence_length=max_sequence_length,
            callback_on_step_end=lambda _0, step, _1, _2, prompt=prompt: [
                update_progress(step, step_count, batch, batch_count),
                dict(),
            ][1],
        ).images
        image_tags = list(map(image_as_tag, images))
        all_imgs.extend(image_tags)
        cur_imgs.extend(image_tags)
        for img in images:
            n = 1
            while True:
                path = f"output/{int(time())}-{side_length}px-{prompt}-{n}.png"
                if os.path.exists(path):
                    n += 1
                else:
                    img.save(path)
                    break


def image_as_tag(img):
    img_io = BytesIO()
    img.save(img_io, "PNG")
    img_io.seek(0)
    img_b64 = base64.b64encode(img_io.read()).decode("ascii")
    return f'<img src="data:image/png;charset=utf-8;base64,{img_b64}">'


@get("/")
def index():
    return """
        <form action="/gen" method="post">
            Prompt: <input name="prompt" type="text" />
            <br>
            Side Length: <input name="side_length" type="text" value="512" />
            <br>
            Batch Size: <input name="batch_size" type="text" value="1" />
            <br>
            Batch Count: <input name="batch_count" type="text" value="1" />
            <br>
            <input value="Submit" type="submit" />
        </form>
        <a href="/all">All images</a>
        <iframe src="/cur" style="width:100%;border:none;overflow:hidden;" onload="this.style.height=(this.contentWindow.document.body.scrollHeight + 20) + 'px';"></iframe>
    """


@get("/cur")
def cur():
    body = f"""
        <p>Status: {status}</p>
        {"<br>".join(cur_imgs)}
    """
    if status != "idle":
        body += """
            <script type="text/javascript">
                setTimeout(() => {document.location = "/cur"}, 30000)
            </script>
        """
    return body


@get("/all")
def all():
    return "<br>".join(all_imgs)


@post("/gen")
def gen():
    prompt = request.forms.get("prompt").strip()
    side_length = int(request.forms.get("side_length"))
    batch_size = int(request.forms.get("batch_size"))
    batch_count = int(request.forms.get("batch_count"))
    prompt_queue.put((prompt, side_length, batch_size, batch_count))
    return f"""
        <p>Received prompt: {request.forms.get('prompt')}</p>
        <script type="text/javascript">
            setTimeout(() => {{document.location = "/"}}, 3000)
        </script>
    """


os.makedirs("output", exist_ok=True)
Thread(target=img_thread_main, daemon=True).start()
run(host="0.0.0.0", port=8080, debug=True)
