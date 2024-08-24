# flux-webui
flux image generation network basic &amp; easy webui

```sh
# download
git clone https://github.com/erer1243/flux-webui
cd flux-webui

# install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pacman -S python-pytorch-opt-cuda # replace with an appropriate pytorch install method

# edit webserver.py, insert your hugging face token near the top of the script
$EDITOR webserver.py

# run
python webserver.py
```
