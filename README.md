# VentBot
> This repository contains the VentBot project built using [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/). Deployed in Flask.

Capstone project. A friendly chatbot that helps humans identify their emotions.

## Run it localy from scratch
- Fork the repository from the [original repo](https://github.com/laisbsc/VentBot/keras_chatbot)
- then, from your own GitHub account, `git clone` your copied repository
- `cd` into the repository
- create a new virtual environment for the project
	- Using pip run `python3 -m venv <project_name>`
	- followed by `source <project_name>/bin/activate` to activate your virtual environment
- install the project dependencies with `pip install -r requirements.txt`.
- Now, you will need to install the `wordnet` corpora using the command `python -m nltk.downloader all`.

  
Want to chat with Ada right now on your local computer?
- Run `python train_chatbot.py` (File responsible for patters and bot responses)
- Then, `python chatbot_function.py` to train the customised dataset.
- Finally, run `python app.py` to execute the final running script and deploy the app locally 
Ada will be available at the [localhost](http://127.0.0.1:5000/) on your browser.

Start chatting!



## Thinking about contributing to this repository?
Head over to the [Contributing](./CONTRIBUTING.md) page and check out the detailes instructions.