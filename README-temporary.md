### Download Essential Files

- Download all files in this [google drive](https://drive.google.com/drive/folders/1OiiwSi8aYb0w6fh8rpTMZS7MupO3pE0S) folder. There are two files:

    - One is `VitaLITy-2.0.0.json`, This is the paper dataset. After downloading, copy this inside the `data` folder of this repository.
    - The other is `data.db.zip`, which contains information about the chromadb data set. I use this to retrieve data when performing enhancements. After downloading, please unzip it and place the `data.db` folder in the project root folder. The directory structure is as shown below.

![1.png](./images/1.png)

### Setup

- Install dependencies: `pip install -r ./requirements.txt` (can make virtual environment if needed)
- Add OPENAI key to .env file. Because this is sensitive, check email/slack.
- Start: `python server.py`. Wait until you see "created query index for ada". Then you can setup and run the frontend repository.