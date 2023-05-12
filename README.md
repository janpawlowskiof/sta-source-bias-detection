# Source detection app

## Usage

1. Build image `docker build -t sta-source-detection .`

2. Run the image. Make sure to mount the models to `\source-detection\models`: 
  `docker run -d --mount type=bind,source=...,target=\source-detection\models sta-source-detection:latest `

