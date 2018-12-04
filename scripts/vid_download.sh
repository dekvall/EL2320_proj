#!/usr/bin/env bash

NAME=$1
URL=$2


youtube-dl -f 18 --output "out.mp4" $URL

ffmpeg -i "out.mp4" -r 5 -y -an $NAME".mp4"

