@ECHO OFF
ECHO START COMPRESSING THE VIDEOS
FOR /F "tokens=*" %%G IN  ('dir /b *.mp4') DO ffmpeg -i "%%G" -s 1280x720 -b:v 4000k -vcodec libx264 -acodec copy "../HD_mp4/%%~nG.mp4" -vstats_file "../HD_mp4/stats/%%~nG.log"
ECHO DONE COMPRESSING ALL THE VIDEOS
