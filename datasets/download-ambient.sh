#!/bin/bash

mkdir ambient
cd ambient

mkdir sad
cd sad
youtube-dl --extract-audio --audio-format mp3 https://www.youtube.com/playlist?list=PLr_rcAQaFAiW9hAMr6raz3OrzzrBLdM4X
cd ..

mkdir happy
cd happy
youtube-dl --extract-audio --audio-format mp3 https://www.youtube.com/playlist?list=PLr_rcAQaFAiVP68yiz2V6XgPSP0WDb8aI
cd ..

mkdir space
cd space
youtube-dl --extract-audio --audio-format mp3 https://www.youtube.com/playlist?list=PLr_rcAQaFAiWX31AeFn-S662zCXsFMQ4u
cd ..

mkdir creepy
cd creepy
youtube-dl --extract-audio --audio-format mp3 https://www.youtube.com/playlist?list=PLr_rcAQaFAiXVjaAl-V7UV8F7QGl59jWn
cd ..

cd ..
