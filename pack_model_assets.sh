#!/bin/bash
cd src
find ./ -name "*.pth" -o \
        -name "*.pkl" -o \
        -name "*.dat" -o \
        -name "*.tar" -o \
        -name "*.caffemodel" \
        | tar -cf ../model_archive.tar -T -