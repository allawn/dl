#!/usr/bin/env bash
g++ -std=c++11 book.cc -o book -I/opt/tensorflow/include -L/opt/tensorflow/lib -ltensorflow_cc