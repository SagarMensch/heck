@echo off
cd /d C:\Users\aigcp_gpuadmin\Downloads\LICRFP\LICF
echo start > train.log
python test_simple.py >> train.log 2>>train.log
echo done >> train.log