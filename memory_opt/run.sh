#!/bin/bash
#  GPU memory operation demo.
#  Author: kevin.xie
#  Email: kaiyuanxie@yeah.net


set -e
current_path=$(cd `dirname $0`; pwd)
make
echo "Run all demo:"

if [ ! -f ${current_path}/testHost2Device ]
  then
    echo "testHost2Device exe file not found!"
    exit 1
  fi
./testHost2Device
echo "[Next]"

if [ ! -f ${current_path}/testDevice2Device ]
  then
    echo "testDevice2Device exe file not found!"
    exit 1
  fi
./testDevice2Device 
echo "[Next]"

if [ ! -f ${current_path}/testSharedMemory ]
  then
    echo "testSharedMemory exe file not found!"
    exit 1
  fi
./testSharedMemory 
echo "[Next]"

if [ ! -f ${current_path}/testZeroCopy ]
  then
    echo "testZeroCopy exe file not found!"
    exit 1
  fi
./testZeroCopy
exit 0

