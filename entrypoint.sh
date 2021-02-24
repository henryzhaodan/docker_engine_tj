#!/bin/bash
# Variables
INIT_FLAG=INITIALIZED 
DIS_UPDATE_FLAG=DIS_UPDATE
MODEL_ARCHIVE_FILE=/tmp/model_archive.tar
MODEL_UNPACK_PATH=.

# Initial work dir
if [ ! -f "${INIT_FLAG}" ]
then
  echo "Initial work directory..." >&1
  cd "kernel/utils/external" && make && cd -
  touch "${INIT_FLAG}"
fi

# Update models
if [ ! -f "${DIS_UPDATE_FLAG}" ]
then
  echo "Updating models..." >&1
  if [[ -n "${MODEL_FILE_LOCAL_PATH}" ]]
  then
    MODEL_ARCHIVE_FILE=${MODEL_FILE_LOCAL_PATH}
  elif [[ -n "${MODEL_FILE_REMOTE_URL}" ]]
  then
    echo "Downloading from ${MODEL_FILE_URL}..." >&1
    wget -q -O "${MODEL_ARCHIVE_FILE}" -c "${MODEL_FILE_URL}"
    DOWNLOAD_FLAG=1
  else
    echo "Unable to download because both MODEL_FILE_LOCAL_FILE and MODEL_FILE_REMOTE_URL are undefined" >&2
    exit 1
  fi
  echo "Unpacking..." >&1
  tar -xf "${MODEL_ARCHIVE_FILE}" -C "${MODEL_UNPACK_PATH}/"
  touch "${DIS_UPDATE_FLAG}"
  if [[ -n "${DOWNLOAD_FLAG}" ]]
  then
    echo "Removing ${MODEL_ARCHIVE_FILE}..." >&1
    rm -f "${MODEL_ARCHIVE_FILE}"
  fi
fi

# Run python code
echo "Starting analysis engine..." >&1
/usr/bin/python3 main.py